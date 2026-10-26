"""Raw 1-D vibration-signal dataset loader."""

import copy
import os
import random

import numpy as np
import torch

from .datasetbase import BasicDataset
from semilearn.datasets.utils import split_labeled_unlabeled_data


# DATASET_FILENAME = "raw_signal_dataset.npz"
DATASET_FILENAME = "raw_signal_dataset_G_1024.npz"


class SignalTransform:
    """Convert a signal to ``[1, length]`` and optionally augment it."""

    # BasicDataset uses this marker to avoid converting a NumPy signal to PIL.
    accepts_ndarray = True

    def __init__(self, mean, std, strength="none"):
        if strength not in {"none", "weak", "strong"}:
            raise ValueError(f"Unknown signal augmentation strength: {strength}")
        self.mean = float(mean)
        self.std = max(float(std), 1e-8)
        self.strength = strength

    def __call__(self, signal):
        x = torch.as_tensor(np.asarray(signal), dtype=torch.float32).flatten().clone()
        x = (x - self.mean) / self.std

        if self.strength == "none":
            return x.unsqueeze(0)

        if self.strength == "weak":
            max_shift = max(1, x.numel() // 100)
            gain_low, gain_high = 0.95, 1.05
            noise_std = 0.005
            max_mask = 0
        else:
            max_shift = max(1, x.numel() // 20)
            gain_low, gain_high = 0.80, 1.20
            noise_std = 0.05
            max_mask = max(1, x.numel() // 10)

        shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        gain = float(torch.empty(1).uniform_(gain_low, gain_high).item())
        x = torch.roll(x, shifts=shift) * gain
        x = x + torch.randn_like(x) * noise_std

        if max_mask:
            mask_length = int(torch.randint(1, max_mask + 1, (1,)).item())
            mask_start = int(torch.randint(0, x.numel() - mask_length + 1, (1,)).item())
            x[mask_start:mask_start + mask_length] = 0.0

        return x.unsqueeze(0)


def load_raw_signal_data(data_dir="./data"):
    """Load and validate signals, labels, and class names from the NPZ file."""
    dataset_path = os.path.join(data_dir, DATASET_FILENAME)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(
            f"Raw signal dataset not found: {dataset_path}. "
            f"Expected '{DATASET_FILENAME}' directly under data_dir."
        )

    with np.load(dataset_path, allow_pickle=False) as dataset:
        required_keys = {"data", "labels", "class_names"}
        missing_keys = required_keys.difference(dataset.files)
        if missing_keys:
            raise KeyError(
                f"{dataset_path} is missing required arrays: {sorted(missing_keys)}"
            )
        data = np.asarray(dataset["data"], dtype=np.float32)
        targets = np.asarray(dataset["labels"], dtype=np.int64)
        class_names = [str(name) for name in dataset["class_names"].tolist()]

    if data.ndim != 2:
        raise ValueError(f"Expected data shape [samples, length], got {data.shape}")
    if targets.ndim != 1 or len(targets) != len(data):
        raise ValueError(
            f"labels must have shape [{len(data)}], got {targets.shape}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"{dataset_path} contains NaN or infinite signal values")

    unique_labels = np.unique(targets)
    expected_labels = np.arange(len(class_names), dtype=np.int64)
    if not np.array_equal(unique_labels, expected_labels):
        raise ValueError(
            "labels must be contiguous and match class_names: "
            f"expected {expected_labels.tolist()}, got {unique_labels.tolist()}"
        )

    return data, targets, class_names


def get_raw_signal_class_names(data_dir="./data"):
    """Return class names in the exact label order stored in the NPZ file."""
    dataset_path = os.path.join(data_dir, DATASET_FILENAME)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Raw signal dataset not found: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as dataset:
        if "class_names" not in dataset.files:
            raise KeyError(f"{dataset_path} is missing required array: class_names")
        return [str(name) for name in dataset["class_names"].tolist()]


def _stratified_split(targets, num_classes, seed):
    """Create reproducible 70/10/20 train/eval/test splits per class."""
    rng = np.random.RandomState(seed)
    split_indices = {"train": [], "eval": [], "test": []}

    for class_id in range(num_classes):
        class_indices = np.flatnonzero(targets == class_id).copy()
        if class_indices.size == 0:
            raise ValueError(f"Class {class_id} has no samples")
        rng.shuffle(class_indices)

        train_end = int(class_indices.size * 0.7)
        eval_end = train_end + int(class_indices.size * 0.1)
        split_indices["train"].extend(class_indices[:train_end])
        split_indices["eval"].extend(class_indices[train_end:eval_end])
        split_indices["test"].extend(class_indices[eval_end:])

    result = {}
    for split_name, indices in split_indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        result[split_name] = indices
    return result


def get_raw_signal(
    args,
    name="raw_signal",
    data_dir="./data",
    include_lb_to_ulb=False,
    return_test=False,
):
    """Build labeled, unlabeled, validation, and test signal datasets."""
    all_data, all_targets, class_names = load_raw_signal_data(data_dir)

    if args.num_classes != len(class_names):
        raise ValueError(
            f"num_classes={args.num_classes}, but {DATASET_FILENAME} contains "
            f"{len(class_names)} classes: {class_names}"
        )

    split_indices = _stratified_split(all_targets, args.num_classes, args.seed)
    train_idx = split_indices["train"]
    eval_idx = split_indices["eval"]
    test_idx = split_indices["test"]

    train_data, train_targets = all_data[train_idx], all_targets[train_idx]
    eval_data, eval_targets = all_data[eval_idx], all_targets[eval_idx]
    test_data, test_targets = all_data[test_idx], all_targets[test_idx]

    # Normalization statistics are calculated from the training split only.
    signal_mean = float(train_data.mean(dtype=np.float64))
    signal_std = float(train_data.std(dtype=np.float64))
    transform_weak = SignalTransform(signal_mean, signal_std, strength="weak")
    transform_strong = SignalTransform(signal_mean, signal_std, strength="strong")
    transform_eval = SignalTransform(signal_mean, signal_std, strength="none")

    # Keep index caches separate from the legacy image dataset directory.
    split_args = copy.copy(args)
    split_args.index_dataset = "raw_signal"
    lb_idx, ulb_idx, lb_clean_idx, lb_noise_idx = split_labeled_unlabeled_data(
        split_args,
        train_data,
        train_targets,
        num_classes=args.num_classes,
        lb_num_labels=args.num_labels,
        ulb_num_labels=args.ulb_num_labels,
        lb_imbalance_ratio=args.lb_imb_ratio,
        ulb_imbalance_ratio=args.ulb_imb_ratio,
        noise_ratio=args.noise_ratio,
        noise_per_class=args.noise_per_class,
        lb_imb_type=args.lb_imb_type,
        ulb_imb_type=args.ulb_imb_type,
        num_steps=args.num_steps,
        include_lb_to_ulb=include_lb_to_ulb,
    )
    lb_idx = np.asarray(lb_idx, dtype=np.int64)
    ulb_idx = np.asarray(ulb_idx, dtype=np.int64)
    lb_clean_idx = np.asarray(lb_clean_idx, dtype=np.int64)
    lb_noise_idx = np.asarray(lb_noise_idx, dtype=np.int64)

    noised_targets = train_targets.copy()
    lb_count = [int(np.sum(train_targets[lb_idx] == c)) for c in range(args.num_classes)]
    ulb_count = [int(np.sum(train_targets[ulb_idx] == c)) for c in range(args.num_classes)]
    lb_clean_count = [int(np.sum(train_targets[lb_clean_idx] == c)) for c in range(args.num_classes)]
    lb_noise_count = [int(np.sum(train_targets[lb_noise_idx] == c)) for c in range(args.num_classes)]

    p_noise = np.zeros((args.num_classes, args.num_classes), dtype=np.float64)
    for source_class in range(args.num_classes):
        denominator = sum(lb_count) - lb_count[source_class]
        if denominator:
            for target_class in range(args.num_classes):
                if source_class != target_class:
                    p_noise[source_class, target_class] = lb_count[target_class] / denominator

    for index in lb_noise_idx:
        source_class = int(train_targets[index])
        if args.noise_type == "sym":
            noised_targets[index] = (
                random.randint(1, args.num_classes - 1) + source_class
            ) % args.num_classes
        elif args.noise_type == "asym":
            noised_targets[index] = np.random.choice(
                args.num_classes, p=p_noise[source_class]
            )
        elif args.noise_type == "circle":
            noised_targets[index] = (source_class + 1) % args.num_classes
        else:
            raise ValueError(f"Unsupported noise_type: {args.noise_type}")

    new_lb_noise_count = [
        int(np.sum(noised_targets[lb_noise_idx] == c)) for c in range(args.num_classes)
    ]
    split_count = {
        split_name: [
            int(np.sum(all_targets[indices] == c)) for c in range(args.num_classes)
        ]
        for split_name, indices in split_indices.items()
    }

    print(
        f"Raw signals loaded from {os.path.join(data_dir, DATASET_FILENAME)}: "
        f"{len(all_data)} samples, length {all_data.shape[1]}, classes {class_names}"
    )
    print(
        "split count: train {}, eval {}, test {}".format(
            split_count["train"], split_count["eval"], split_count["test"]
        )
    )
    print(f"normalization: mean={signal_mean:.6g}, std={signal_std:.6g}")
    print(f"lb count: {lb_count}")
    print(f"ulb count: {ulb_count}")

    lb_dset = BasicDataset(
        train_data[lb_idx],
        train_targets[lb_idx],
        noised_targets[lb_idx],
        args.num_classes,
        False,
        weak_transform=transform_weak,
        strong_transform=transform_strong,
        onehot=False,
    )
    ulb_dset = BasicDataset(
        train_data[ulb_idx],
        train_targets[ulb_idx],
        None,
        args.num_classes,
        True,
        weak_transform=transform_weak,
        strong_transform=transform_strong,
        onehot=False,
    )
    eval_dset = BasicDataset(
        eval_data,
        eval_targets,
        None,
        args.num_classes,
        False,
        weak_transform=transform_eval,
        strong_transform=None,
        onehot=False,
    )
    test_dset = BasicDataset(
        test_data,
        test_targets,
        None,
        args.num_classes,
        False,
        weak_transform=transform_eval,
        strong_transform=None,
        onehot=False,
    )

    count_message = {
        "lb_count": lb_count,
        "ulb_count": ulb_count,
        "lb_clean_count": lb_clean_count,
        "lb_noise_count": lb_noise_count,
        "new_lb_noise_count": new_lb_noise_count,
    }
    if return_test:
        return lb_dset, ulb_dset, eval_dset, test_dset, count_message
    return lb_dset, ulb_dset, eval_dset, count_message
