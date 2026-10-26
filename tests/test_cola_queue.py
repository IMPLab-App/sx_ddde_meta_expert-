"""Operational tests use isolated files and tiny child processes, never training."""
import contextlib
import copy
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

import cola_two_kl_queue as queue


class TestQueue(unittest.TestCase):
    def test_second_queue_is_rejected(self):
        lock = queue.LOGS / 'queue.lock'
        with lock.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run(['bash', str(queue.ROOT / 'run_cola_two_kl_24.sh'), '--dry-run'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 75)
            self.assertIn('Another', result.stderr)

    def test_child_exit_codes_and_existing_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); log = root / 'child.log'
            task = dict(variant='test', run_name='isolated', seed=2)
            with patch.object(queue, 'LOGS', root), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'exit 7'):
                    queue.run_stage(task, 'train', [sys.executable, '-c', 'print("failure");raise SystemExit(7)'], log)
                self.assertIn('exit_code=7', log.read_text())
                self.assertEqual(json.loads((root/'current.json').read_text())['status'], 'failed')
                with self.assertRaises(FileExistsError):
                    queue.run_stage(task, 'train', [sys.executable, '-c', 'raise SystemExit(0)'], log)

    def test_failure_stops_before_second_task(self):
        tasks = [dict(variant='full', run_name=f'isolated{i}', seed=2, model_dir=f'model{i}', config='test.yaml') for i in range(24)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest = root/'manifest.json'; manifest.write_text(json.dumps(dict(tasks=tasks)))
            calls = []
            def child(task, stage, *args):
                calls.append((task['run_name'], stage))
                raise RuntimeError('simulated train failure')
            with patch.object(queue, 'ROOT', root), patch.object(queue, 'LOGS', root/'logs'), patch.object(queue, 'MANIFEST', manifest), \
                 patch.object(queue, 'validate_task'), patch.object(queue, 'completed', return_value=False), \
                 patch.object(queue, 'run_stage', side_effect=child), patch.object(queue.subprocess, 'run'), \
                 patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    queue.run_queue()
            self.assertEqual(calls, [('isolated0', 'train')])
            self.assertEqual(json.loads((root/'logs/queue.json').read_text())['status'], 'failed')

    def test_safe_skip_rejects_changed_artifacts_and_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); model=root/'models'; model.mkdir()
            state=dict(model={'x':torch.ones(1)},ema_model={'x':torch.ones(1)},optimizer={},best_it=2,epoch=1,it=4,
                       cola_two_kl={'tau_dyn':torch.tensor(2.)})
            torch.save(state,model/'latest_model.pth'); torch.save(state,model/'model_best.pth')
            evaluation=model/'evaluation'; evaluation.mkdir()
            split=dict(top1_accuracy=.5,balanced_accuracy=.5,macro_precision=.5,macro_recall=.5,macro_f1=.5)
            for key in ('confusion_matrix_csv','confusion_matrix_png'):
                p=evaluation/key;p.write_text('test artifact');split[key]=str(p)
            result=dict(complete=True,checkpoint=str(model/'model_best.pth'),heldout=split,balanced=split,imbalanced=split)
            (evaluation/'evaluation.json').write_text(json.dumps(result))
            task=dict(model_dir='models',variant='full',num_train_iter=4,config_sha256='isolated-test')
            with patch.object(queue, 'ROOT', root):
                self.assertFalse(queue.completed(task))
                marker=queue.validate_outputs(task); queue.atomic_json(model/'completed.json',marker)
                self.assertTrue(queue.completed(task))
                changed=copy.deepcopy(state);changed['model']['x'].add_(1);torch.save(changed,model/'latest_model.pth')
                with self.assertRaisesRegex(RuntimeError, 'fingerprints'):
                    queue.completed(task)
                changed['it']=3;torch.save(changed,model/'latest_model.pth')
                with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                    queue.completed(task)


if __name__ == '__main__':
    unittest.main()
