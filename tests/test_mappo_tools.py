"""Checkpoint integration checks using tiny real rollouts and fixed eval scores.

Run with: python -m unittest discover -s tests -p test_mappo_tools.py
"""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from deep_rl_for_swarms.mappo_tools import (
    _scipy_dll_directory, resolve_checkpoint, resume_training, test_checkpoint)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob('*') if path.is_file()}


class CheckpointToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dll_directory = _scipy_dll_directory()
        cls.dll_directory.__enter__()
        import tensorflow as tf
        from deep_rl_for_swarms.run_multiagent_mappo import train
        cls.tf = tf
        cls.temporary = tempfile.TemporaryDirectory(prefix='mappo_checkpoint_tests_')
        cls.root = Path(cls.temporary.name)
        # Evaluations are fixed so best-checkpoint retention is deterministic;
        # graph construction, rollout, optimization and checkpoint IO are real.
        with patch('deep_rl_for_swarms.rl_algo.mappo._evaluate', return_value=(0.0, 0.0)):
            cls.source = Path(train(8, str(cls.root), rollout_steps=8,
                                    minibatch_steps=4, epochs=1, eval_episodes=1, test=False))
            cls.source_hashes = file_hashes(cls.source)
            cls.continued = Path(resume_training(
                cls.source / '8', 4, note='续训测试', learning_rate=1e-4))
            cls.chained = Path(resume_training(cls.continued / '12', 4, note='第二次续训'))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        cls.dll_directory.__exit__(None, None, None)

    def test_restores_every_variable_and_updates_actor_critic_and_adam(self):
        original = self.tf.train.NewCheckpointReader(str(self.source / '8' / 'model'))
        baseline = self.tf.train.NewCheckpointReader(str(self.continued / '8' / 'model'))
        final = self.tf.train.NewCheckpointReader(str(self.continued / '12' / 'model'))
        names = original.get_variable_to_shape_map()
        self.assertEqual(names, baseline.get_variable_to_shape_map())
        for name in names:
            np.testing.assert_array_equal(original.get_tensor(name), baseline.get_tensor(name),
                                          err_msg='Not restored: ' + name)
        for group in ['mappo_policy/pol/', 'central_critic/']:
            weights = [name for name in names if name.startswith(group) and '/Adam' not in name]
            self.assertTrue(any(not np.array_equal(original.get_tensor(name), final.get_tensor(name))
                                for name in weights), group + ' did not update')
        self.assertTrue(any(not np.array_equal(original.get_tensor(name), final.get_tensor(name))
                            for name in names if '/Adam' in name))

    def test_additional_budget_and_local_best_baseline(self):
        checkpoints = read_json(self.continued / 'checkpoints.json')
        self.assertEqual(checkpoints['best']['step'], 8)
        self.assertEqual(checkpoints['final']['step'], 12)
        with (self.continued / 'progress.csv').open() as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(int(rows[-1]['TimestepsSoFar']), 12)
        self.assertEqual(int(rows[-1]['TimestepsThisRun']), 4)
        self.assertEqual(read_json(self.continued / 'config.json')['num_timesteps'], 4)
        self.assertEqual(file_hashes(self.source), self.source_hashes)

    def test_inherits_settings_and_records_multistage_lineage(self):
        config = read_json(self.continued / 'config.json')
        continuation = config['continuation']
        self.assertEqual(continuation['source_step'], 8)
        self.assertEqual(continuation['additional_timesteps'], 4)
        self.assertEqual(continuation['target_total_timesteps'], 12)
        self.assertEqual(continuation['note'], '续训测试')
        self.assertEqual(config['training']['rollout_steps'], 8)
        self.assertEqual(config['training']['learning_rate'], 1e-4)
        self.assertEqual(continuation['training_overrides'],
                         {'learning_rate': {'from': 3e-4, 'to': 1e-4}})
        chained = read_json(self.chained / 'config.json')
        self.assertEqual(chained['training']['learning_rate'], 1e-4)
        self.assertEqual(chained['continuation']['target_total_timesteps'], 16)
        self.assertEqual(len(chained['continuation']['history']), 2)
        self.assertEqual(chained['continuation']['history'][0]['note'], '续训测试')

    def test_checkpoint_path_forms_and_run_selection(self):
        for path in [self.source / '8', self.source / '8' / 'model',
                     self.source / '8' / 'model.index', self.source / '8' / 'model.meta', self.source]:
            self.assertEqual(resolve_checkpoint(path)['step'], 8)
        self.assertEqual(resolve_checkpoint(self.continued, model='best')['step'], 8)
        self.assertEqual(resolve_checkpoint(self.continued, model='final')['step'], 12)

    def test_rejects_invalid_requests_before_creating_a_run(self):
        before = set(self.root.iterdir())
        with self.assertRaises(ValueError):
            resume_training(self.source, 0)
        with self.assertRaises(ValueError):
            resume_training(self.source, 4, test=True, test_seed=10_000)
        with self.assertRaises(ValueError):
            resume_training(self.source, 4, learning_rate=float('nan'))
        with self.assertRaises(ValueError):
            resolve_checkpoint(self.source / '999')
        with self.assertRaises(ValueError):
            test_checkpoint(self.source, animation_stride=0)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_existing_test_output_is_protected(self):
        directory = self.continued / 'test' / '12'
        directory.mkdir(parents=True, exist_ok=True)
        sentinel = directory / 'summary.json'
        sentinel.write_text('{"preserved": true}')
        with self.assertRaises(FileExistsError):
            test_checkpoint(self.continued)
        self.assertEqual(read_json(sentinel), {'preserved': True})


if __name__ == '__main__':
    unittest.main()
