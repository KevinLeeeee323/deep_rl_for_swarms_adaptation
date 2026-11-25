"""Real Windows spawn, stable metrics, reuse safety and worker configuration."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from deep_rl_for_swarms.mappo_tools import _scipy_dll_directory
from deep_rl_for_swarms.mappo_eval_pool import evaluation_workers, evaluate_candidates


class ParallelEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = _scipy_dll_directory()
        cls.context.__enter__()
        cls.temporary = tempfile.TemporaryDirectory(prefix='mappo_parallel_tests_')
        cls.root = Path(cls.temporary.name)
        from deep_rl_for_swarms.run_multiagent_mappo import train
        from deep_rl_for_swarms.common.mappo_evaluation import make_scenarios
        score = dict(mean_episode_return=0., std_episode_return=0., success_rate=0.,
                     mean_minimum_distance=1., mean_final_distance=2.)
        with patch('deep_rl_for_swarms.rl_algo.mappo._evaluate', return_value=score):
            cls.run_dir = Path(train(8, str(cls.root), rollout_steps=4, minibatch_steps=4, epochs=1,
                                 eval_interval=4, checkpoint_interval=4, eval_episodes=1, test=False))
        config = json.loads((cls.run_dir / 'config.json').read_text())
        cls.suite = make_scenarios(config['environment_settings'], 30000, 5)
        for index, scene in enumerate(cls.suite['scenarios']):
            radius = 2.1 + .02 * index
            center = 30. + index
            scene['initial_state'] = {
                'pursuer_states': [[center + radius * np.cos(i * 2 * np.pi / 10),
                                    center + radius * np.sin(i * 2 * np.pi / 10), float(i)]
                                   for i in range(10)], 'evader_states': [[center, center]]}
        cls.suite_path = cls.root / 'suite.json'
        cls.suite_path.write_text(json.dumps(cls.suite))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        cls.context.__exit__(None, None, None)

    def test_spawn_matches_serial_reports_in_original_order_and_preserves_parent_state(self):
        from deep_rl_for_swarms.mappo_tools import discover_checkpoints
        sources = discover_checkpoints(self.run_dir)
        before_environment = dict(os.environ)
        before_random = np.random.get_state()
        serial = list(evaluate_candidates(sources, self.suite, 1))
        parallel = list(evaluate_candidates(sources, self.suite, 2))
        by_checkpoint = {source['checkpoint']: reports for source, reports, _ in serial}
        self.assertEqual(len(parallel), 2)
        for source, reports, elapsed in parallel:
            self.assertEqual(reports, by_checkpoint[source['checkpoint']])
            self.assertEqual([report['scenario'] for report in reports],
                             [scene['id'] for scene in self.suite['scenarios']])
            self.assertGreater(elapsed, 0.)
        self.assertEqual(before_environment, dict(os.environ))
        # The low-level serial evaluator constructs an environment, so only the
        # public API promises preservation across that construction (tested below).
        np.random.set_state(before_random)

    def test_partial_legacy_report_reuse_matches_fresh_ranking_and_never_changes_source(self):
        from deep_rl_for_swarms.mappo_tools import test_all_checkpoints, test_checkpoint
        from deep_rl_for_swarms.common.mappo_evaluation import atomic_json
        original = test_all_checkpoints(self.run_dir, test_episodes=5, test_scenarios=str(self.suite_path),
                                        output_dir=self.root / 'original', workers=1)
        original_dir = Path(original['output_dir'])
        metadata = original['metadata']
        metadata.update(status='running', results=metadata['results'][:1])
        # Legacy reports have no execution/reuse metadata.
        metadata.pop('execution')
        atomic_json(original_dir / 'metadata.json', metadata)
        before = {path.name: path.read_bytes() for path in original_dir.iterdir()}
        random_state = np.random.get_state()
        resumed = test_all_checkpoints(
            self.run_dir, test_episodes=5, test_scenarios=str(self.suite_path), workers=2,
            output_dir=self.root / 'reused', reuse_report=original_dir)
        after_random = np.random.get_state()
        self.assertEqual(random_state[0], after_random[0])
        np.testing.assert_array_equal(random_state[1], after_random[1])
        self.assertEqual(random_state[2:], after_random[2:])
        self.assertEqual(len(resumed['metadata']['reused_checkpoints']), 1)
        self.assertEqual(resumed['metadata']['status'], 'complete')
        for first, second in zip(original['rows'], resumed['rows']):
            self.assertEqual({k: v for k, v in first.items() if k != 'elapsed_seconds'},
                             {k: v for k, v in second.items() if k != 'elapsed_seconds'})
        self.assertEqual(before, {path.name: path.read_bytes() for path in original_dir.iterdir()})
        self.assertEqual(len(list(Path(resumed['output_dir']).iterdir())), 3)
        single = test_checkpoint(self.run_dir / '4', test_episodes=5, test_scenarios=str(self.suite_path),
                                 output_dir=self.root / 'single_reused', workers=2,
                                 reuse_report=resumed['output_dir'])
        self.assertEqual(single['metadata']['execution']['workers'], 0)
        self.assertEqual(len(single['rows']), 1)
        altered = json.loads(json.dumps(self.suite))
        altered['scenarios'][0]['initial_state']['evader_states'][0][0] += .1
        different = self.root / 'different.json'
        atomic_json(different, altered)
        with self.assertRaisesRegex(ValueError, 'same scenarios'):
            test_all_checkpoints(self.run_dir, test_episodes=5, test_scenarios=str(different),
                                 output_dir=self.root / 'must_not_exist', reuse_report=original_dir)
        self.assertFalse((self.root / 'must_not_exist').exists())
        bad_metadata = dict(metadata, actions='stochastic')
        atomic_json(original_dir / 'metadata.json', bad_metadata)
        with self.assertRaisesRegex(ValueError, 'incompatible actions'):
            test_all_checkpoints(self.run_dir, test_episodes=5, test_scenarios=str(self.suite_path),
                                 output_dir=self.root / 'must_not_exist', reuse_report=original_dir)
        self.assertFalse((self.root / 'must_not_exist').exists())

    def test_worker_counts_are_validated_and_resume_option_reaches_training(self):
        from deep_rl_for_swarms.mappo_tools import resolve_checkpoint, resume_training, test_all_checkpoints
        self.assertEqual(evaluation_workers(), min(8, os.cpu_count() or 1))
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                evaluation_workers(invalid)
        source = resolve_checkpoint(self.run_dir / '8')
        source['config']['test']['workers'] = 1
        with patch('deep_rl_for_swarms.mappo_tools.resolve_checkpoint', return_value=source), \
                patch('deep_rl_for_swarms.run_multiagent_mappo.train', return_value='new_run') as train:
            # Previous runs may record a lower worker count. New invocations use
            # the current default unless explicitly overridden on this command.
            resume_training(self.run_dir / '8', 4)
            self.assertEqual(train.call_args[1]['test_workers'], evaluation_workers())
            resume_training(self.run_dir / '8', 4, test_workers=3)
            self.assertEqual(train.call_args[1]['test_workers'], 3)
        with self.assertRaises(ValueError):
            test_all_checkpoints(self.run_dir, workers=0, output_dir=self.root / 'invalid')
        self.assertFalse((self.root / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
