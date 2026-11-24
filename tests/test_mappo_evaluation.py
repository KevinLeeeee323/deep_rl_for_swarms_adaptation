"""Metric definitions, retained checkpoints, and automatic batch selection."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from deep_rl_for_swarms.mappo_tools import _scipy_dll_directory

with _scipy_dll_directory():
    from deep_rl_for_swarms.common.mappo_evaluation import (
        evaluate_scenarios, load_scenarios, make_scenarios, rank_models, summarize, write_ranked_csv)


class ScriptedEnv:
    """Exercise capture threshold and the full 1024-step timeout cheaply."""
    nr_agents = 2
    capture_radius = 1.0
    timestep_limit = 1024

    def seed(self, seed):
        self.current_seed = seed

    def reset(self, initial_state):
        self.step_count = 0
        self.world = SimpleNamespace(distance_matrix=np.array([[4., 5., 0.]]))
        return [0]

    def step(self, actions):
        self.step_count += 1
        # Seed 1 captures at step 2; seed 2 captures precisely at the limit;
        # seed 3 reaches distance exactly 1, which must remain a failure.
        if self.current_seed == 1:
            distance = 0.5 if self.step_count == 2 else 3.
        elif self.current_seed == 2:
            distance = 0.75 if self.step_count == 1024 else 2.
        else:
            distance = 1.
        self.world.distance_matrix[0, :2] = [distance, distance + 1]
        done = self.step_count == 1024 or distance < 1.
        return [0], [-distance], done, {}


class EvaluationMetricsTests(unittest.TestCase):
    def test_capture_boolean_timeout_and_distance_aggregation(self):
        suite = {'scenarios': [{'id': str(seed), 'seed': seed, 'initial_state': {}}
                                for seed in (1, 2, 3)]}
        np.random.seed(42)
        before = np.random.get_state()
        reports = evaluate_scenarios(ScriptedEnv(), lambda obs: np.random.rand(1), suite)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual([r['captured'] for r in reports], [True, True, False])
        self.assertEqual([r['steps'] for r in reports], [2, 1024, 1024])
        metrics = summarize(reports)
        self.assertEqual(metrics['success_rate'], 2 / 3)
        self.assertEqual(metrics['mean_minimum_distance'], (0.5 + 0.75 + 1) / 3)
        self.assertEqual(metrics['mean_final_distance'], (0.5 + 0.75 + 1) / 3)
        self.assertEqual(metrics['mean_capture_steps_success'], 513.)
        self.assertLess(metrics['success_rate_ci95_low'], 2 / 3)
        self.assertGreater(metrics['success_rate_ci95_high'], 2 / 3)

    def test_no_successes_and_exact_ties_are_supported(self):
        rows = [{'captured': False, 'steps': 1024, 'capture_step': None,
                 'episode_return': -10., 'minimum_nearest_distance': 1., 'final_nearest_distance': 2.}]
        metrics = summarize(rows)
        self.assertIsNone(metrics['mean_capture_steps_success'])
        tied = [dict(metrics, step=step, checkpoint=str(step)) for step in (2, 1)]
        self.assertEqual([r['rank'] for r in rank_models(tied)], [1, 1])

    def test_ranking_uses_success_then_minimum_then_final_distance(self):
        rows = [dict(step=step, checkpoint=str(step), success_rate=rate,
                     mean_minimum_distance=minimum, mean_final_distance=final)
                for step, rate, minimum, final in [(1, .5, .1, .1), (2, .6, 2, 3),
                                                   (3, .6, 1, 5), (4, .6, 1, 4)]]
        self.assertEqual([r['step'] for r in rank_models(rows)], [4, 3, 2, 1])

    def test_saved_suite_is_reused_and_count_mismatch_rejected(self):
        settings = {'nr_pursuers': 10, 'nr_evaders': 1, 'world_size': 100}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'scenarios.json'
            first, _ = load_scenarios(settings, 30000, 200, path)
            second, _ = load_scenarios(settings, 30000, 200, path)
            self.assertEqual(first, second)
            self.assertEqual(len(first['scenarios']), 200)
            self.assertEqual(first['scenarios'][-1]['seed'], 30199)
            with self.assertRaises(ValueError):
                load_scenarios(settings, 30000, 5, path)

    def test_csv_aliases_support_multiple_labels_and_preserve_old_snapshots(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            rows = [dict(step=step, checkpoint=str(root / str(step) / 'model'),
                         success_rate=rate, mean_minimum_distance=1., mean_final_distance=2.)
                    for step, rate in [(10, .6), (20, .5), (30, .4)]]
            original_rows = [dict(row) for row in rows]
            snapshot = {'best': rows[0]['checkpoint'], 'selected': rows[0]['checkpoint'],
                        'final': rows[1]['checkpoint']}
            first = write_ranked_csv(root / 'first.csv', rows, snapshot)
            self.assertEqual([row['aliases'] for row in first], ['best;selected', 'final', ''])
            self.assertEqual(rows, original_rows)
            before = (root / 'first.csv').read_bytes()
            with (root / 'first.csv').open(encoding='utf-8-sig', newline='') as source:
                reader = csv.DictReader(source)
                self.assertEqual(reader.fieldnames[:4], ['rank', 'step', 'aliases', 'checkpoint'])
                self.assertEqual([row['aliases'] for row in reader], ['best;selected', 'final', ''])
            snapshot['selected'] = rows[1]['checkpoint']
            second = write_ranked_csv(root / 'second.csv', rows, snapshot)
            self.assertEqual([row['aliases'] for row in second], ['best', 'selected;final', ''])
            self.assertEqual((root / 'first.csv').read_bytes(), before)
            self.assertEqual(first[0]['aliases'], 'best;selected')


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = _scipy_dll_directory()
        cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_periodic_checkpoints_survive_worse_scores_and_final_is_saved_once(self):
        from deep_rl_for_swarms.run_multiagent_mappo import train
        scores = [dict(mean_episode_return=score, std_episode_return=0., success_rate=0.,
                       mean_minimum_distance=2., mean_final_distance=3.) for score in (3., 2., 1.)]
        with tempfile.TemporaryDirectory() as root:
            with patch('deep_rl_for_swarms.rl_algo.mappo._evaluate', side_effect=scores):
                run = Path(train(20, root, rollout_steps=8, minibatch_steps=4, epochs=1,
                                 eval_interval=8, checkpoint_interval=8, eval_episodes=1, test=False))
            manifest = json.loads((run / 'checkpoints.json').read_text())
            self.assertEqual([r['step'] for r in manifest['checkpoints']], [8, 16, 20])
            self.assertEqual(manifest['best']['step'], 8)
            self.assertEqual(manifest['final']['step'], 20)
            self.assertTrue(all((run / str(step) / 'model.index').is_file() for step in (8, 16, 20)))

    def test_saving_and_monitoring_intervals_are_independent(self):
        from deep_rl_for_swarms.run_multiagent_mappo import train
        score = dict(mean_episode_return=0., std_episode_return=0., success_rate=0.,
                     mean_minimum_distance=2., mean_final_distance=3.)
        with tempfile.TemporaryDirectory() as root:
            with patch('deep_rl_for_swarms.rl_algo.mappo._evaluate', return_value=score):
                run = Path(train(20, root, rollout_steps=4, minibatch_steps=4, epochs=1,
                                 eval_interval=12, checkpoint_interval=8, eval_episodes=1, test=False))
            manifest = json.loads((run / 'checkpoints.json').read_text())
            self.assertEqual([r['step'] for r in manifest['checkpoints']], [8, 12, 16, 20])
            self.assertIsNone(manifest['checkpoints'][0]['eval_mean_return'])
            self.assertNotIn('eval_metrics', manifest['checkpoints'][0])
            self.assertEqual(manifest['best']['step'], 12)
            self.assertEqual(manifest['final']['step'], 20)

    def test_monitor_initial_states_match_original_reset_exactly(self):
        from deep_rl_for_swarms.ma_envs.envs.point_envs.pursuit_evasion import PursuitEvasionEnv
        settings = dict(nr_pursuers=10, nr_evaders=1, obs_mode='sum_obs_no_ori',
                        comm_radius=200 * np.sqrt(2), world_size=100,
                        distance_bins=8, bearing_bins=8, torus=True, dynamics='unicycle')
        suite = make_scenarios(settings, 10000, 20, placement_scale=0.95 - 0.05)
        env = PursuitEvasionEnv(**settings)
        state = np.random.get_state()
        try:
            for scene in suite['scenarios']:
                np.random.seed(scene['seed'])
                env.seed(scene['seed'])
                env.reset()
                np.testing.assert_array_equal(env.world.agent_states, scene['initial_state']['pursuer_states'])
                np.testing.assert_array_equal(env.world.landmark_states, scene['initial_state']['evader_states'])
        finally:
            env.close()
            np.random.set_state(state)

    def test_default_post_training_batch_matches_standalone_without_artifacts(self):
        from deep_rl_for_swarms.run_multiagent_mappo import train
        from deep_rl_for_swarms.mappo_tools import test_checkpoint, resolve_checkpoint
        # Surround the evader close to its next position so these integration
        # episodes terminate quickly using the real graph and environment.
        settings = dict(nr_pursuers=10, nr_evaders=1, obs_mode='sum_obs_no_ori',
                        comm_radius=200 * np.sqrt(2), world_size=100,
                        distance_bins=8, bearing_bins=8, torus=True, dynamics='unicycle')
        suite = make_scenarios(settings, 30000, 2)
        for scene in suite['scenarios']:
            scene['initial_state'] = {
                'pursuer_states': [[30 + 2.1 * np.cos(i * 2 * np.pi / 10),
                                    30 + 2.1 * np.sin(i * 2 * np.pi / 10), float(i)]
                                   for i in range(10)], 'evader_states': [[30., 30.]]}
        with tempfile.TemporaryDirectory() as root:
            suite_path = Path(root) / 'suite.json'
            suite_path.write_text(json.dumps(suite))
            with patch('deep_rl_for_swarms.rl_algo.mappo._evaluate', return_value=dict(
                    mean_episode_return=0., std_episode_return=0., success_rate=0.,
                    mean_minimum_distance=1., mean_final_distance=2.)):
                run = Path(train(8, root, rollout_steps=4, minibatch_steps=4, epochs=1,
                                 eval_interval=4, checkpoint_interval=4, eval_episodes=1,
                                 test_episodes=2, test_scenarios=str(suite_path)))
            report = run / 'test' / 'validation_2_seed30000'
            metadata = json.loads((report / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['status'], 'complete')
            self.assertEqual(len(metadata['results']), 2)
            self.assertEqual(list(report.glob('scenario_*')), [])
            self.assertEqual(len(list(report.iterdir())), 3)
            self.assertIn(resolve_checkpoint(run, 'selected')['step'], [4, 8])
            with (report / 'summary.csv').open(encoding='utf-8-sig', newline='') as source:
                rows = list(csv.DictReader(source))
            chosen = int(rows[0]['step'])
            self.assertIn('selected', rows[0]['aliases'].split(';'))
            self.assertEqual(metadata['alias_snapshot']['selected'], rows[0]['checkpoint'])
            manifest = json.loads((run / 'checkpoints.json').read_text())
            for row in rows:
                expected = ';'.join(name for name in ('best', 'selected', 'final')
                                    if (run / manifest[name]['path']).resolve() == Path(row['checkpoint']))
                self.assertEqual(row['aliases'], expected)
            single = test_checkpoint(run / str(chosen), test_episodes=2, test_scenarios=str(suite_path))
            for key in ('success_count', 'success_rate', 'mean_minimum_distance', 'mean_final_distance'):
                self.assertEqual(single['rows'][0][key], metadata['results'][0][key])
            self.assertEqual(single['rows'][0]['aliases'], rows[0]['aliases'])
            unchanged = (report / 'summary.csv').read_bytes()
            other = test_checkpoint(run / rows[1]['step'], test_episodes=2, test_scenarios=str(suite_path))
            self.assertNotIn('selected', other['rows'][0]['aliases'].split(';'))
            self.assertEqual((report / 'summary.csv').read_bytes(), unchanged)
            self.assertEqual(resolve_checkpoint(run, 'selected')['step'], chosen)
            with self.assertRaises(FileExistsError):
                test_checkpoint(run / str(chosen), test_episodes=2, test_scenarios=str(suite_path))


if __name__ == '__main__':
    unittest.main()
