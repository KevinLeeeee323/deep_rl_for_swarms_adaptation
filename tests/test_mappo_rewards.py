"""Capture reward boundaries and independence from the termination rule."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from deep_rl_for_swarms.mappo_tools import _scipy_dll_directory, _reusable_results, reward_settings


class CaptureRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dll = _scipy_dll_directory()
        cls.dll.__enter__()
        from deep_rl_for_swarms.ma_envs.envs.point_envs.pursuit_evasion import PursuitEvasionEnv
        cls.environment_class = PursuitEvasionEnv

    @classmethod
    def tearDownClass(cls):
        cls.dll.__exit__(None, None, None)

    def make_env(self, distance, bonus):
        env = self.environment_class(nr_pursuers=10, nr_evaders=1, obs_mode='sum_obs_no_ori',
                                     comm_radius=200 * np.sqrt(2), world_size=100,
                                     distance_bins=8, bearing_bins=8, torus=True,
                                     dynamics='unicycle', capture_bonus=bonus)
        self.addCleanup(env.close)
        env.reset(initial_state={'pursuer_states': [[30 + distance, 30., 0.]] +
                                 [[50. + i, 50., 0.] for i in range(9)],
                                 'evader_states': [[30., 30.]]})
        return env

    def test_bonus_is_shared_and_paid_only_below_strict_capture_threshold(self):
        for bonus in (0., 1.):
            for distance in (.5, 1., 1.000001, 5.):
                with self.subTest(bonus=bonus, distance=distance):
                    env = self.make_env(distance, bonus)
                    with patch.object(env.world, 'step'):
                        _, rewards, done, info = env.step(np.zeros((10, 2)))
                    captured = distance < 1
                    expected = -distance / env.obs_radius + (bonus if captured else 0.)
                    np.testing.assert_allclose(rewards, np.full(10, expected), rtol=0, atol=1e-15)
                    self.assertAlmostEqual(float(np.mean(rewards)), expected)
                    self.assertEqual(done, captured)
                    self.assertEqual(info['captured'], captured)

    def test_timeout_has_no_capture_bonus_but_capture_on_last_step_does(self):
        for distance in (.5, 5.):
            env = self.make_env(distance, 1.)
            env.timestep = env.timestep_limit - 1
            with patch.object(env.world, 'step'):
                _, rewards, done, info = env.step(np.zeros((10, 2)))
            self.assertTrue(done)
            self.assertEqual(info['captured'], distance < 1)
            self.assertAlmostEqual(rewards[0], -distance / env.obs_radius + float(distance < 1))

    def test_arbitrary_positive_reward_cannot_cause_false_capture(self):
        env = self.make_env(5., 1.)
        with patch.object(env.world, 'step'), patch.object(env, 'get_reward', return_value=np.ones(10)):
            _, _, done, info = env.step(np.zeros((10, 2)))
        self.assertFalse(done)
        self.assertFalse(info['captured'])

    def test_legacy_defaults_validation_and_report_reuse_guard(self):
        self.assertEqual(reward_settings()['capture_bonus'], 0.)
        self.assertEqual(reward_settings({'reward': reward_settings(capture_bonus=1)})['capture_bonus'], 1.)
        self.assertEqual(reward_settings({'reward': reward_settings(capture_bonus=1)}, 0)['mode'], 'legacy_distance')
        for bonus in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                reward_settings(capture_bonus=bonus)
            with self.assertRaises(ValueError):
                self.environment_class(capture_bonus=bonus)
        with tempfile.TemporaryDirectory() as root:
            Path(root, 'metadata.json').write_text(json.dumps({'reward': reward_settings()}))
            with self.assertRaisesRegex(ValueError, 'incompatible reward'):
                _reusable_results(root, {}, {'reward': reward_settings(capture_bonus=1)}, [])


if __name__ == '__main__':
    unittest.main()
