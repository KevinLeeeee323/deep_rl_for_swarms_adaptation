#!/usr/bin/env python3
import argparse
import datetime
import inspect
import json
import os
import platform

import numpy as np


def train(num_timesteps, log_dir, seed=0, rollout_steps=2048, minibatch_steps=64, epochs=5,
          eval_interval=50_000, eval_episodes=20, eval_seed=10_000,
          test=True, test_seed=20_000, test_scenarios=None, test_model='best',
          animation_fps=20, animation_stride=4, resume=None):
    import tensorflow as tf
    from deep_rl_for_swarms.common import logger
    from deep_rl_for_swarms.common import tf_util as U
    from deep_rl_for_swarms.ma_envs.envs.point_envs.pursuit_evasion import PursuitEvasionEnv
    from deep_rl_for_swarms.rl_algo import mappo

    np.random.seed(seed)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_dir = os.path.join(log_dir, '{}_{}steps'.format(timestamp, num_timesteps))
    os.makedirs(log_dir, exist_ok=False)
    env_settings = dict(nr_pursuers=10, nr_evaders=1, obs_mode='sum_obs_no_ori',
                        comm_radius=200 * np.sqrt(2), world_size=100,
                        distance_bins=8, bearing_bins=8, torus=True,
                        dynamics='unicycle')
    if resume is not None:
        env_settings = resume['config']['environment_settings'].copy()
    env = PursuitEvasionEnv(**env_settings)
    eval_env = None
    try:
        eval_env = PursuitEvasionEnv(**env_settings)
        suite_path = None
        if test:
            from deep_rl_for_swarms.common.mappo_test import load_test_scenarios, run_tests
            from deep_rl_for_swarms.common.mappo_visualization import ffmpeg_executable
            if set(range(test_seed, test_seed + 5)).intersection(range(eval_seed, eval_seed + eval_episodes)):
                raise ValueError('Test seeds must differ from evaluation seeds.')
            ffmpeg_executable()
            suite, suite_path = load_test_scenarios(env_settings, test_seed, test_scenarios)
            for scene in suite['scenarios']:
                eval_env.reset(initial_state=scene['initial_state'])
        algorithm_settings = {
            name: parameter.default
            for name, parameter in inspect.signature(mappo.learn).parameters.items()
            if parameter.default is not inspect.Parameter.empty
            and name not in ('eval_env', 'resume_checkpoint', 'initial_timesteps')
        }
        algorithm_settings.update(rollout_steps=rollout_steps, minibatch_steps=minibatch_steps,
                                  epochs=epochs, eval_interval=eval_interval,
                                  eval_episodes=eval_episodes, eval_seed=eval_seed)
        if resume is not None:
            algorithm_settings.update(resume['training'])
        config = {
            'algorithm': 'MAPPO', 'environment': 'PursuitEvasionEnv',
            'started_at': datetime.datetime.now().astimezone().isoformat(),
            'run_directory': os.path.abspath(log_dir),
            'num_timesteps': num_timesteps, 'seed': seed,
            'environment_settings': env_settings, 'training': algorithm_settings,
            'environment_runtime': {'timestep_limit': env.timestep_limit,
                                    'capture_radius': env.capture_radius,
                                    'world_dt': env.world.dt, 'action_repeat': env.world.action_repeat,
                                    'observation_shape': [int(size) for size in env.observation_space.shape],
                                    'action_shape': [int(size) for size in env.action_space.shape],
                                    'pursuer_max_linear_velocity': env.agents[0].max_lin_velocity,
                                    'pursuer_max_angular_velocity': env.agents[0].max_ang_velocity,
                                    'evader_max_speed': env.world.scripted_agents[0].max_speed,
                                    'observation_radius': env.obs_radius,
                                    'action_low': env.action_space.low.tolist(),
                                    'action_high': env.action_space.high.tolist()},
            'network': {'actor_hidden_sizes': [64], 'embedding_sizes': [64],
                        'critic_hidden_sizes': [64], 'gradient_clip_norm': 0.5,
                        'activation': 'relu', 'layer_norm': False,
                        'action_distribution': 'diagonal_gaussian',
                        'gaussian_fixed_var': True,
                        'critic_state': 'pursuer poses and velocities, evader position, episode progress',
                        'critic_state_size': env.nr_agents * 6 + env.nr_evaders * 2 + 1,
                        'value_loss_coefficient': 0.5,
                        'advantage_normalization_epsilon': 1e-8},
            'optimizer': {'name': 'Adam', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8},
            'test': {'enabled': test, 'episodes': 5, 'seed': test_seed,
                     'scenarios_path': str(suite_path) if suite_path is not None else None,
                     'model': test_model, 'max_episode_steps': 1024,
                     'fps': animation_fps, 'frame_stride': animation_stride,
                     'actions': 'policy_mean'},
            'checkpoint_policy': {'retained_models': ['best', 'final'],
                                  'selection_metric': 'mean_episode_return',
                                  'selection_order': 'higher_is_better',
                                  'eval_actions': 'policy_mean'},
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                        'tensorflow': tf.__version__, 'processes': 1,
                        'tensorflow_intra_op_threads': 1, 'tensorflow_inter_op_threads': 1},
        }
        if resume is not None:
            from deep_rl_for_swarms.mappo_tools import validate_environment
            validate_environment(resume['config'], env)
            config['continuation'] = resume['continuation']
        with open(os.path.join(log_dir, 'config.json'), 'w', encoding='utf-8') as output:
            json.dump(config, output, indent=2, allow_nan=False)
        with tf.Graph().as_default(), U.single_threaded_session() as session:
            tf.set_random_seed(seed)
            with logger.scoped_configure(dir=log_dir, format_strs=['stdout', 'csv']):
                policy = mappo.learn(
                    env, session, num_timesteps, logger.get_dir(),
                    eval_env=eval_env,
                    resume_checkpoint=resume['checkpoint'] if resume is not None else None,
                    initial_timesteps=resume['step'] if resume is not None else 0,
                    **algorithm_settings)
                if test:
                    with open(os.path.join(log_dir, 'checkpoints.json')) as source:
                        selection = json.load(source)
                    summary = run_tests(eval_env, session, policy, log_dir, selection, suite,
                                        model=test_model, fps=animation_fps,
                                        frame_stride=animation_stride)
                    logger.log('Test success rate {:.1%}'.format(summary['success_rate']))
        return os.path.abspath(log_dir)
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()


def main():
    parser = argparse.ArgumentParser(description='Train MAPPO on continuous-control PursuitEvasion.')
    parser.add_argument('--num-timesteps', type=int, default=10_000_000)
    parser.add_argument('--rollout-steps', type=int, default=2048)
    parser.add_argument('--minibatch-steps', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-interval', type=int, default=50_000,
                        help='Training steps between evaluations, rounded up to an update boundary.')
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--eval-seed', type=int, default=10_000,
                        help='First seed in the fixed sequence of evaluation episode seeds.')
    parser.add_argument('--no-test', dest='test', action='store_false',
                        help='Skip the automatic five-scenario test and MP4 export.')
    parser.add_argument('--test-seed', type=int, default=20_000)
    parser.add_argument('--test-scenarios', help='Path to a reusable JSON file of five test scenarios.')
    parser.add_argument('--test-model', choices=['best', 'final'], default='best')
    parser.add_argument('--animation-fps', type=int, default=20)
    parser.add_argument('--animation-stride', type=int, default=4,
                        help='Render every Nth test step, including the initial and final states.')
    parser.add_argument('--log-dir', default='runs/',
                        help='Parent directory for timestamped training runs (default: logs/).')
    args = parser.parse_args()
    for name in ('num_timesteps', 'rollout_steps', 'minibatch_steps', 'epochs',
                 'eval_interval', 'eval_episodes', 'animation_fps', 'animation_stride'):
        if getattr(args, name) < 1:
            parser.error('--' + name.replace('_', '-') + ' must be positive')
    for name, count in (('seed', 1), ('eval_seed', args.eval_episodes), ('test_seed', 5)):
        if not 0 <= getattr(args, name) <= 2**32 - count:
            parser.error('--' + name.replace('_', '-') + ' is outside the supported NumPy seed range')
    train(**vars(args))


if __name__ == '__main__':
    main()
