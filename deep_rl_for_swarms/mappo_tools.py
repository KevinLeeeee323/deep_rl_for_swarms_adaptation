"""Checkpoint utilities: resume MAPPO training or test a saved model.

Run from the repository root with ``python -m deep_rl_for_swarms.mappo_tools``.
The source run is read-only except for new standalone test reports.
"""

import argparse
from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


TRAINING_KEYS = ('rollout_steps', 'minibatch_steps', 'epochs', 'gamma', 'lam',
                 'clip', 'learning_rate', 'entropy_coef', 'eval_interval',
                 'eval_episodes', 'eval_seed')


@contextmanager
def _scipy_dll_directory():
    """Support older Windows SciPy wheels missing their DLL initialization.

    Limit the extra DLL search directory to this command's lifetime; leave the
    installed environment and the caller's process configuration unchanged.
    """
    spec = importlib.util.find_spec('scipy') if os.name == 'nt' else None
    directory = Path(spec.origin).parent / '.libs' if spec and spec.origin else None
    if directory is None or not directory.is_dir():
        yield
    elif hasattr(os, 'add_dll_directory'):
        with os.add_dll_directory(str(directory)):
            yield
    else:
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        previous = ctypes.create_unicode_buffer(32768)
        kernel.GetDllDirectoryW(len(previous), previous)
        if not kernel.SetDllDirectoryW(str(directory)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            kernel.SetDllDirectoryW(previous.value or None)


def _read_json(path):
    with Path(path).open(encoding='utf-8-sig') as source:
        return json.load(source)


def resolve_checkpoint(checkpoint, model='final'):
    """Accept a step directory, model prefix/.index/.meta, or a run directory."""
    path = Path(checkpoint).expanduser().resolve()
    if path.is_dir() and (path / 'config.json').is_file():
        run_dir = path
        selection = _read_json(run_dir / 'checkpoints.json')
        path = (run_dir / selection[model]['path']).resolve()
    else:
        if path.is_dir():
            path = path / 'model'
        elif path.suffix in ('.index', '.meta'):
            path = path.with_suffix('')
        run_dir = path.parent.parent
    if path.parent.parent != run_dir or not path.parent.name.isdigit():
        raise ValueError('Checkpoint must be in a numeric step directory inside its run.')
    if not Path(str(path) + '.index').is_file() or not list(path.parent.glob(path.name + '.data-*')):
        raise ValueError('Missing checkpoint .index or .data files: {}'.format(path))
    config = _read_json(run_dir / 'config.json')
    if config.get('algorithm') != 'MAPPO' or config.get('environment') != 'PursuitEvasionEnv':
        raise ValueError('Only MAPPO / PursuitEvasionEnv checkpoints are supported.')
    # Graph construction must match this repository's current training graph.
    expected = {'actor_hidden_sizes': [64], 'embedding_sizes': [64],
                'critic_hidden_sizes': [64], 'activation': 'relu', 'layer_norm': False,
                'action_distribution': 'diagonal_gaussian', 'gaussian_fixed_var': True,
                'gradient_clip_norm': 0.5, 'value_loss_coefficient': 0.5,
                'advantage_normalization_epsilon': 1e-8}
    for key, value in expected.items():
        if config.get('network', {}).get(key) != value:
            raise ValueError('Unsupported network setting {}: expected {!r}.'.format(key, value))
    step = int(path.parent.name)
    record = {'step': step, 'path': path.relative_to(run_dir).as_posix()}
    selection_path = run_dir / 'checkpoints.json'
    if selection_path.exists():
        for entry in _read_json(selection_path).values():
            if (run_dir / entry['path']).resolve() == path:
                if entry['step'] != step:
                    raise ValueError('Checkpoint step disagrees with checkpoints.json.')
                record.update(entry)
                break
    return {'run_dir': run_dir, 'checkpoint': str(path), 'step': step,
            'config': config, 'record': record}


def validate_environment(config, env):
    """Reject changed observation/action spaces and dynamics before restoring."""
    expected = config.get('environment_runtime', {})
    actual = {'observation_shape': list(env.observation_space.shape),
              'action_shape': list(env.action_space.shape),
              'action_low': env.action_space.low.tolist(),
              'action_high': env.action_space.high.tolist(),
              'timestep_limit': env.timestep_limit, 'capture_radius': env.capture_radius,
              'world_dt': env.world.dt, 'action_repeat': env.world.action_repeat,
              'pursuer_max_linear_velocity': env.agents[0].max_lin_velocity,
              'pursuer_max_angular_velocity': env.agents[0].max_ang_velocity,
              'evader_max_speed': env.world.scripted_agents[0].max_speed,
              'observation_radius': env.obs_radius}
    for key, value in actual.items():
        if key in expected and expected[key] != value:
            raise ValueError('Environment setting {} differs from the saved run.'.format(key))


def _scenario_path(source, seed, explicit_path=None):
    if explicit_path is not None:
        return str(Path(explicit_path).expanduser().resolve())
    if seed != source['config'].get('test', {}).get('seed', 20_000):
        return None
    saved = source['config'].get('test', {}).get('scenarios_path')
    if saved:
        path = Path(saved)
        if path.is_file():
            return str(path.resolve())
        if not path.is_absolute() and (source['run_dir'] / path).is_file():
            return str((source['run_dir'] / path).resolve())
    # Run-local copies keep a moved run usable even when its old absolute
    # test_scenarios path no longer exists.
    for path in sorted((source['run_dir'] / 'test').glob('*/scenarios.json')):
        suite = _read_json(path)
        if (suite.get('environment_settings') == source['config']['environment_settings']
                and [scene['seed'] for scene in suite['scenarios']] == list(range(seed, seed + 5))):
            return str(path.resolve())
    return None


def _validate_seeds(seed, eval_seed, eval_episodes, test_seed, test):
    for name, value, count in [('seed', seed, 1), ('eval_seed', eval_seed, eval_episodes),
                                ('test_seed', test_seed, 5)]:
        if not 0 <= value <= 2**32 - count:
            raise ValueError('{} is outside the supported NumPy seed range.'.format(name))
    if test and set(range(test_seed, test_seed + 5)).intersection(
            range(eval_seed, eval_seed + eval_episodes)):
        raise ValueError('Test seeds must differ from evaluation seeds.')


def resume_training(checkpoint, additional_timesteps, model='final', log_dir=None,
                    note='', **overrides):
    """Restore all model/Adam variables and train for an additional step budget.

    A fresh environment and random streams are started; this is not a bitwise
    continuation of the interrupted sampler. Hyperparameters default to config.json.
    """
    from deep_rl_for_swarms.run_multiagent_mappo import train

    if additional_timesteps < 1:
        raise ValueError('Additional timesteps must be positive.')
    source = resolve_checkpoint(checkpoint, model)
    config = source['config']
    optimizer = {'name': 'Adam', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
    if config.get('optimizer') != optimizer:
        raise ValueError('Saved optimizer settings do not match the current Adam optimizer.')
    training = {key: config['training'][key] for key in TRAINING_KEYS}
    test_config = config.get('test', {})
    options = {'seed': config['seed'], 'test': test_config.get('enabled', True),
               'test_seed': test_config.get('seed', 20_000), 'test_scenarios': None,
               'test_model': test_config.get('model', 'best'),
               'animation_fps': test_config.get('fps', 20),
               'animation_stride': test_config.get('frame_stride', 4)}
    for key, value in overrides.items():
        if value is None:
            continue
        if key in training:
            training[key] = value
        elif key in options:
            options[key] = value
        else:
            raise ValueError('Unknown continuation option: {}'.format(key))
    for key in ('rollout_steps', 'minibatch_steps', 'epochs', 'eval_interval', 'eval_episodes'):
        if training[key] < 1:
            raise ValueError('{} must be positive.'.format(key))
    for key in ('gamma', 'lam'):
        if not 0 <= training[key] <= 1:
            raise ValueError('{} must be between zero and one.'.format(key))
    if not np.isfinite(training['learning_rate']) or training['learning_rate'] <= 0:
        raise ValueError('learning_rate must be finite and positive.')
    if not 0 < training['clip'] < 1 or not np.isfinite(training['entropy_coef']) or training['entropy_coef'] < 0:
        raise ValueError('clip must be in (0, 1) and entropy_coef must be finite and nonnegative.')
    if options['animation_fps'] < 1 or options['animation_stride'] < 1:
        raise ValueError('Animation fps and stride must be positive.')
    _validate_seeds(options['seed'], training['eval_seed'], training['eval_episodes'],
                    options['test_seed'], options['test'])
    options['test_scenarios'] = _scenario_path(source, options['test_seed'], options['test_scenarios'])
    parent = {'source_run_directory': str(source['run_dir']),
              'source_config': str(source['run_dir'] / 'config.json'),
              'source_checkpoint': source['checkpoint'], 'source_step': source['step']}
    history = copy.deepcopy(config.get('continuation', {}).get('history', []))
    history.append(dict(parent, additional_timesteps=additional_timesteps, note=note))
    changes = {key: {'from': config['training'][key], 'to': value}
               for key, value in training.items() if value != config['training'][key]}
    source['training'] = training
    source['continuation'] = dict(
        parent, additional_timesteps=additional_timesteps,
        target_total_timesteps=source['step'] + additional_timesteps,
        restored_components=['actor', 'central_critic', 'Adam'],
        sampler_state_restored=False, note=note, training_overrides=changes, history=history)
    runner_settings = {key: training[key] for key in (
        'rollout_steps', 'minibatch_steps', 'epochs', 'eval_interval', 'eval_episodes', 'eval_seed')}
    return train(additional_timesteps, str(log_dir or source['run_dir'].parent),
                 resume=source, **dict(runner_settings, **options))


def test_checkpoint(checkpoint, model='final', test_seed=None, test_scenarios=None,
                    animation_fps=None, animation_stride=None):
    """Test a checkpoint on the saved suite, writing run/test/<step>/ artifacts."""
    source = resolve_checkpoint(checkpoint, model)
    config = source['config']
    saved_test = config.get('test', {})
    seed = saved_test.get('seed', 20_000) if test_seed is None else test_seed
    fps = saved_test.get('fps', 20) if animation_fps is None else animation_fps
    stride = saved_test.get('frame_stride', 4) if animation_stride is None else animation_stride
    if fps < 1 or stride < 1:
        raise ValueError('Animation fps and stride must be positive.')
    _validate_seeds(config['seed'], config['training']['eval_seed'],
                    config['training']['eval_episodes'], seed, True)
    output_dir = source['run_dir'] / 'test' / str(source['step'])
    if output_dir.exists():
        raise FileExistsError('Test output already exists; move it before rerunning: {}'.format(output_dir))

    import tensorflow as tf
    from deep_rl_for_swarms.common import tf_util as U
    from deep_rl_for_swarms.common.mappo_test import load_test_scenarios, run_tests
    from deep_rl_for_swarms.common.mappo_visualization import ffmpeg_executable
    from deep_rl_for_swarms.ma_envs.envs.point_envs.pursuit_evasion import PursuitEvasionEnv
    from deep_rl_for_swarms.policies.mlp_mean_embedding_policy import MlpPolicy

    ffmpeg_executable()
    suite, _ = load_test_scenarios(config['environment_settings'], seed,
                                    _scenario_path(source, seed, test_scenarios))
    env = PursuitEvasionEnv(**config['environment_settings'])
    random_state = np.random.get_state()
    try:
        validate_environment(config, env)
        np.random.seed(config['seed'])
        with tf.Graph().as_default(), U.single_threaded_session() as session:
            tf.set_random_seed(config['seed'])
            obs = tf.placeholder(tf.float32, (None,) + env.observation_space.shape,
                                 name='observations')
            policy = MlpPolicy('mappo_policy', env.observation_space, env.action_space,
                               hid_size=config['network']['actor_hidden_sizes'],
                               feat_size=config['network']['embedding_sizes'], observation_input=obs)
            label = str(source['step'])
            return run_tests(env, session, policy, source['run_dir'],
                             {label: source['record']}, suite, model=label,
                             fps=fps, frame_stride=stride)
    finally:
        env.close()
        np.random.set_state(random_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    resume = commands.add_parser('resume', help='Continue training into a new run directory.')
    test = commands.add_parser('test', help='Test a checkpoint into its run/test/<step>/ directory.')
    for command in (resume, test):
        command.add_argument('checkpoint', help='Step directory, model prefix, or run directory.')
        command.add_argument('--model', choices=['best', 'final'], default='final',
                             help='Selection when checkpoint is a run directory (default: final).')
        command.add_argument('--test-seed', type=int)
        command.add_argument('--test-scenarios')
        command.add_argument('--animation-fps', type=int)
        command.add_argument('--animation-stride', type=int)
    resume.add_argument('--additional-timesteps', type=int, required=True,
                        help='Extra environment steps; 400000 + 600000 ends at 1000000.')
    resume.add_argument('--log-dir', help='Parent of new run; defaults to source run parent.')
    resume.add_argument('--note', default='', help='Free-text remark saved in config.json.')
    resume.add_argument('--seed', type=int)
    for name in ('rollout_steps', 'minibatch_steps', 'epochs', 'eval_interval', 'eval_episodes', 'eval_seed'):
        resume.add_argument('--' + name.replace('_', '-'), type=int)
    for name in ('learning_rate', 'entropy_coef', 'gamma', 'lam', 'clip'):
        resume.add_argument('--' + name.replace('_', '-'), type=float)
    testing = resume.add_mutually_exclusive_group()
    testing.add_argument('--test', dest='test', action='store_true')
    testing.add_argument('--no-test', dest='test', action='store_false')
    resume.set_defaults(test=None)
    resume.add_argument('--test-model', choices=['best', 'final'])
    args = vars(parser.parse_args())
    command = args.pop('command')
    try:
        with _scipy_dll_directory():
            if command == 'resume':
                run_dir = resume_training(**args)
                print('Continuation saved to {}'.format(run_dir))
            else:
                summary = test_checkpoint(**args)
                print('Step {}: success rate {:.1%}, mean return {:.6f}'.format(
                    summary['checkpoint']['step'], summary['success_rate'], summary['mean_episode_return']))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
