"""Checkpoint utilities: resume MAPPO training or test a saved model.

Run from the repository root with ``python -m deep_rl_for_swarms.mappo_tools``.
Testing writes reports and a selected-checkpoint alias, without changing weights.
"""

import argparse
from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np


TRAINING_KEYS = ('rollout_steps', 'minibatch_steps', 'epochs', 'gamma', 'lam',
                 'clip', 'learning_rate', 'entropy_coef', 'eval_interval',
                 'eval_episodes', 'eval_seed', 'checkpoint_interval')


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
            if not isinstance(entry, dict) or 'path' not in entry:
                continue
            if (run_dir / entry['path']).resolve() == path:
                if entry['step'] != step:
                    raise ValueError('Checkpoint step disagrees with checkpoints.json.')
                record.update(entry)
                break
    return {'run_dir': run_dir, 'checkpoint': str(path), 'step': step,
            'config': config, 'record': record}


def reward_settings(config=None, capture_bonus=None):
    """Legacy configs imply no bonus; explicit overrides start a new experiment."""
    saved = (config or {}).get('reward', {})
    bonus = saved.get('capture_bonus', 0.0) if capture_bonus is None else capture_bonus
    if not np.isfinite(bonus) or bonus < 0:
        raise ValueError('capture_bonus must be finite and nonnegative.')
    return {'version': 2 if bonus > 0 else 1,
            'mode': 'distance_plus_capture_bonus' if bonus > 0 else 'legacy_distance',
            'capture_bonus': float(bonus),
            'distance_term': '-min(nearest_distance, obs_radius) / obs_radius',
            'capture_condition': 'nearest_distance < capture_radius',
            'team_reward': 'shared; aggregated by mean'}


def validate_environment(config, env, allow_reward_change=False):
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
    if not allow_reward_change and env.capture_bonus != reward_settings(config)['capture_bonus']:
        raise ValueError('Environment capture bonus differs from the saved run.')


def _scenario_path(source, seed, explicit_path=None, episodes=5):
    if explicit_path is not None:
        return str(Path(explicit_path).expanduser().resolve())
    if seed != source['config'].get('test', {}).get('seed', 20_000):
        return None
    saved = source['config'].get('test', {}).get('scenarios_path')
    if saved:
        path = Path(saved)
        if not path.is_absolute() and not path.is_file():
            path = source['run_dir'] / path
        if path.is_file():
            suite = _read_json(path)
            if [scene['seed'] for scene in suite['scenarios']] == list(range(seed, seed + episodes)):
                return str(path.resolve())
    # Run-local copies keep a moved run usable even when its old absolute
    # test_scenarios path no longer exists.
    for path in sorted((source['run_dir'] / 'test').glob('*/scenarios.json')):
        suite = _read_json(path)
        if (suite.get('environment_settings') == source['config']['environment_settings']
                and [scene['seed'] for scene in suite['scenarios']] == list(range(seed, seed + episodes))):
            return str(path.resolve())
    return None


def _validate_seeds(seed, eval_seed, eval_episodes, test_seed, test, test_episodes=5):
    if eval_episodes < 1 or test_episodes < 1:
        raise ValueError('Evaluation and test episode counts must be positive.')
    for name, value, count in [('seed', seed, 1), ('eval_seed', eval_seed, eval_episodes),
                                ('test_seed', test_seed, test_episodes)]:
        if not 0 <= value <= 2**32 - count:
            raise ValueError('{} is outside the supported NumPy seed range.'.format(name))
    if test and set(range(test_seed, test_seed + test_episodes)).intersection(
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
    training = {key: config['training'][key] for key in TRAINING_KEYS if key in config['training']}
    training.setdefault('checkpoint_interval', 50_000)
    test_config = config.get('test', {})
    options = {'seed': config['seed'], 'test': True,
               'test_seed': test_config.get('seed', 30_000) if test_config.get('purpose') == 'checkpoint_selection' else 30_000,
               'test_episodes': test_config.get('episodes', 200) if test_config.get('purpose') == 'checkpoint_selection' else 200,
               'test_scenarios': None, 'test_workers': None, 'capture_bonus': None}
    for key, value in overrides.items():
        if value is None:
            continue
        if key in training:
            training[key] = value
        elif key in options:
            options[key] = value
        else:
            raise ValueError('Unknown continuation option: {}'.format(key))
    for key in ('rollout_steps', 'minibatch_steps', 'epochs', 'eval_interval', 'eval_episodes', 'checkpoint_interval'):
        if training[key] < 1:
            raise ValueError('{} must be positive.'.format(key))
    for key in ('gamma', 'lam'):
        if not 0 <= training[key] <= 1:
            raise ValueError('{} must be between zero and one.'.format(key))
    if not np.isfinite(training['learning_rate']) or training['learning_rate'] <= 0:
        raise ValueError('learning_rate must be finite and positive.')
    if not 0 < training['clip'] < 1 or not np.isfinite(training['entropy_coef']) or training['entropy_coef'] < 0:
        raise ValueError('clip must be in (0, 1) and entropy_coef must be finite and nonnegative.')
    _validate_seeds(options['seed'], training['eval_seed'], training['eval_episodes'],
                    options['test_seed'], options['test'], options['test_episodes'])
    from deep_rl_for_swarms.mappo_eval_pool import evaluation_workers
    options['test_workers'] = evaluation_workers(options['test_workers'])
    old_reward = reward_settings(config)
    new_reward = reward_settings(config, options['capture_bonus'])
    options['capture_bonus'] = new_reward['capture_bonus']
    reward_changes = ({'capture_bonus': {'from': old_reward['capture_bonus'], 'to': new_reward['capture_bonus']}}
                      if old_reward != new_reward else {})
    options['test_scenarios'] = _scenario_path(source, options['test_seed'], options['test_scenarios'],
                                             options['test_episodes'])
    parent = {'source_run_directory': str(source['run_dir']),
              'source_config': str(source['run_dir'] / 'config.json'),
              'source_checkpoint': source['checkpoint'], 'source_step': source['step']}
    history = copy.deepcopy(config.get('continuation', {}).get('history', []))
    history.append(dict(parent, additional_timesteps=additional_timesteps, note=note,
                        reward_overrides=reward_changes))
    changes = {key: {'from': config['training'].get(key, 50_000), 'to': value}
               for key, value in training.items() if value != config['training'].get(key, 50_000)}
    source['training'] = training
    source['continuation'] = dict(
        parent, additional_timesteps=additional_timesteps,
        target_total_timesteps=source['step'] + additional_timesteps,
        restored_components=['actor', 'central_critic', 'Adam'],
        sampler_state_restored=False, note=note, training_overrides=changes,
        reward_overrides=reward_changes, history=history)
    runner_settings = {key: training[key] for key in (
        'rollout_steps', 'minibatch_steps', 'epochs', 'eval_interval', 'eval_episodes', 'eval_seed', 'checkpoint_interval')}
    return train(additional_timesteps, str(log_dir or source['run_dir'].parent),
                 resume=source, **dict(runner_settings, **options))


def export_checkpoint(checkpoint, model='final', test_seed=None, test_scenarios=None,
                    animation_fps=None, animation_stride=None):
    """Test a checkpoint on the saved suite, writing run/test/<step>/ artifacts."""
    source = resolve_checkpoint(checkpoint, model)
    config = source['config']
    saved_test = config.get('test', {})
    seed = (20_000 if saved_test.get('purpose') == 'checkpoint_selection'
            else saved_test.get('seed', 20_000)) if test_seed is None else test_seed
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
    env = PursuitEvasionEnv(capture_bonus=reward_settings(config)['capture_bonus'],
                            **config['environment_settings'])
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


def discover_checkpoints(run_dir):
    """Discover every physical checkpoint, including legacy best/final runs."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not (run_dir / 'config.json').is_file():
        raise ValueError('Expected a run directory containing config.json.')
    paths = sorted((path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()),
                   key=lambda path: int(path.name))
    sources = [resolve_checkpoint(path) for path in paths]
    if not sources:
        raise ValueError('No saved checkpoints found in {}'.format(run_dir))
    return sources


def _reusable_results(report_dir, suite, metadata, sources):
    """Read a snapshot of complete models without modifying the source report."""
    report_dir = Path(report_dir).expanduser().resolve()
    previous = _read_json(report_dir / 'metadata.json')
    if reward_settings({'reward': previous.get('reward', {})}) != metadata['reward']:
        raise ValueError('Reused report has incompatible reward settings.')
    if _read_json(report_dir / 'scenarios.json') != suite:
        raise ValueError('Reused report must contain exactly the same scenarios and initial states.')
    for key in ('purpose', 'seed', 'episodes', 'ranking', 'actions', 'max_episode_steps',
                'capture_comparison', 'capture_radius', 'distance_geometry', 'distance_definition'):
        # Normalize tuples in the in-memory ranking to their JSON list form.
        if previous.get(key) != json.loads(json.dumps(metadata[key])):
            raise ValueError('Reused report has incompatible {}.'.format(key))
    candidates = {source['checkpoint']: source for source in sources}
    rows = []
    seen = set()
    fields = ('episodes', 'success_count', 'success_rate', 'success_rate_ci95_low',
              'success_rate_ci95_high', 'mean_minimum_distance', 'mean_minimum_distance_failure', 'mean_final_distance',
              'mean_episode_return', 'std_episode_return', 'mean_episode_steps',
              'mean_capture_steps_success', 'step', 'checkpoint', 'elapsed_seconds')
    for row in previous.get('results', []):
        checkpoint = row.get('checkpoint')
        if checkpoint not in candidates:
            continue
        if 'mean_minimum_distance_failure' not in row:
            print('Re-evaluating step {}: reused report lacks failure-only distance.'.format(
                candidates[checkpoint]['step']), flush=True)
            continue
        if (checkpoint in seen or row.get('episodes') != metadata['episodes']
                or row.get('step') != candidates[checkpoint]['step']
                or checkpoint not in previous.get('checkpoints', [])):
            raise ValueError('Invalid completed checkpoint in reused report.')
        seen.add(checkpoint)
        rows.append({field: row[field] for field in fields})
    return rows, str(report_dir)


def _test_sources(sources, test_seed=None, test_episodes=None, test_scenarios=None, output_dir=None,
                  select_candidate=False, workers=None, reuse_report=None):
    from deep_rl_for_swarms.common.mappo_evaluation import (
        CHECKPOINT_ALIASES, RANKING, atomic_json,
        load_scenarios, summarize, write_ranked_csv)
    from deep_rl_for_swarms.mappo_eval_pool import evaluation_workers, evaluate_candidates

    workers = evaluation_workers(workers)
    source = sources[0]
    config = source['config']
    saved = config.get('test', {}) if config.get('test', {}).get('purpose') == 'checkpoint_selection' else {}
    seed = saved.get('seed', 30_000) if test_seed is None else test_seed
    episodes = saved.get('episodes', 200) if test_episodes is None else test_episodes
    output_dir = Path(output_dir or source['run_dir'] / 'test' /
                      'validation_{}_seed{}'.format(episodes, seed)).resolve()
    if output_dir.exists():
        raise FileExistsError('Test output already exists; choose --output-dir or move it: {}'.format(output_dir))
    manifest_path = source['run_dir'] / 'checkpoints.json'
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    alias_snapshot = {name: str((source['run_dir'] / manifest[name]['path']).resolve())
                      for name in CHECKPOINT_ALIASES
                      if isinstance(manifest.get(name), dict) and manifest[name].get('path')}
    if select_candidate:
        # A batch selects its own winner only after all models finish. A single
        # checkpoint test merely displays the aliases captured at its start.
        alias_snapshot.pop('selected', None)
    for candidate in sources:
        if candidate['config']['environment_settings'] != config['environment_settings']:
            raise ValueError('All candidate checkpoints must use the same environment settings.')
        if reward_settings(candidate['config']) != reward_settings(config):
            raise ValueError('All candidate checkpoints must use the same reward settings.')
        training = candidate['config']['training']
        _validate_seeds(candidate['config']['seed'], training['eval_seed'], training['eval_episodes'],
                        seed, True, episodes)
    suite, suite_path = load_scenarios(config['environment_settings'], seed, episodes,
                                       _scenario_path(source, seed, test_scenarios, episodes))

    from deep_rl_for_swarms.ma_envs.envs.point_envs.pursuit_evasion import PursuitEvasionEnv

    env = PursuitEvasionEnv(capture_bonus=reward_settings(config)['capture_bonus'],
                            **config['environment_settings'])
    random_state = np.random.get_state()
    rows = []
    created_output = False
    metadata = {'purpose': 'checkpoint_selection', 'status': 'running', 'seed': seed,
                'episodes': episodes, 'scenarios_path': str(suite_path), 'ranking': RANKING,
                'actions': 'policy_mean', 'max_episode_steps': 1024, 'reward': reward_settings(config),
                'capture_comparison': 'distance < capture_radius', 'capture_radius': env.capture_radius,
                'distance_geometry': 'torus' if env.torus else 'euclidean',
                'distance_definition': 'nearest pursuer to evader; minimum includes initial state; '
                                       'final is at capture or timeout; means include all scenarios',
                'checkpoints': [candidate['checkpoint'] for candidate in sources],
                'alias_snapshot': dict(alias_snapshot), 'results': [],
                'execution': {'workers': min(workers, len(sources) * episodes),
                              'backend': 'spawn_processes' if workers > 1 else 'serial',
                              'parallel_unit': 'checkpoint_scenario_chunks',
                              'tensorflow_threads_per_worker': 1, 'device': 'CPU',
                              'elapsed_seconds_definition': 'model first chunk start to last chunk end; '
                                                            'excludes process startup and initial queue wait'}}
    batch_start = time.perf_counter()
    try:
        for candidate in sources:
            validate_environment(candidate['config'], env)
        # Validate externally supplied initial states before creating report output.
        for scene in suite['scenarios']:
            env.reset(initial_state=scene['initial_state'])
        if reuse_report is not None:
            rows, reused_path = _reusable_results(reuse_report, suite, metadata, sources)
            metadata.update(reused_report=reused_path,
                            reused_checkpoints=[row['checkpoint'] for row in rows])
        reused = {row['checkpoint'] for row in rows}
        pending = [candidate for candidate in sources if candidate['checkpoint'] not in reused]
        metadata['execution']['workers'] = min(workers, len(pending) * episodes) if pending else 0
        active_workers = metadata['execution']['workers']
        metadata['execution']['backend'] = ('spawn_processes' if active_workers > 1
                                            else 'serial' if active_workers else 'reused_only')
        print('Validation: {} models, {} scenarios/model, {} workers, {} models reused'.format(
            len(sources), episodes, metadata['execution']['workers'], len(rows)), flush=True)
        output_dir.mkdir(parents=True, exist_ok=False)
        created_output = True
        atomic_json(output_dir / 'scenarios.json', suite)
        if rows:
            metadata['results'] = write_ranked_csv(output_dir / 'summary.csv', rows, alias_snapshot)
        atomic_json(output_dir / 'metadata.json', metadata)
        for candidate, reports, elapsed in evaluate_candidates(pending, suite, workers):
            rows.append(dict(summarize(reports), step=candidate['step'],
                             checkpoint=candidate['checkpoint'], elapsed_seconds=elapsed))
            metadata['results'] = write_ranked_csv(output_dir / 'summary.csv', rows, alias_snapshot)
            atomic_json(output_dir / 'metadata.json', metadata)
            print('Validation step {} finished: {}/{} captures'.format(
                candidate['step'], rows[-1]['success_count'], episodes), flush=True)
        ranked = write_ranked_csv(output_dir / 'summary.csv', rows, alias_snapshot)
        if select_candidate:
            alias_snapshot['selected'] = ranked[0]['checkpoint']
            ranked = write_ranked_csv(output_dir / 'summary.csv', rows, alias_snapshot)
        metadata.update(status='complete', alias_snapshot=dict(alias_snapshot), results=ranked,
                        batch_wall_seconds=time.perf_counter() - batch_start)
        atomic_json(output_dir / 'metadata.json', metadata)
        return {'output_dir': str(output_dir), 'rows': ranked, 'metadata': metadata}
    except BaseException as error:
        if created_output and (output_dir / 'metadata.json').exists():
            metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                            error=str(error) or type(error).__name__)
            atomic_json(output_dir / 'metadata.json', metadata)
        raise
    finally:
        env.close()
        np.random.set_state(random_state)


def test_all_checkpoints(run_dir, test_seed=None, test_episodes=None, test_scenarios=None, output_dir=None,
                         workers=None, reuse_report=None):
    from deep_rl_for_swarms.common.mappo_evaluation import atomic_json
    sources = discover_checkpoints(run_dir)
    result = _test_sources(sources, test_seed, test_episodes, test_scenarios, output_dir,
                           select_candidate=True, workers=workers, reuse_report=reuse_report)
    leader = result['rows'][0]
    manifest_path = sources[0]['run_dir'] / 'checkpoints.json'
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    source = next(candidate for candidate in sources if candidate['checkpoint'] == leader['checkpoint'])
    manifest['selected'] = dict(source['record'], validation_metrics=leader,
                                validation_report=os.path.relpath(result['output_dir'], str(source['run_dir'])))
    atomic_json(manifest_path, manifest)
    return result


def test_checkpoint(checkpoint, model='final', test_seed=None, test_scenarios=None,
                    test_episodes=None, output_dir=None, artifacts=False,
                    animation_fps=None, animation_stride=None, workers=None, reuse_report=None):
    if ((animation_fps is not None and animation_fps < 1)
            or (animation_stride is not None and animation_stride < 1)):
        raise ValueError('Animation fps and stride must be positive.')
    if artifacts:
        if (test_episodes not in (None, 5) or output_dir is not None
                or workers is not None or reuse_report is not None):
            raise ValueError('--artifacts uses five scenarios and legacy test/<step>/ output; '
                             '--workers and --reuse-report apply to headless validation only.')
        return export_checkpoint(checkpoint, model, test_seed, test_scenarios,
                                 animation_fps, animation_stride)
    source = resolve_checkpoint(checkpoint, model)
    if output_dir is None:
        saved = source['config'].get('test', {})
        current = saved.get('purpose') == 'checkpoint_selection'
        episodes = test_episodes if test_episodes is not None else saved.get('episodes', 200) if current else 200
        seed = test_seed if test_seed is not None else saved.get('seed', 30_000) if current else 30_000
        output_dir = source['run_dir'] / 'test' / str(source['step']) / 'validation_{}_seed{}'.format(episodes, seed)
    return _test_sources([source], test_seed, test_episodes, test_scenarios, output_dir,
                         workers=workers, reuse_report=reuse_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    resume = commands.add_parser('resume', help='Continue training into a new run directory.')
    test = commands.add_parser('test', help='Test a checkpoint into its run/test/<step>/ directory.')
    batch = commands.add_parser('test-all', help='Rank every saved checkpoint on the same 200 scenarios.')
    batch.add_argument('run_dir')
    for command in (resume, test):
        command.add_argument('checkpoint', help='Step directory, model prefix, or run directory.')
        command.add_argument('--model', choices=['best', 'final', 'selected'], default='final',
                             help='Selection when checkpoint is a run directory (default: final).')
    for command in (resume, test, batch):
        command.add_argument('--test-seed', type=int)
        command.add_argument('--test-scenarios')
        command.add_argument('--test-episodes', type=int, help='Selection scenarios (default: 200).')
    for command in (test, batch):
        command.add_argument('--output-dir', help='New output directory; existing reports are protected.')
        command.add_argument('--workers', type=int,
                             help='Parallel CPU processes (default: up to 8); 1 runs serially.')
        command.add_argument('--reuse-report', help='Reuse complete models from a matching report into a NEW output directory.')
    test.add_argument('--artifacts', action='store_true', help='Export the legacy five-scenario MP4/PNG/CSV suite.')
    test.add_argument('--animation-fps', type=int)
    test.add_argument('--animation-stride', type=int)
    resume.add_argument('--additional-timesteps', type=int, required=True,
                        help='Extra environment steps; 400000 + 600000 ends at 1000000.')
    resume.add_argument('--log-dir', help='Parent of new run; defaults to source run parent.')
    resume.add_argument('--note', default='', help='Free-text remark saved in config.json.')
    resume.add_argument('--seed', type=int)
    resume.add_argument('--test-workers', type=int, help='Post-training evaluation processes (default: up to 8).')
    resume.add_argument('--capture-bonus', type=float,
                        help='Shared terminal capture bonus; inherited by default, 1 enables the new reward, 0 restores legacy.')
    for name in ('rollout_steps', 'minibatch_steps', 'epochs', 'eval_interval', 'eval_episodes', 'eval_seed', 'checkpoint_interval'):
        resume.add_argument('--' + name.replace('_', '-'), type=int)
    for name in ('learning_rate', 'entropy_coef', 'gamma', 'lam', 'clip'):
        resume.add_argument('--' + name.replace('_', '-'), type=float)
    testing = resume.add_mutually_exclusive_group()
    testing.add_argument('--test', dest='test', action='store_true')
    testing.add_argument('--no-test', dest='test', action='store_false')
    resume.set_defaults(test=None)
    args = vars(parser.parse_args())
    command = args.pop('command')
    try:
        with _scipy_dll_directory():
            if command == 'resume':
                run_dir = resume_training(**args)
                print('Continuation saved to {}'.format(run_dir))
            elif command == 'test-all':
                result = test_all_checkpoints(**args)
                print('Ranked {} models: {}'.format(len(result['rows']), result['output_dir']))
            else:
                summary = test_checkpoint(**args)
                if 'rows' in summary:
                    print('Validation report: {}'.format(summary['output_dir']))
                else:
                    print('Step {}: success rate {:.1%}'.format(summary['checkpoint']['step'], summary['success_rate']))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
