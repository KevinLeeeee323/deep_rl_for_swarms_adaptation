"""Reusable pursuit scenarios and headless policy evaluation metrics."""

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


RANKING = [('success_rate', 'descending'), ('mean_minimum_distance', 'ascending'),
           ('mean_final_distance', 'ascending')]
CHECKPOINT_ALIASES = ('best', 'selected', 'final')


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=str(path.parent),
                                         delete=False) as output:
            temporary = output.name
            json.dump(value, output, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, str(path))
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def make_scenarios(env_settings, seed, episodes, placement_scale=0.9):
    if episodes < 1 or not 0 <= seed <= 2**32 - episodes:
        raise ValueError('Scenario count must be positive and seeds within the NumPy range.')
    if env_settings['nr_evaders'] != 1:
        raise ValueError('PursuitEvasionEnv evaluation supports exactly one evader.')
    scenarios = []
    for index, episode_seed in enumerate(range(seed, seed + episodes), 1):
        random = np.random.RandomState(episode_seed)
        pursuers = random.rand(env_settings['nr_pursuers'], 3)
        pursuers[:, :2] = env_settings['world_size'] * (placement_scale * pursuers[:, :2] + 0.05)
        pursuers[:, 2] *= 2 * np.pi
        prey = env_settings['world_size'] * (placement_scale * random.rand(1, 2) + 0.05)
        scenarios.append({'id': 'scenario_{:02d}'.format(index), 'seed': episode_seed,
                          'initial_state': {'pursuer_states': pursuers.tolist(),
                                            'evader_states': prey.tolist()}})
    return {'schema_version': 1, 'environment': 'PursuitEvasionEnv',
            'environment_settings': env_settings, 'scenarios': scenarios}


def load_scenarios(env_settings, seed, episodes, path=None):
    if episodes < 1 or not 0 <= seed <= 2**32 - episodes:
        raise ValueError('Scenario count must be positive and seeds within the NumPy range.')
    settings_hash = hashlib.sha256(json.dumps(env_settings, sort_keys=True).encode()).hexdigest()[:10]
    if path is None:
        suffix = '' if episodes == 5 else '_{}episodes'.format(episodes)
        path = Path('test_scenarios') / 'pursuit_{}_seed{}{}.json'.format(settings_hash, seed, suffix)
    path = Path(path)
    if path.exists():
        with path.open(encoding='utf-8-sig') as source:
            suite = json.load(source)
        if (suite.get('schema_version') != 1 or suite.get('environment') != 'PursuitEvasionEnv'
                or suite.get('environment_settings') != env_settings
                or [scene['seed'] for scene in suite['scenarios']] != list(range(seed, seed + episodes))
                or len({scene['id'] for scene in suite['scenarios']}) != episodes):
            raise ValueError('Saved scenarios must match environment, episode count and seeds.')
    else:
        suite = make_scenarios(env_settings, seed, episodes)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, suite)
    return suite, path.resolve()


def evaluate_scenarios(env, action_fn, suite, progress=None):
    """No rendering or trajectory exports; distances include the initial state.

    Capture is checked after each environment step, including step 1024, exactly
    as in the environment. Distances come from its torus-aware distance matrix.
    """
    reports = []
    random_state = np.random.get_state()
    try:
        for index, scene in enumerate(suite['scenarios'], 1):
            np.random.seed(scene['seed'])
            env.seed(scene['seed'])
            observations = env.reset(initial_state=scene['initial_state'])
            initial_distance = float(np.min(env.world.distance_matrix[-1, :env.nr_agents]))
            minimum_distance = initial_distance
            episode_return = 0.0
            capture_step = None
            for step in range(1, min(1024, env.timestep_limit) + 1):
                observations, rewards, done, _ = env.step(action_fn(observations))
                episode_return += float(np.mean(rewards))
                final_distance = float(np.min(env.world.distance_matrix[-1, :env.nr_agents]))
                minimum_distance = min(minimum_distance, final_distance)
                if final_distance < env.capture_radius:
                    capture_step = step
                if done or capture_step is not None:
                    break
            report = {'scenario': scene['id'], 'seed': scene['seed'], 'steps': step,
                      'episode_return': episode_return, 'captured': capture_step is not None,
                      'capture_step': capture_step, 'initial_nearest_distance': initial_distance,
                      'minimum_nearest_distance': minimum_distance,
                      'final_nearest_distance': final_distance}
            if not all(np.isfinite(report[key]) for key in (
                    'episode_return', 'initial_nearest_distance', 'minimum_nearest_distance',
                    'final_nearest_distance')):
                raise ValueError('Non-finite evaluation metric in {}'.format(scene['id']))
            reports.append(report)
            if progress is not None:
                progress(index, len(suite['scenarios']))
    finally:
        np.random.set_state(random_state)
    return reports


def summarize(reports):
    if not reports:
        raise ValueError('Evaluation needs at least one scenario.')
    episodes = len(reports)
    successes = sum(report['captured'] for report in reports)
    rate = successes / episodes
    # Wilson interval describes scenario uncertainty for this fixed policy.
    z = 1.959963984540054
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes**2)) / denominator
    return {'episodes': episodes, 'success_count': int(successes), 'success_rate': rate,
            'success_rate_ci95_low': max(0.0, float(center - half)),
            'success_rate_ci95_high': min(1.0, float(center + half)),
            'mean_minimum_distance': float(np.mean([r['minimum_nearest_distance'] for r in reports])),
            'mean_final_distance': float(np.mean([r['final_nearest_distance'] for r in reports])),
            'mean_episode_return': float(np.mean([r['episode_return'] for r in reports])),
            'std_episode_return': float(np.std([r['episode_return'] for r in reports])),
            'mean_episode_steps': float(np.mean([r['steps'] for r in reports])),
            'mean_capture_steps_success': (float(np.mean([r['capture_step'] for r in reports
                                                         if r['captured']])) if successes else None)}


def rank_models(rows):
    ordered = sorted(rows, key=lambda row: (-row['success_rate'], row['mean_minimum_distance'],
                                          row['mean_final_distance'], row['step'], row['checkpoint']))
    ranked = []
    previous = None
    rank = 0
    for index, row in enumerate(ordered, 1):
        score = tuple(row[key] for key, _ in RANKING)
        if score != previous:
            rank = index
        ranked.append(dict(row, rank=rank))
        previous = score
    return ranked


def write_ranked_csv(path, rows, alias_snapshot=None):
    """Export aliases from this report's snapshot, never from a live manifest."""
    rows = rank_models(rows)
    if not rows:
        raise ValueError('No evaluated checkpoints to write.')
    if alias_snapshot is not None:
        alias_paths = {name: os.path.normcase(os.path.realpath(prefix))
                       for name, prefix in alias_snapshot.items()
                       if name in CHECKPOINT_ALIASES and prefix}
        rows = [dict(row, aliases=';'.join(
            name for name in CHECKPOINT_ALIASES
            if alias_paths.get(name) == os.path.normcase(os.path.realpath(row['checkpoint']))))
                for row in rows]
    else:
        rows = [dict(row, aliases=row.get('aliases', '')) for row in rows]
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8-sig',
                                         dir=str(path.parent), delete=False) as output:
            temporary = output.name
            fields = ['rank', 'step', 'aliases', 'checkpoint'] + [key for key in rows[0]
                                                       if key not in ('rank', 'step', 'aliases', 'checkpoint')]
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, str(path))
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return rows
