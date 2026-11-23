"""Fixed pursuit scenarios and post-training test reports."""

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from deep_rl_for_swarms.common.mappo_visualization import save_animation, save_distance_plot


def load_test_scenarios(env_settings, seed, path=None):
    settings_hash = hashlib.sha256(json.dumps(env_settings, sort_keys=True).encode()).hexdigest()[:10]
    if path is None:
        path = Path('test_scenarios') / 'pursuit_{}_seed{}.json'.format(settings_hash, seed)
    path = Path(path)
    seeds = list(range(seed, seed + 5))
    if path.exists():
        with path.open() as source:
            suite = json.load(source)
        if (suite.get('schema_version') != 1 or suite.get('environment_settings') != env_settings
                or [scene['seed'] for scene in suite['scenarios']] != seeds):
            raise ValueError('Saved test scenarios must match the environment settings and test seeds.')
        return suite, path.resolve()

    scenarios = []
    for index, episode_seed in enumerate(seeds, 1):
        random = np.random.RandomState(episode_seed)
        pursuers = random.rand(env_settings['nr_pursuers'], 3)
        pursuers[:, :2] = env_settings['world_size'] * (0.9 * pursuers[:, :2] + 0.05)
        pursuers[:, 2] *= 2 * np.pi
        prey = env_settings['world_size'] * (0.9 * random.rand(1, 2) + 0.05)
        scenarios.append({'id': 'scenario_{:02d}'.format(index), 'seed': episode_seed,
                          'initial_state': {'pursuer_states': pursuers.tolist(),
                                            'evader_states': prey.tolist()}})
    suite = {'schema_version': 1, 'environment': 'PursuitEvasionEnv',
             'environment_settings': env_settings, 'scenarios': scenarios}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=str(path.parent), delete=False) as output:
        json.dump(suite, output, indent=2, allow_nan=False)
        temporary_path = output.name
    os.replace(temporary_path, str(path))
    return suite, path.resolve()


def run_tests(env, session, policy, run_dir, selection, suite, model='best', fps=20, frame_stride=4):
    import tensorflow as tf

    start_time = time.perf_counter()
    run_dir = Path(run_dir)
    checkpoint = selection[model]
    prefix = run_dir / checkpoint['path']
    # A standalone test graph needs only policy variables, not training/Adam ops.
    tf.train.Saver(var_list=policy.get_variables()).restore(session, str(prefix))
    output_dir = run_dir / 'test' / model
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / 'scenarios.json').open('w') as output:
        json.dump(suite, output, indent=2, allow_nan=False)

    reports = []
    random_state = np.random.get_state()
    try:
        for scene in suite['scenarios']:
            np.random.seed(scene['seed'])
            env.seed(scene['seed'])
            observations = env.reset(initial_state=scene['initial_state'])
            positions = [env.world.agent_states[:, :2].copy()]
            prey_positions = [env.world.landmark_states.copy()]
            distances = [env.world.distance_matrix[-1, :env.nr_agents].copy()]
            episode_return = 0.0
            capture_step = None
            for step in range(1, min(1024, env.timestep_limit) + 1):
                actions, _ = policy.act(False, np.vstack(observations))
                observations, reward, done, _ = env.step(actions)
                episode_return += float(np.mean(reward))
                positions.append(env.world.agent_states[:, :2].copy())
                prey_positions.append(env.world.landmark_states.copy())
                distances.append(env.world.distance_matrix[-1, :env.nr_agents].copy())
                if np.min(distances[-1]) < env.capture_radius:
                    capture_step = step
                if done:
                    break

            scene_dir = output_dir / scene['id']
            scene_dir.mkdir()
            distances = np.asarray(distances)
            nearest = distances.min(axis=1)
            with (scene_dir / 'distances.csv').open('w', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(['step'] + ['pursuer_{}'.format(i) for i in range(env.nr_agents)]
                                + ['nearest_distance'])
                for step, values in enumerate(distances):
                    writer.writerow([step] + values.tolist() + [float(nearest[step])])
            series = {'Pursuer {}'.format(i): distances[:, i] for i in range(env.nr_agents)}
            series['Nearest pursuer'] = nearest
            series['Capture threshold'] = np.full(len(distances), env.capture_radius)
            save_distance_plot(scene_dir / 'distances.png', series, 'Distance to prey')
            save_animation(scene_dir / 'animation.mp4', np.asarray(positions), env.world_size,
                           fps=fps, frame_stride=frame_stride,
                           prey_positions=np.asarray(prey_positions))
            report = {'scenario': scene['id'], 'seed': scene['seed'],
                      'steps': len(distances) - 1, 'episode_return': episode_return,
                      'captured': capture_step is not None, 'capture_step': capture_step,
                      'initial_nearest_distance': float(nearest[0]),
                      'minimum_nearest_distance': float(nearest.min()),
                      'final_nearest_distance': float(nearest[-1])}
            reports.append(report)
            print('Test {} completed, captured={}, steps={}'.format(
                scene['id'], report['captured'], report['steps']), flush=True)
    finally:
        np.random.set_state(random_state)

    with (output_dir / 'summary.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    summary = {'model': model, 'checkpoint': checkpoint, 'episodes': len(reports),
               'actions': 'policy_mean', 'fps': fps, 'frame_stride': frame_stride,
               'success_count': sum(report['captured'] for report in reports),
               'success_rate': float(np.mean([report['captured'] for report in reports])),
               'mean_episode_return': float(np.mean([report['episode_return'] for report in reports])),
               'capture_radius': env.capture_radius, 'capture_comparison': 'distance < capture_radius',
               'distance_geometry': 'torus' if env.torus else 'euclidean',
               'elapsed_seconds': time.perf_counter() - start_time, 'scenarios': reports}
    with (output_dir / 'summary.json').open('w') as output:
        json.dump(summary, output, indent=2, allow_nan=False)
    return summary
