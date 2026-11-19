import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from deep_rl_for_swarms.env_adapters.pursuit import PursuitAdapter
from deep_rl_for_swarms.algorithms.mappo.core import MAPPO

REFERENCE_COMMIT = '2e51997b30b49c2d288f841ba648cc1d193fe3b8'


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def provenance():
    def git(*args):
        return subprocess.check_output(['git', *args], text=True).strip()
    root = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'], text=True).strip())
    source = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        if any(part in ('runs', '.git', '__pycache__') for part in path.relative_to(root).parts):
            continue
        source.update(str(path.relative_to(root)).encode())
        source.update(path.read_bytes())
    return dict(source_sha256=source.hexdigest(), commit=git('rev-parse','HEAD'), branch=git('branch','--show-current'),
                dirty=bool(git('status','--porcelain')), reference_commit=REFERENCE_COMMIT)


def scenarios(config, count, seed, split):
    env = PursuitAdapter(config['environment'], seed)
    items = []
    for i in range(count):
        env.reset()
        items.append(dict(id=f'{split}-{i:04d}', **env.initial()))
    env.close()
    return dict(environment=config['environment'], split=split, seed=seed, scenarios=items)


def summarize(rows):
    success = [r['steps'] for r in rows if r['success']]
    return dict(episodes=len(rows), success_rate=float(np.mean([r['success'] for r in rows])),
                capture_steps=float(np.mean(success)) if success else None,
                penalized_steps=float(np.mean([r['steps'] if r['success'] else 1024 for r in rows])),
                raw_return=float(np.mean([r['raw_return'] for r in rows])))


def evaluate(config, suite, model=None, policy='model', seed=0, trajectories=False):
    if config['environment'] != suite['environment']:
        raise ValueError('Evaluation environment differs from the scenario suite')
    env = PursuitAdapter(config['environment'], seed)
    random = np.random.RandomState(seed)
    rows, traces = [], []
    for item in suite['scenarios']:
        obs, _ = env.reset(item)
        total = 0.0
        trace = dict(pursuers=[env.raw.world.agent_states.copy()],
                     prey=[env.raw.world.landmark_states.copy()], distances=[env.distances()])
        for t in range(1024):
            action = random.uniform(-1, 1, (env.num_agents, 2)) if policy == 'random' else model.act(obs, True)[0]
            obs, _, rewards, term, trunc, info = env.step(action)
            total += float(rewards[0])
            if trajectories:
                trace['pursuers'].append(env.raw.world.agent_states.copy())
                trace['prey'].append(env.raw.world.landmark_states.copy())
                trace['distances'].append(env.distances())
            if term or trunc:
                break
        rows.append(dict(scene_id=item['id'], success=term, steps=t+1,
                         raw_return=total, terminated=term, truncated=trunc,
                         final_min_distance=info['min_distance']))
        if trajectories:
            traces.append({k: np.asarray(v) for k,v in trace.items()})
    env.close()
    return summarize(rows), rows, traces


def checkpoint(path, model, config, steps, seed, validation_hash, timing=None):
    saved = dict(model=model.state_dict(), optimizer=model.optimizer.state_dict(),
                 config=config, steps=steps, seed=seed, validation_hash=validation_hash,
                 provenance=provenance())
    if timing is not None:
        saved['timing'] = dict(timing, checkpoint_saved_at=timing['recorded_at'],
                               checkpoint_elapsed_seconds=timing['duration_seconds'])
    torch.save(saved, path)


def update_checkpoint_timing(path, timing):
    """Add run-end metadata, keeping this checkpoint's parameters and step count."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    saved['timing'].update(timing)
    torch.save(saved, path)


def load_model(path, device='cpu'):
    # Only load trusted local project checkpoints.
    saved = torch.load(path, map_location=device, weights_only=False)
    env = PursuitAdapter(saved['config']['environment'])
    model = MAPPO(env.observation_space.shape[0], env.state_dim, saved['config']['algorithm'], device)
    model.load_state_dict(saved['model'])
    model.eval()
    env.close()
    return model, saved
