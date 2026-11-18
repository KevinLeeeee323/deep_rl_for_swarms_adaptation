"""python -m deep_rl_for_swarms.experiments.pursuit.train --help"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from deep_rl_for_swarms.env_adapters.pursuit import PursuitAdapter
from deep_rl_for_swarms.algorithms.mappo.core import MAPPO
from .common import save_json, scenarios, evaluate, checkpoint, digest, provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/pursuit/reference.json')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['cpu','mps'], default='cpu')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.steps is not None:
        config['experiment']['total_steps'] = args.steps
    if config['experiment']['total_steps'] <= 0:
        p.error('steps must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = PursuitAdapter(config['environment'], args.seed)
    model = MAPPO(env.observation_space.shape[0], env.state_dim, config['algorithm'], args.device)
    suite = scenarios(config, config['experiment']['validation_episodes'], 100000, 'validation')
    save_json(output/'config.json', config)
    save_json(output/'validation.json', suite)
    save_json(output/'manifest.json', dict(seed=args.seed, device=args.device, provenance=provenance(),
        validation_hash=digest(suite), config_hash=digest(config)))
    obs, state = env.reset()
    total, last_eval, best, history = 0, 0, None, []
    start = time.monotonic()
    with (output/'metrics.jsonl').open('w', buffering=1) as log:
        initial, _, _ = evaluate(config, suite, model)
        row = dict(steps=0, validation=initial, elapsed=0.0)
        log.write(json.dumps(row)+'\n')
        history.append(row)
        checkpoint(output/'untrained.pt', model, config, 0, args.seed, digest(suite))
        while total < config['experiment']['total_steps']:
            rollout = {k: [] for k in ('obs','states','latent','log_prob','rewards','values',
                                      'next_values','terminated','truncated')}
            length = min(config['algorithm']['rollout_steps'], config['experiment']['total_steps']-total)
            for _ in range(length):
                action, latent, logp = model.act(obs)
                value = float(model.value(state))
                next_obs, next_state, rewards, term, trunc, _ = env.step(action)
                transition = dict(obs=obs.copy(), states=state.copy(), latent=latent, log_prob=logp,
                    rewards=float(rewards[0]), values=value, next_values=float(model.value(next_state)),
                    terminated=term, truncated=trunc)
                for k,v in transition.items():
                    rollout[k].append(v)
                obs, state = env.reset() if term or trunc else (next_obs, next_state)
                total += 1
            metrics = model.update(rollout)
            row = dict(steps=total, elapsed=time.monotonic()-start, **metrics)
            if total-last_eval >= config['experiment']['eval_interval'] or total == config['experiment']['total_steps']:
                result, _, _ = evaluate(config, suite, model)
                row['validation'] = result
                score = (result['success_rate'], -result['penalized_steps'], result['raw_return'])
                if best is None or score > best:
                    best = score
                    checkpoint(output/'best.pt', model, config, total, args.seed, digest(suite))
                last_eval = total
            log.write(json.dumps(row)+'\n')
            history.append(row)
            checkpoint(output/'latest.pt', model, config, total, args.seed, digest(suite))
            print(json.dumps(row), flush=True)
    save_json(output/'history.json', history)
    env.close()


if __name__ == '__main__':
    main()
