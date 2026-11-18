import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .common import scenarios, save_json, evaluate, load_model, digest, provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/pursuit/reference.json')
    p.add_argument('--suite', required=True)
    p.add_argument('--generate', type=int, help='Generate this many initial states, then exit')
    p.add_argument('--split', choices=['validation','test','demo'], default='test')
    p.add_argument('--seed', type=int, default=200000)
    p.add_argument('--checkpoint')
    p.add_argument('--random', action='store_true')
    p.add_argument('--output')
    p.add_argument('--trajectories', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text())
    if args.generate is not None:
        if args.generate < 1 or Path(args.suite).exists():
            p.error('positive count and a new suite path required')
        save_json(args.suite, scenarios(config, args.generate, args.seed, args.split))
        return
    if not args.output or (not args.random and not args.checkpoint):
        p.error('Provide --output and either --random or --checkpoint')
    model, saved = (None, None) if args.random else load_model(args.checkpoint)
    if saved:
        config = saved['config']
    suite = json.loads(Path(args.suite).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    summary, rows, traces = evaluate(config, suite, model,
        policy='random' if args.random else 'model', seed=args.seed, trajectories=args.trajectories)
    save_json(output/'results.json', dict(summary=summary, episodes=rows,
        suite_hash=digest(suite), config=config, provenance=provenance(),
        policy='random' if args.random else 'deterministic_actor', action_seed=args.seed,
        checkpoint=str(args.checkpoint), checkpoint_sha256=(hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
            if args.checkpoint and not args.random else None)))
    for i, trace in enumerate(traces):
        np.savez_compressed(output/f'trace-{i:04d}.npz', **trace)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
