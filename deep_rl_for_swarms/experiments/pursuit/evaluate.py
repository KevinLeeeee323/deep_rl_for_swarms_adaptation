import argparse
import json
from pathlib import Path
import numpy as np
import torch
import tempfile
from .common import scenarios, save_json, evaluate, load_model, digest, provenance
from .artifacts import environment_signature, file_hash, trace_path, scene_directory


def evaluation_identity(config, suite, checkpoint=None, seed=200000):
    if config['environment'] != suite['environment']:
        raise ValueError('Evaluation environment differs from the fixed suite; use a matching suite/baseline.')
    return dict(suite_hash=digest(suite), environment_signature=environment_signature(),
                action_seed=seed, policy='deterministic_actor' if checkpoint else 'random',
                checkpoint_sha256=file_hash(checkpoint) if checkpoint else None)


def validate_cached(output, identity, suite, trajectories=True):
    output = Path(output)
    try:
        result = json.loads((output/'results.json').read_text())
        if any(result.get(k) != v for k, v in identity.items()):
            raise ValueError('scene, environment, policy, seed or checkpoint identity differs')
        if [r['scene_id'] for r in result['episodes']] != [s['id'] for s in suite['scenarios']]:
            raise ValueError('scene list differs')
        if trajectories:
            for i, row in enumerate(result['episodes']):
                if file_hash(trace_path(output, i, result)) != row.get('trace_sha256'):
                    raise ValueError(f'trajectory {i} is missing or changed')
        return result
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f'Cannot reuse {output}: {error}. Use another output/baseline directory.') from error


def evaluate_directory(config, suite, output, checkpoint=None, seed=200000,
                       trajectories=True, layout='scenes', reuse=False):
    """Publish complete evaluation data atomically; never overwrite an old evaluation."""
    output = Path(output)
    identity = evaluation_identity(config, suite, checkpoint, seed)
    if output.exists():
        if reuse:
            return validate_cached(output, identity, suite, trajectories)
        raise FileExistsError(output)
    model = None
    if checkpoint:
        model, saved = load_model(checkpoint)
        if saved['config'] != config:
            raise ValueError('Use the checkpoint configuration for model evaluation')
    summary, rows, traces = evaluate(config, suite, model,
        policy='model' if checkpoint else 'random', seed=seed, trajectories=trajectories)
    result = dict(summary=summary, episodes=rows, config=config, provenance=provenance(),
                  checkpoint=str(checkpoint) if checkpoint else None, layout=layout, **identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        staging = Path(temporary)/'evaluation'
        staging.mkdir()
        for i, trace in enumerate(traces):
            scene_directory(staging, i, result).mkdir(parents=True, exist_ok=True)
            path = trace_path(staging, i, result)
            np.savez_compressed(path, **trace)
            rows[i]['trace_sha256'] = file_hash(path)
        save_json(staging/'results.json', result)
        if output.exists():
            raise FileExistsError(output)
        staging.rename(output)
    return result


def main(argv=None):
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
    p.add_argument('--layout', choices=['scenes', 'flat'], default='scenes')
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text())
    if args.generate is not None:
        if args.generate < 1 or Path(args.suite).exists():
            p.error('positive count and a new suite path required')
        save_json(args.suite, scenarios(config, args.generate, args.seed, args.split))
        return
    if not args.output or (not args.random and not args.checkpoint):
        p.error('Provide --output and either --random or --checkpoint')
    if args.random and args.checkpoint:
        p.error('Choose either --random or --checkpoint')
    if not args.random:
        config = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['config']
    suite = json.loads(Path(args.suite).read_text())
    result = evaluate_directory(config, suite, args.output,
        checkpoint=None if args.random else args.checkpoint, seed=args.seed,
        trajectories=args.trajectories, layout=args.layout)
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
