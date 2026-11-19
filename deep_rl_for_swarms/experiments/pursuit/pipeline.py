"""Always train, then optionally evaluate and visualize the resulting best model."""
import argparse
import json
from pathlib import Path

from . import train
from .artifacts import PROJECT_ROOT
from .postprocess import Workflow, add_options, preflight, process_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT_ROOT/'configs/pursuit/reference.json'))
    parser.add_argument('--output', default=str(PROJECT_ROOT/'runs'))
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    add_options(parser)
    args = parser.parse_args(argv)
    # Reject an incompatible fixed suite/baseline before spending time on training.
    preflight(args, json.loads(Path(args.config).read_text()))
    workflow = Workflow()
    error = None
    try:
        command = ['--config', args.config, '--output', args.output,
                   '--seed', str(args.seed), '--device', args.device]
        if args.steps is not None:
            command += ['--steps', str(args.steps)]
        workflow.stage('training', lambda: train.main(command, on_created=workflow.attach))
        process_run(args, workflow)
    except BaseException as caught:
        error = caught
        raise
    finally:
        workflow.finish(error)
    print(f'Results: {workflow.output / "summary.md"}', flush=True)
    return workflow.output


if __name__ == '__main__':
    main()
