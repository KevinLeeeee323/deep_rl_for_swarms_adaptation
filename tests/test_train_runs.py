"""Check run directories, elapsed semantics and checkpoint metadata end to end."""
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import pytest
import torch

from deep_rl_for_swarms.experiments.pursuit import train
from deep_rl_for_swarms.experiments.pursuit.common import load_model


def configure_run(tmp_path, monkeypatch):
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/pursuit/reference.json').read_text())
    config['algorithm'].update(hidden=8, rollout_steps=4, epochs=1, minibatch=4)
    config['experiment'].update(total_steps=8, eval_interval=4, validation_episodes=1)
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    parent = tmp_path/'runs'
    parent.mkdir()
    # Freeze calendar time so two launches with the same seed exercise name collisions.
    epoch = time.time()
    monkeypatch.setattr(train.time, 'time', lambda: epoch)
    monkeypatch.setattr(sys, 'argv', ['train', '--config', str(path), '--output', str(parent), '--seed', '3'])
    return parent, epoch


def test_run_directory_timing_and_earlier_best_are_preserved(tmp_path, monkeypatch):
    parent, epoch = configure_run(tmp_path, monkeypatch)
    clock = [0.]
    monkeypatch.setattr(train.time, 'monotonic', lambda: clock[0])
    scores = iter([0., 1., 0.])
    def evaluate(*args):
        clock[0] += 100.  # Validation time must not enter the current post-update elapsed.
        return dict(success_rate=next(scores), penalized_steps=1024., raw_return=-10.), [], []
    update = train.MAPPO.update
    def timed_update(model, rollout):
        metrics = update(model, rollout)
        clock[0] += 10.
        return metrics
    monkeypatch.setattr(train, 'evaluate', evaluate)
    monkeypatch.setattr(train.MAPPO, 'update', timed_update)
    train.main()

    stamp = datetime.fromtimestamp(epoch).strftime('%Y%m%d-%H%M%S')
    output = parent/f'mappo-{stamp}-seed3'
    assert list(parent.iterdir()) == [output]
    timing = json.loads((output/'timing.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['algorithm'] == 'mappo' and manifest['seed'] == 3
    assert manifest['timing'] == timing
    assert timing['status'] == 'completed' and timing['duration_seconds'] == 320.
    assert datetime.fromisoformat(timing['ended_at']) >= datetime.fromisoformat(timing['started_at'])
    saved = {}
    for kind, steps in [('untrained', 0), ('best', 4), ('latest', 8)]:
        _, saved[kind] = load_model(output/f'{kind}.pt')
        assert saved[kind]['steps'] == steps
        assert all(saved[kind]['timing'][key] == value for key, value in timing.items())
    assert sorted(path.name for path in output.glob('*.pt')) == ['best.pt', 'latest.pt', 'untrained.pt']
    assert any(not torch.equal(value, saved['latest']['model'][key])
               for key, value in saved['best']['model'].items())
    history = json.loads((output/'history.json').read_text())
    assert [row['elapsed'] for row in history] == [100., 110., 220.]
    assert saved['best']['timing']['checkpoint_elapsed_seconds'] == 210.
    assert saved['latest']['timing']['checkpoint_elapsed_seconds'] == 320.

    # A same-second, same-seed launch must fail without touching the first experiment.
    checkpoint_bytes = (output/'best.pt').read_bytes()
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
    assert (output/'best.pt').read_bytes() == checkpoint_bytes


def test_interrupted_run_records_end_time_in_checkpoint(tmp_path, monkeypatch):
    parent, _ = configure_run(tmp_path, monkeypatch)
    monkeypatch.setattr(train, 'evaluate', lambda *args: ({}, [], []))
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(train.MAPPO, 'update', interrupt)
    with pytest.raises(KeyboardInterrupt):
        train.main()
    output, = parent.iterdir()
    timing = json.loads((output/'timing.json').read_text())
    assert timing['status'] == 'interrupted' and timing['ended_at'] is not None
    _, saved = load_model(output/'untrained.pt')
    assert saved['steps'] == 0 and saved['timing']['ended_at'] == timing['ended_at']
    assert not (output/'best.pt').exists()
