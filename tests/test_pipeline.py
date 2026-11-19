"""Exercise orchestration, cache safety, resuming, and both artifact layouts."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import pytest
import torch

from deep_rl_for_swarms.algorithms.mappo.core import MAPPO
from deep_rl_for_swarms.experiments.pursuit import pipeline, postprocess, render, audit, compare
from deep_rl_for_swarms.experiments.pursuit.artifacts import DEFAULT_SUITE, trace_path
from deep_rl_for_swarms.experiments.pursuit.common import checkpoint, save_json
from deep_rl_for_swarms.experiments.pursuit.evaluate import (
    evaluate_directory, evaluation_identity, validate_cached)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'configs/pursuit/reference.json').read_text())


def make_run(root):
    root.mkdir()
    save_json(root/'config.json', CONFIG)
    save_json(root/'timing.json', dict(status='completed', duration_seconds=1.))
    (root/'metrics.jsonl').write_text(json.dumps(dict(steps=8, validation=dict(
        success_rate=0., raw_return=-1.)))+'\n')
    checkpoint(root/'best.pt', MAPPO(503, 42, CONFIG['algorithm']), CONFIG, 8, 0, 'validation')
    return root


def options(run, suite, baseline, **kwargs):
    args = argparse.Namespace(run=str(run), suite=str(suite), random_output=str(baseline),
        action_seed=200000, skip_evaluation=False, skip_visualization=False,
        skip_videos=False, stride=1024, fps=2)
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def test_frozen_suite_is_original_twenty():
    suite = json.loads(DEFAULT_SUITE.read_text())
    assert suite == json.loads((ROOT/'docs/verification/test-suite.json').read_text())
    assert len(suite['scenarios']) == 20 and suite['seed'] == 200000


@pytest.fixture(scope='module')
def actual_evaluations(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp('evaluations')
    suite = json.loads(DEFAULT_SUITE.read_text())
    suite['scenarios'] = suite['scenarios'][:1]
    for layout in ('scenes', 'flat'):
        evaluate_directory(CONFIG, suite, root/layout, layout=layout)
    save_json(root/'suite.json', suite)
    return root, suite


def test_cache_checks_identities_and_trajectory_integrity(actual_evaluations, tmp_path):
    root, suite = actual_evaluations
    identity = evaluation_identity(CONFIG, suite)
    result = validate_cached(root/'scenes', identity, suite)
    assert result['summary']['episodes'] == 1
    for key, wrong in [('action_seed', 5), ('environment_signature', 'changed'),
                       ('suite_hash', 'changed'), ('policy', 'model')]:
        with pytest.raises(ValueError, match='Cannot reuse'):
            validate_cached(root/'scenes', dict(identity, **{key: wrong}), suite)
    import shutil
    broken = tmp_path/'broken'
    shutil.copytree(root/'scenes', broken)
    trace_path(broken, 0, result).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='trajectory'):
        validate_cached(broken, identity, suite)
    with pytest.raises(ValueError, match='differs'):
        evaluation_identity(dict(CONFIG, environment={}), suite)


def test_render_audit_and_compare_support_both_layouts(actual_evaluations, tmp_path):
    root, _ = actual_evaluations
    for layout in ('scenes', 'flat'):
        folder = render.render_scene(root/layout, stride=1024, fps=2, video=True)
        assert (folder/'distance-0000.png').exists()
        movie = folder/'episode-0000.mp4'
        with imageio.get_reader(movie) as reader:
            assert reader.get_data(0).shape == (640, 640, 3)
        before = movie.stat().st_mtime_ns
        render.render_scene(root/layout, stride=1024, fps=2, video=True, reuse=True)
        assert movie.stat().st_mtime_ns == before
    audit.main(['--suite',str(root/'suite.json'),'--evaluations',str(root/'scenes'),
                str(root/'flat'),'--output',str(tmp_path/'audit.json')])
    compare.main(['--before',str(root/'flat'),'--after',str(root/'scenes'),
                  '--stride','1024','--fps','2','--output',str(tmp_path/'comparison.mp4')])
    assert (tmp_path/'comparison.mp4').stat().st_size > 0


@pytest.mark.parametrize('skip_eval,skip_vis', [(False,False),(False,True),(True,False),(True,True)])
def test_pipeline_always_trains_and_honors_skips(tmp_path, monkeypatch, skip_eval, skip_vis):
    calls = []
    run = tmp_path/'run'
    def training(argv, on_created):
        calls.append('training')
        make_run(run)
        on_created(run)
        return run
    monkeypatch.setattr(pipeline.train, 'main', training)
    monkeypatch.setattr(pipeline, 'preflight', lambda *a: None)
    monkeypatch.setattr(postprocess, 'preflight', lambda *a: {'scenarios':[{}]})
    def evaluation(*args, **kwargs):
        calls.append('evaluation')
        return dict(suite_hash='fixed',summary=dict(episodes=1,success_rate=0.,capture_steps=None,
                    penalized_steps=1024.,raw_return=-1.))
    monkeypatch.setattr(postprocess, 'evaluate_directory', evaluation)
    # Check visual selection without encoding video in these orchestration cases.
    monkeypatch.setattr(postprocess.plot, 'main', lambda *a: calls.append('plot'))
    # Synthetic evaluator still publishes enough data for render enumeration.
    original_evaluation = evaluation
    def publish(config, suite, output, **kwargs):
        result = original_evaluation(config,suite,output,**kwargs)
        save_json(Path(output)/'results.json', dict(episodes=[]))
        return result
    monkeypatch.setattr(postprocess, 'evaluate_directory', publish)
    args=['--config',str(ROOT/'configs/pursuit/reference.json'),
          '--random-output',str(tmp_path/'random'),'--output',str(tmp_path)]
    if skip_eval: args+=['--skip-evaluation']
    if skip_vis: args+=['--skip-visualization']
    assert pipeline.main(args) == run
    assert calls.count('training') == 1
    assert calls.count('evaluation') == (0 if skip_eval else 2)
    assert calls.count('plot') == (0 if skip_vis else 1)
    report=json.loads((run/'pipeline.json').read_text())
    assert report['status']=='completed'


def test_postprocess_failure_preserves_checkpoint_and_can_resume(tmp_path, monkeypatch, actual_evaluations):
    source, suite = actual_evaluations
    run=make_run(tmp_path/'run')
    args=options(run, source/'suite.json', tmp_path/'random', skip_videos=True)
    # Real evaluator runs once, then renders can fail independently of training.
    def fail(*a, **kw):
        raise RuntimeError('encoder fixture failure')
    original=postprocess.render.render_scene
    monkeypatch.setattr(postprocess.render, 'render_scene', fail)
    command=['--run',str(run),'--suite',str(source/'suite.json'),
             '--random-output',str(args.random_output),'--skip-videos','--stride','1024']
    weights=(run/'best.pt').read_bytes()
    with pytest.raises(RuntimeError, match='encoder fixture'):
        postprocess.main(command)
    assert (run/'best.pt').read_bytes()==weights
    assert json.loads((run/'postprocess.json').read_text())['status']=='failed'
    baseline=(Path(args.random_output)/'results.json').read_bytes()
    monkeypatch.setattr(postprocess.render, 'render_scene', original)
    # Resuming must read cached trajectories, never run another model/random episode.
    import deep_rl_for_swarms.experiments.pursuit.evaluate as evaluator
    monkeypatch.setattr(evaluator, 'evaluate', lambda *a,**kw: pytest.fail('unexpected new rollout'))
    # environment_signature fingerprints the production evaluator in common, not this test hook.
    postprocess.main(command)
    assert (Path(args.random_output)/'results.json').read_bytes()==baseline
    assert (run/'best.pt').read_bytes()==weights
    assert json.loads((run/'postprocess.json').read_text())['status']=='completed'
    assert not (run/'eval_untrained').exists()
    assert (run/'eval_best/scene-0000/distance-0000.png').exists()
    assert not (run/'eval_best/scene-0000/episode-0000.mp4').exists()


def test_baseline_lock_and_partial_cache(tmp_path):
    baseline=tmp_path/'random'
    with postprocess.baseline_lock(baseline):
        with pytest.raises(RuntimeError,match='busy'):
            with postprocess.baseline_lock(baseline):
                pass
    baseline.mkdir()
    with pytest.raises(ValueError,match='Cannot reuse'):
        validate_cached(baseline,{}, {'scenarios':[]})


def test_incompatible_suite_fails_before_training(tmp_path, monkeypatch):
    suite=json.loads(DEFAULT_SUITE.read_text())
    suite['environment']['world_size']=50
    path=tmp_path/'wrong.json'
    save_json(path,suite)
    monkeypatch.setattr(pipeline.train,'main',lambda *a,**kw: pytest.fail('training must not start'))
    with pytest.raises(ValueError,match='differs'):
        pipeline.main(['--suite',str(path),'--output',str(tmp_path/'runs')])
    assert not (tmp_path/'runs').exists()


def test_interrupted_training_is_reported_without_postprocessing(tmp_path, monkeypatch):
    run=tmp_path/'run'
    def interrupt(argv,on_created):
        make_run(run)
        on_created(run)
        raise KeyboardInterrupt
    monkeypatch.setattr(pipeline.train,'main',interrupt)
    with pytest.raises(KeyboardInterrupt):
        pipeline.main(['--skip-evaluation','--skip-visualization'])
    report=json.loads((run/'pipeline.json').read_text())
    assert report['status']=='interrupted'
    assert report['stages']['training']['status']=='interrupted'
    assert set(report['stages'])=={'training'}
    assert not (run/'eval_best').exists()
