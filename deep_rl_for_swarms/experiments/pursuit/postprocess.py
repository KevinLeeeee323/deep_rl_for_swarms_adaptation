"""Evaluate and visualize an existing run, without training it again."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch

from .artifacts import DEFAULT_SUITE, PROJECT_ROOT, scene_directory
from .common import save_json
from .evaluate import evaluate_directory, evaluation_identity, validate_cached
from . import plot, render


def now():
    return datetime.now().astimezone().isoformat(timespec='seconds')


class Workflow:
    def __init__(self, filename='pipeline.json'):
        self.output = None
        self.filename = filename
        self.start = time.monotonic()
        self.record = dict(started_at=now(), status='running', stages={})

    def attach(self, output):
        self.output = Path(output)
        self.save()

    def save(self):
        if self.output is not None:
            save_json(self.output/self.filename, self.record)

    def stage(self, name, function):
        started = time.monotonic()
        entry = dict(status='running', started_at=now())
        self.record['stages'][name] = entry
        self.save()
        print(f'[{name}] starting', flush=True)
        try:
            value = function()
            entry['status'] = 'completed'
            return value
        except BaseException as error:
            entry.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                         error=f'{type(error).__name__}: {error}')
            raise
        finally:
            entry.update(ended_at=now(), duration_seconds=time.monotonic()-started)
            self.save()

    def finish(self, error=None):
        self.record.update(status=('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
                           if error else 'completed', ended_at=now(),
                           duration_seconds=time.monotonic()-self.start)
        if error:
            self.record['error'] = f'{type(error).__name__}: {error}'
        self.save()
        if self.output is not None:
            write_summary(self.output, self.record)


@contextmanager
def baseline_lock(output):
    """Do not let concurrent pipelines publish/render the same shared baseline."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent/f'.{output.name}.lock'
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f'Baseline is busy: {lock}. Retry postprocess after the other run finishes; '
                           'if it was forcibly killed, check before removing the stale lock.') from error
    try:
        yield
    finally:
        lock.rmdir()


def add_options(parser):
    parser.add_argument('--suite', default=str(DEFAULT_SUITE))
    parser.add_argument('--random-output', default=str(PROJECT_ROOT/'runs/eval_random'))
    parser.add_argument('--action-seed', type=int, default=200000)
    parser.add_argument('--skip-evaluation', action='store_true')
    parser.add_argument('--skip-visualization', action='store_true')
    parser.add_argument('--skip-videos', action='store_true')
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--fps', type=int, default=15)


def preflight(args, config):
    if args.stride < 1 or args.fps < 1:
        raise ValueError('stride and fps must be positive')
    if args.skip_evaluation:
        return None
    suite = json.loads(Path(args.suite).read_text())
    if not suite.get('scenarios'):
        raise ValueError('Evaluation suite must contain scenes')
    identity = evaluation_identity(config, suite, seed=args.action_seed)
    if Path(args.random_output).exists():
        validate_cached(args.random_output, identity, suite)
    return suite


def process_run(args, workflow):
    output = workflow.output
    config = json.loads((output/'config.json').read_text())
    suite = preflight(args, config)
    evaluations = []
    if not args.skip_evaluation:
        best = output/'best.pt'
        saved = torch.load(best, map_location='cpu', weights_only=False)
        model_result = workflow.stage('model_evaluation', lambda: evaluate_directory(
            saved['config'], suite, output/'eval_best', checkpoint=best,
            seed=args.action_seed, reuse=True))
        def random_baseline():
            with baseline_lock(args.random_output):
                return evaluate_directory(config, suite, args.random_output,
                                          seed=args.action_seed, reuse=True)
        random_result = workflow.stage('random_baseline', random_baseline)
        evaluations = [output/'eval_best', Path(args.random_output)]
        workflow.record['evaluation'] = dict(suite=str(Path(args.suite).resolve()),
            suite_hash=model_result['suite_hash'], best_steps=saved['steps'],
            best=model_result['summary'], random=random_result['summary'],
            random_output=str(Path(args.random_output).resolve()))
        workflow.save()
    else:
        for stage in ('model_evaluation', 'random_baseline'):
            workflow.record['stages'][stage] = dict(status='skipped')
    if args.skip_visualization:
        workflow.record['stages']['visualization'] = dict(status='skipped')
        workflow.save()
        return

    def visualize():
        plot_args = ['--runs', str(output), '--output', str(output/'figures')]
        if evaluations:
            plot_args += ['--evaluations', *map(str, evaluations)]
        plot.main(plot_args)
        for root in evaluations:
            def render_all():
                result = json.loads((root/'results.json').read_text())
                for index in range(len(result['episodes'])):
                    print(f'[render] {root.name} scene {index+1}/{len(result["episodes"])}', flush=True)
                    render.render_scene(root, index, args.stride, args.fps,
                                        video=not args.skip_videos, reuse=True)
            if root == Path(args.random_output):
                with baseline_lock(root):
                    render_all()
            else:
                render_all()
    workflow.stage('visualization', visualize)


def write_summary(output, record):
    lines = ['# 本次运行结果', '', f'流程状态：{record["status"]}。', '']
    timing = output/'timing.json'
    if timing.exists():
        t = json.loads(timing.read_text())
        lines += [f'训练状态：{t["status"]}；训练用时 {t["duration_seconds"]:.2f} 秒。', '']
    lines += [f'本流程用时：{record["duration_seconds"]:.2f} 秒（包含本次执行的后处理）。', '']
    if record.get('error'):
        lines += [f'错误：`{record["error"]}`；已完成的模型和评估数据保留，可使用 postprocess 补做。', '']
    figures = [f'[{label}]({path})' for label, path in
               [('学习曲线', 'figures/learning.png'), ('捕获率对照', 'figures/evaluation.png')]
               if (output/path).exists()]
    if figures:
        lines += [' · '.join(figures), '']
    result = record.get('evaluation')
    if result:
        lines += [f'best.pt 对应 {result["best_steps"]} 环境步；固定评估场景见 `{result["suite"]}`。', '',
                  '| 策略 | 场景数 | 捕获率 | 成功条件平均步数 | 失败按1024计入用时 | 平均原始回报 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name in ('best', 'random'):
            row = result[name]
            capture = 'N/A' if row['capture_steps'] is None else f'{row["capture_steps"]:.2f}'
            lines.append(f'| {name} | {row["episodes"]} | {row["success_rate"]:.1%} | {capture} | '
                         f'{row["penalized_steps"]:.2f} | {row["raw_return"]:.3f} |')
        random_path = os.path.relpath(result['random_output'], output)
        lines += ['', '[模型评估记录](eval_best/results.json) · '
                  f'[共享随机对照]({random_path}/results.json)', '',
                  '场景目录位于 eval_best/scene-XXXX/；图表位于 figures/。', '',
                  '这些固定场景用于日常比较和调参，不代表独立的最终泛化测试。'
                  'best.pt 仍仅由训练内验证集选择；一次固定随机策略运行是可复现参考。', '']
        evaluation = json.loads((output/'eval_best/results.json').read_text())
        lines += ['## 模型逐场景结果', '',
                  '| 场景 | 成功 | 步数 | 原始回报 | 文件 |',
                  '| --- | --- | ---: | ---: | --- |']
        for index, row in enumerate(evaluation['episodes']):
            folder = scene_directory(output/'eval_best', index, evaluation)
            links = []
            for label, filename in [('轨迹', f'trace-{index:04d}.npz'),
                                    ('视频', f'episode-{index:04d}.mp4'),
                                    ('距离图', f'distance-{index:04d}.png')]:
                target = folder/filename
                if target.exists():
                    links.append(f'[{label}]({target.relative_to(output).as_posix()})')
            lines.append(f'| {row["scene_id"]} | {"是" if row["success"] else "否"} | '
                         f'{row["steps"]} | {row["raw_return"]:.3f} | {" · ".join(links)} |')
        lines += ['']
    lines += ['## 阶段状态', '']
    lines += [f'- {name}: {stage["status"]}' for name, stage in record['stages'].items()]
    (output/'summary.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    add_options(parser)
    args = parser.parse_args(argv)
    if not (Path(args.run)/'config.json').is_file():
        parser.error('--run must identify an existing training directory containing config.json')
    torch.set_num_threads(1)
    workflow = Workflow('postprocess.json')
    workflow.attach(args.run)
    error = None
    try:
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
