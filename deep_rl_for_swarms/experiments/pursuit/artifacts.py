"""Evaluation layout and identities shared by CLI tools and the pipeline."""
import hashlib
import inspect
import json
import platform
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITE = PROJECT_ROOT / 'configs/pursuit/evaluation20.json'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_signature():
    from .common import evaluate
    package = PROJECT_ROOT / 'deep_rl_for_swarms'
    paths = sorted((package / 'ma_envs').rglob('*.py'))
    paths += [package / 'env_adapters/pursuit.py']
    identity = dict(
        sources={str(p.relative_to(package)): file_hash(p) for p in paths},
        evaluator=inspect.getsource(evaluate),
        python=platform.python_version(),
        dependencies={name: version(name) for name in
                      ('numpy', 'scipy', 'gym', 'shapely', 'fast-histogram')})
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def scene_directory(root, index, result):
    root = Path(root)
    return root / f'scene-{index:04d}' if result.get('layout') == 'scenes' else root


def trace_path(root, index, result):
    return scene_directory(root, index, result) / f'trace-{index:04d}.npz'
