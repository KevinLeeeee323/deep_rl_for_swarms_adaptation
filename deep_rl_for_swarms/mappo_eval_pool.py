"""Spawned CPU evaluators with isolated environments, graphs and random streams.

Keep this module outside common: common.__init__ imports SciPy before Windows
workers can configure the legacy wheel's DLL search directory.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import multiprocessing
import os
import time


def evaluation_workers(workers=None):
    if workers is None:
        return min(8, os.cpu_count() or 1)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError('Evaluation workers must be a positive integer.')
    if os.name == 'nt' and workers > 61:
        raise ValueError('Windows supports at most 61 evaluation workers.')
    return workers


@contextmanager
def _child_environment():
    # Set before spawning: NumPy may be imported before a worker's initializer.
    settings = {name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                      'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    settings.update(CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2')
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class PolicyEvaluator:
    """One reusable policy session and one environment, owned by one process."""

    def __init__(self, config):
        from deep_rl_for_swarms.mappo_tools import _scipy_dll_directory, reward_settings
        self._dll = _scipy_dll_directory()
        self._dll.__enter__()
        self.env = None
        self.session = None
        try:
            import tensorflow as tf
            from deep_rl_for_swarms.ma_envs.envs.point_envs.pursuit_evasion import PursuitEvasionEnv
            from deep_rl_for_swarms.policies.mlp_mean_embedding_policy import MlpPolicy

            self.env = PursuitEvasionEnv(capture_bonus=reward_settings(config)['capture_bonus'],
                                        **config['environment_settings'])
            self.graph = tf.Graph()
            session_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1,
                                            device_count={'GPU': 0})
            self.session = tf.Session(graph=self.graph, config=session_config)
            with self.graph.as_default(), self.session.as_default():
                tf.set_random_seed(config['seed'])
                self.obs = tf.placeholder(tf.float32, (None,) + self.env.observation_space.shape,
                                          name='observations')
                policy = MlpPolicy('mappo_policy', self.env.observation_space, self.env.action_space,
                                   hid_size=[64], feat_size=[64], observation_input=self.obs)
                self.action_mode = policy.pd.mode()
                self.saver = tf.train.Saver(var_list=policy.get_variables())
            self.checkpoint = None
        except BaseException:
            self.close()
            raise

    def evaluate(self, checkpoint, suite, progress=None):
        from deep_rl_for_swarms.common.mappo_evaluation import evaluate_scenarios
        with self.graph.as_default(), self.session.as_default():
            if self.checkpoint != checkpoint:
                self.saver.restore(self.session, checkpoint)
                self.checkpoint = checkpoint
            return evaluate_scenarios(
                self.env, lambda observations: self.session.run(self.action_mode, {self.obs: observations}),
                suite, progress)

    def close(self):
        if self.session is not None:
            self.session.close()
        if self.env is not None:
            self.env.close()
        self._dll.__exit__(None, None, None)


_worker = None


def _initialize_worker(config):
    global _worker
    import atexit
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    from deep_rl_for_swarms.mappo_tools import _scipy_dll_directory
    with _scipy_dll_directory():
        import tensorflow as tf
        tf.logging.set_verbosity(tf.logging.ERROR)
    _worker = PolicyEvaluator(config)
    atexit.register(_worker.close)


def _evaluate_chunk(checkpoint, suite):
    start = time.monotonic()
    reports = _worker.evaluate(checkpoint, suite)
    return reports, start, time.monotonic()


def evaluate_candidates(sources, suite, workers):
    """Yield complete models; aggregate reports in original scenario order.

    Smaller scenario chunks balance early captures against 1024-step timeouts.
    Only the parent writes reports. No nested pools or shared TensorFlow objects.
    """
    if not sources:
        return
    workers = min(evaluation_workers(workers), len(sources) * len(suite['scenarios']))
    if workers == 1:
        evaluator = PolicyEvaluator(sources[0]['config'])
        try:
            for source in sources:
                start = time.perf_counter()

                def progress(index, count):
                    if index % 10 == 0 or index == count:
                        print('Validation step {}: {}/{} scenarios'.format(
                            source['step'], index, count), flush=True)

                reports = evaluator.evaluate(source['checkpoint'], suite, progress)
                yield source, reports, time.perf_counter() - start
        finally:
            evaluator.close()
        return

    # At least two chunks per worker per model allow load balancing.
    chunk_size = max(1, (len(suite['scenarios']) + 2 * workers - 1) // (2 * workers))
    buffers = [[] for _ in sources]
    starts = [None for _ in sources]
    ends = [None for _ in sources]
    counts = [0 for _ in sources]
    futures = {}
    with _child_environment(), ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize_worker, initargs=(sources[0]['config'],)) as pool:
        try:
            for model_index, source in enumerate(sources):
                for offset in range(0, len(suite['scenarios']), chunk_size):
                    chunk = dict(suite, scenarios=suite['scenarios'][offset:offset + chunk_size])
                    future = pool.submit(_evaluate_chunk, source['checkpoint'], chunk)
                    futures[future] = (model_index, offset)
            for future in as_completed(futures):
                model_index, offset = futures[future]
                reports, start, end = future.result()
                buffers[model_index].append((offset, reports))
                starts[model_index] = start if starts[model_index] is None else min(starts[model_index], start)
                ends[model_index] = end if ends[model_index] is None else max(ends[model_index], end)
                counts[model_index] += len(reports)
                source = sources[model_index]
                print('Validation step {}: {}/{} scenarios ({} workers)'.format(
                    source['step'], counts[model_index], len(suite['scenarios']), workers), flush=True)
                if counts[model_index] == len(suite['scenarios']):
                    ordered = [report for _, chunk in sorted(buffers[model_index]) for report in chunk]
                    yield source, ordered, ends[model_index] - starts[model_index]
                    buffers[model_index] = []
        except BaseException:
            for future in futures:
                future.cancel()
            raise
