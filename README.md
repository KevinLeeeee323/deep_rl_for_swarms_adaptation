# Deep RL for Swarm Systems
This repository contains the codebase for the paper "Deep Reinforcement Learning for Swarm Systems". The code for TRPO is based on the OpenAI Baselines and the environment structure is inspired by OpenAI's multi-agent particle environments.

https://github.com/openai/baselines/

https://github.com/openai/multiagent-particle-envs/

## 项目结构与用途

本项目用于多智能体连续控制实验，包含智能体聚集（Rendezvous）和追逃（Pursuit-Evasion）环境。当前 TRPO 入口默认运行聚集任务，可通过匹配环境和观测策略接入追逃任务。MAPPO 入口运行追逃任务，集成定期评估、best/final 模型保存和训练后的固定场景测试。

```text
deep_rl_for_swarms/
  run_multiagent_trpo.py           # TRPO 训练入口，默认 Rendezvous
  run_multiagent_mappo.py          # MAPPO 训练入口，默认 Pursuit-Evasion
  rl_algo/
    trpo_mpi/                     # 支持 MPI 多进程的 TRPO 算法
    mappo.py                      # MAPPO 训练、集中式 critic 和模型筛选
  policies/                       # MLP、mean embedding 等策略网络
  ma_envs/
    base.py                       # 共享的世界状态与运动学实现
    envs/point_envs/              # 聚集、单 prey 追逃和多 prey 追逃环境
    agents/point_agents/          # 智能体观测、动作及 prey 的逃逸策略
  common/                         # 日志、概率分布、模型读写等工具
    mappo_test.py                 # 固定测试场景、成功率和距离统计
    mappo_visualization.py        # MP4 动画及距离曲线导出
test_scenarios/                   # 可复用的 5 个固定测试场景及其初始状态
runs/                             # 每次训练的配置、日志、模型和测试结果
setup.py                          # Python 包与依赖配置
.venv/                            # 本地 Python 运行环境
```

MAPPO 从训练入口创建追逃环境，经 `rl_algo/mappo.py` 完成训练和评估，再由 `common/mappo_test.py` 加载选定模型进行测试。`test_scenarios/` 中的场景跨训练复用，每次训练的输出保存在 `runs/<时间戳>_<总步数>steps/` 中。

## Installation
```
(optional) virtualenv -p python3 ~/virtual_envs/drl_for_swarms
(optional) source ~/virtual_envs/drl_for_swarms/bin/activate

git clone git@github.com:LCAS/deep_rl_for_swarms.git
cd deep_rl_for_swarms
pip install -e .
```

## Run
single core
```
cd deep_rl_for_swarms
python run_multiagent_trpo.py
```

multi core
```
cd deep_rl_for_swarms
mpirun -np 4 python run_multiagent_trpo.py
```

## MAPPO

From the repository root, use the existing TensorFlow 1 environment.

```sh
python -m deep_rl_for_swarms.run_multiagent_mappo
```

MAPPO trains the continuous-control `PursuitEvasionEnv`, using its example
configuration of 10 pursuers, one scripted evader, `sum_obs_no_ori` observations,
`unicycle` dynamics, a 100-unit periodic world, and communication radius
`200 * sqrt(2)`. Pursuers share the mean-embedding actor. The actor combines
embedded observations of other pursuers with the evader's relative distance
and bearing. A centralized critic receives all pursuer poses and velocities,
the evader position, and episode progress. The actor is updated
with clipped PPO and generalized advantage estimation.

MAPPO uses a single process and a single-threaded TensorFlow session.
`--num-timesteps` counts environment steps, with all agents acting at each
step. Each update collects up to `--rollout-steps` steps (default 2048).
Each of the `--epochs` passes (default 5) shuffles those steps into minibatches
of up to `--minibatch-steps` steps (default 64), keeping all agents from a
step together.

Training writes cumulative wall-clock time as `ElapsedSeconds` in standard output
and `progress.csv`. Each invocation creates a directory named
`<timestamp>_<num-timesteps>steps` inside `--log-dir` (default `runs/`).
For example, a 16-step run writes to `runs/20261123_173000_123456_16steps/`.
Every `--eval-interval` training steps (default 50,000), MAPPO evaluates the
updated policy in a separate environment. Evaluation runs at the first update
boundary at or after each interval, and always at the final training step.
With 2048-step rollouts, the first evaluation runs at step 51,200.
Each evaluation runs `--eval-episodes` complete episodes (default 20), using
deterministic policy mean actions. The fixed episode seeds start at
`--eval-seed` (default 10,000), so each evaluation uses the same 20 initial
scenarios. Evaluation preserves the training NumPy random state and its steps
are excluded from the training budget. Elapsed time includes evaluation.

The selection score is the mean episode return, computed by summing the mean
agent reward over each full episode. Higher scores select a new best model.
`EvalMeanReturn` and `EvalStdReturn` are logged on evaluation rows in
`progress.csv`, with `nan` on other rows. `BestEvalReturn` tracks the best score.
These scores measure performance on the fixed evaluation scenarios.

Checkpoints are saved only for a new best model and the final model. Each saved
model has its own `<training-step>/` directory containing the `model` prefix.
The run's `checkpoints.json` identifies the best and final prefixes relative
to the run directory, together with their evaluation scores. Superseded best
checkpoints are removed after their replacements are saved. If the final model
also achieves the best score, both entries point to the same step directory.

A TensorFlow 1 checkpoint consists of `.data-00000-of-00001` (variable values),
`.index` (variable index), and `.meta` (graph) files sharing the same prefix.
The `checkpoint` text file tracks the latest saved prefix. Pass the prefix
returned by `tf.train.latest_checkpoint(step_dir)` to `Saver.restore`.

### Post-training tests and animation

After training, the best checkpoint is restored and tested on five fixed
scenarios, each for one episode of at most 1024 steps. Use `--test-model final`
to test the final checkpoint or `--no-test` to skip testing and animation.
Tests use deterministic policy mean actions. Default test seeds 20,000–20,004
are separate from evaluation seeds 10,000–10,019. A test seed range overlapping
the evaluation range is rejected.

The scenario seeds, initial pursuer positions and headings, and evader positions
are saved before training to `test_scenarios/pursuit_<settings-hash>_seed20000.json`.
Future runs with the same settings reuse those states. Use `--test-scenarios`
to choose another JSON path and `--test-seed` to choose a different fixed set.
The tested scenarios are also copied into the run's test results.

MP4 export requires system FFmpeg or the optional `imageio-ffmpeg` package.
The latter includes the FFmpeg executable. Install it in the training environment:

```sh
python -m pip install imageio-ffmpeg==0.4.9
```

All artifacts are inside the timestamped training directory:

```text
<timestamp>_<num-timesteps>steps/
  config.json
  progress.csv
  checkpoints.json
  <best-step>/model.data-00000-of-00001, model.index, model.meta, checkpoint
  <final-step>/model.data-00000-of-00001, model.index, model.meta, checkpoint
  test/best/
    scenarios.json
    summary.json
    summary.csv
    scenario_01/
      animation.mp4
      distances.png
      distances.csv
    ... scenario_02 through scenario_05
```

`config.json` records environment settings, algorithm hyperparameters, network
settings, training/evaluation/test seeds, animation settings, and runtime versions.
Test summaries identify the restored checkpoint and record each scenario's return,
length, capture result and step, and initial/minimum/final nearest distance.
`summary.json` includes the overall success rate. Capture follows the environment's
strict nearest-distance threshold of 1 unit. Distances respect the periodic world
geometry. CSV files and plots contain the initial state at step 0 and every
interaction step. Videos show every fourth step plus the initial and final states,
at 20 fps. Adjust these with `--animation-stride` and `--animation-fps`.

For a short smoke run:

```sh
python -m deep_rl_for_swarms.run_multiagent_mappo --num-timesteps 16 --rollout-steps 8 --minibatch-steps 4 --epochs 1 --eval-interval 8 --eval-episodes 1 --no-test --log-dir runs
```
