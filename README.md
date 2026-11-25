# Deep RL for Swarm Systems
This repository contains the codebase for the paper "Deep Reinforcement Learning for Swarm Systems". The code for TRPO is based on the OpenAI Baselines and the environment structure is inspired by OpenAI's multi-agent particle environments.

https://github.com/openai/baselines/

https://github.com/openai/multiagent-particle-envs/

## 项目结构与用途

本项目用于多智能体连续控制实验，包含智能体聚集（Rendezvous）和追逃（Pursuit-Evasion）环境。当前 TRPO 入口默认运行聚集任务。MAPPO 入口运行追逃任务，集成固定 20 场景监控、周期模型保存和训练后的 200 场景批量筛选。

```text
deep_rl_for_swarms/
  run_multiagent_trpo.py           # TRPO 训练入口，默认 Rendezvous
  run_multiagent_mappo.py          # MAPPO 训练入口，默认 Pursuit-Evasion
  mappo_tools.py                   # 续训、单模型测试和 test-all 批量筛选
  mappo_eval_pool.py               # CPU 多进程场景评测，进程独享环境与策略会话
  rl_algo/
    trpo_mpi/                     # 支持 MPI 多进程的 TRPO 算法
    mappo.py                      # MAPPO 训练、集中式 critic 和模型筛选
  policies/                       # MLP、mean embedding 等策略网络
  ma_envs/
    base.py                       # 共享的世界状态与运动学实现
    envs/point_envs/              # 聚集、单 prey 追逃和多 prey 追逃环境
    agents/point_agents/          # 智能体观测、动作及 prey 的逃逸策略
  common/                         # 日志、概率分布、模型读写等工具
    mappo_evaluation.py           # 可复用场景、无渲染评测、指标与排名
    mappo_test.py                 # 可选的五场景轨迹导出
    mappo_visualization.py        # MP4 动画及距离曲线导出
test_scenarios/                   # 可复用的验证/演示场景及其初始状态
runs/                             # 每次训练的配置、日志、模型和测试结果
setup.py                          # Python 包与依赖配置
.venv/                            # 本地 Python 运行环境
```

MAPPO 经 `rl_algo/mappo.py` 完成训练和监控，再由 `mappo_tools.test_all_checkpoints` 复评全部保留模型。场景跨模型复用，每次训练的输出保存在 `runs/<时间戳>_<本次新增步数>steps/` 中。

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
.\.venv\python.exe -m deep_rl_for_swarms.run_multiagent_mappo
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
`EvalMeanReturn`, `EvalStdReturn`, `EvalSuccessRate`, `EvalMeanMinimumDistance`
and `EvalMeanFinalDistance` are logged on evaluation rows in `progress.csv`,
with `nan` on other rows. `BestEvalReturn` tracks the best monitor return.
These scores measure performance on the fixed evaluation scenarios.

Every `--checkpoint-interval` new environment steps (default 50,000), keep a
checkpoint at the next update boundary. Always save the final step, including
a partial last interval; a coincident periodic/final checkpoint is saved once.
Monitor improvements and resumed baselines are also retained. Old models are
never deleted. Saving and monitoring have independent intervals.
Each model has its own `<cumulative-training-step>/` directory with the `model`
prefix. `checkpoints.json` lists all new saves in `checkpoints`; `best` remains
the highest 20-scenario monitor return, `final` is the last step, and `selected`
points to the leader of the latest completed post-training validation ranking.

A TensorFlow 1 checkpoint consists of `.data-00000-of-00001` (variable values),
`.index` (variable index), and `.meta` (graph) files sharing the same prefix.
The `checkpoint` text file tracks the latest saved prefix. Pass the prefix
returned by `tf.train.latest_checkpoint(step_dir)` to `Saver.restore`.

### Post-training validation and optional animation

By default, training invokes `test-all`: evaluate every physically saved model
on the same 200 scenarios, each for at most 1024 steps, using policy mean actions.
Use `--no-test` to skip this step without changing monitoring or checkpoint saving.
Default validation seeds 30,000–30,199 are separate from monitor seeds
10,000–10,019 and the legacy five demonstration seeds 20,000–20,004.
Use `--test-episodes`, `--test-seed`, or `--test-scenarios` to choose another
shared suite. A validation seed range overlapping monitoring is rejected.

Post-training validation uses up to four CPU processes by default (capped by
logical CPU count). `--test-workers N` controls this for training and `resume`;
standalone `test` and `test-all` use `--workers N`. Set N to 1 for serial execution.
The pool splits each model's scenarios into small chunks, so both single-model
and multi-model validation can use all workers. Each process owns an independent
environment and single-threaded TensorFlow CPU session; GPU memory is not used.
Original scenario order is restored before computing aggregate metrics, keeping
the same seeds, capture rules and ranking as serial evaluation. Fixed 20-scenario
monitoring during training still runs serially. Small suites may take longer in
parallel because Windows must start fresh Python/TensorFlow processes.

The scenario seeds, initial pursuer positions and headings, and evader positions
are saved before training to `test_scenarios/pursuit_<settings-hash>_seed30000_200episodes.json`.
Future runs with the same settings reuse those states. Use `--test-scenarios`
to choose another JSON path and `--test-seed` to choose a different fixed set.
The tested scenarios are also copied into the run's test results.

Default validation exports only a ranked CSV plus metadata/scenario JSON files,
without per-scenario directories, plots, trajectories or video. This is a
checkpoint-selection validation suite, not a final untouched test set.
Optional five-scenario MP4 export requires `test --artifacts`, system FFmpeg
or the optional `imageio-ffmpeg` package.
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
  <each-retained-step>/model.data-00000-of-00001, model.index, model.meta, checkpoint
  test/validation_200_seed30000/
    scenarios.json
    metadata.json
    summary.csv
```

`summary.csv` has one row per model, sorted by **success rate descending,
mean minimum distance ascending, mean final distance ascending**. Exact ties
share a rank, with stable step/path ordering for display. Distances use torus
geometry. Capture means distance strictly less than 1 after an environment step;
capture at step 1024 counts as success. Both distance means include successes
and failures. Minimum distance includes the initial state; final distance is
measured at capture or timeout. It is not the time-average distance.

Columns include `rank`, `step`, `aliases`, `checkpoint`, `episodes`, `success_count`,
`success_rate`, `mean_minimum_distance`, `mean_final_distance`,
`success_rate_ci95_low/high` (Wilson), `mean_episode_return`, `std_episode_return`,
`mean_episode_steps`, `mean_capture_steps_success` (blank if no captures),
and `elapsed_seconds`. Only the three stated metrics participate in ranking.
The interval describes scenario uncertainty for a fixed policy, not variation
across independently trained policies. CSV is updated after each model;
`metadata.json` marks the batch as running, complete, failed, or interrupted.
It also records the worker count and total `batch_wall_seconds`. Under parallel
evaluation, each model's `elapsed_seconds` measures first chunk start to last
chunk completion, excluding initial process startup/queue wait. These model times
can overlap; sum them only for serial evaluation. Reused rows retain their old times.

`aliases` joins all matching model labels with semicolons, for example
`best;selected` or `final`; unlabeled periodic models have an empty value.
The labels are an export snapshot, also recorded in `metadata.json` as
`alias_snapshot`. Batch validation assigns `selected` to its own completed
ranking's leader. A single-model test displays existing labels without selecting
a new winner. Later validations update the run's alias but do not rewrite old CSVs.

For a short smoke run:

```sh
.\.venv\python.exe -m deep_rl_for_swarms.run_multiagent_mappo --num-timesteps 16 --rollout-steps 8 --minibatch-steps 4 --epochs 1 --eval-interval 8 --eval-episodes 1 --no-test --log-dir runs
```

### 续训与独立测试

辅助入口统一放在 `deep_rl_for_swarms/mappo_tools.py`，提供 `resume`、`test`、`test-all`。
在仓库根目录运行：

```powershell
# 从 40 万步模型追加 60 万环境步，最终累计 100 万步
.\.venv\python.exe -m deep_rl_for_swarms.mappo_tools resume runs\20261123_105105_337779_400000steps\400000 --additional-timesteps 600000 --note "从40万步模型继续训练60万步"

# 单独测试 40 万步模型
.\.venv\python.exe -m deep_rl_for_swarms.mappo_tools test runs\20261123_105105_337779_400000steps\400000

# 一键复评某个 run 保留的全部模型，默认 200 场景
.\.venv\python.exe -m deep_rl_for_swarms.mappo_tools test-all runs\20261123_122746_387474_600000steps

# 指定 4 个评测进程；单模型 test 也支持 --workers
.\.venv\python.exe -m deep_rl_for_swarms.mappo_tools test-all runs\20261125_090408_702662_499648steps --workers 4

# 中断旧串行评测后，复用已完成的模型，剩余模型并行评测到新目录
.\.venv\python.exe -m deep_rl_for_swarms.mappo_tools test-all runs\20261125_090408_702662_499648steps --workers 4 --reuse-report runs\20261125_090408_702662_499648steps\test\validation_200_seed30000 --output-dir runs\20261125_090408_702662_499648steps\test\validation_200_seed30000_parallel

# 也可以传运行目录，用 best/final/selected 选择模型；默认 final
.\.venv\python.exe -m deep_rl_for_swarms.mappo_tools test runs\20261123_105105_337779_400000steps --model best

# 原来的五场景动画改为显式启用(200场景目前就不生成场景动画了, 后续可以加上一个, 指定场景生成动画的)
.\.venv\python.exe -m deep_rl_for_swarms.mappo_tools test runs\20261123_105105_337779_400000steps\400000 --artifacts
```

`resume` 默认继承来源 `config.json` 的环境、训练参数和种子，
恢复 actor、集中式 critic 和 Adam 的动量等状态，所有原有可训练参数继续更新。
如需改变超参数，可以显式传入 `--learning-rate`、`--entropy-coef`、`--gamma`、
`--lam`、`--clip`、`--epochs` 等；使用 `--no-test` 可跳过续训后的自动批量评测。
旧配置缺少 `checkpoint_interval` 时补为 50000；旧五场景测试配置升级为
200 场景/种子 30000。新协议的场景数和种子继续继承，可显式覆盖。
`--help` 列出全部可覆盖选项。当前工具要求模型结构和环境运行参数与现有实现兼容。

续训创建独立的 `<时间戳>_<追加步数>steps/`，默认放在来源 run 的同级目录，
可用 `--log-dir` 指定新的父目录。旧训练日志和模型不被覆盖。
新配置中的 `num_timesteps` 表示本次追加步数；`continuation` 记录来源 run/config/模型、
来源累计步数、追加步数、目标累计步数、备注、参数覆盖及多次续训的 `history`。
例如从 400000 步追加 600000 步，`source_step=400000`、`additional_timesteps=600000`、
`target_total_timesteps=1000000`。日志 `TimestepsSoFar` 和模型文件夹使用累计步数，
`TimestepsThisRun` 使用本次新增步数。评估间隔按本次新增步数计数。

续训会重新评估起点模型，在新 run 中保留本地基线。后续刷新 best 时基线也不删除。
不复制其他未选中的来源模型。环境、NumPy 和 TensorFlow 随机流重新开始；
旧检查点未保存采样器与回合状态，因此这不是逐位复现原训练轨迹的断点恢复。

独立 `test` 支持步数文件夹、`model` 前缀、`.index`/`.meta` 文件或 run 文件夹，
默认输出为来源 run 的 `test/<模型累计步数>/validation_200_seed30000/`，
内容与批量评测相同，只有一个模型行；批量输出为 `test/validation_200_seed30000/`。
这两个新路径可与已有五场景结果共存。已有报告会报错保护；可先移动旧目录，
或用 `--output-dir` 指定新目录。`test-all` 也支持该选项。
`--test-episodes`、`--test-scenarios`、`--test-seed` 控制验证集。
`--workers` 控制独立评测的进程数；新训练和续训用 `--test-workers`，默认最多 4 个。
续训继承新配置中的评测进程数，可显式覆盖；旧配置没有此字段时使用默认值。
修改代码不会让正在运行的串行评测切换到并行。需要切换时，可先在旧终端按
Ctrl+C，然后用 `--reuse-report <旧报告目录>`、`--output-dir <新报告目录>` 继续。
程序逐项核对场景初始状态和评测规则，复用旧 `metadata.json` 中完整评测完的模型；
当前只完成一部分场景的模型仍需重新评测。旧 CSV/JSON 保留，所有模型完成后才
更新 `selected`。旧版串行报告也可复用；复用要求检查点权重仍是当时的原文件。
建议先用 4 个进程；可根据 CPU/内存占用尝试 8 个，同时运行多个批次时要合计进程数。
`--artifacts` 则保留原来的五场景 `test/<step>/` 目录格式，才使用动画参数。
批量完成后 `checkpoints.json` 增加 `selected` 别名；旧 `best`/`final` 含义不变。
旧运行被移动后，工具会尝试使用运行目录内已复制的测试场景。
Windows 下辅助命令会临时补充旧 SciPy wheel 的 DLL 搜索目录，不修改依赖安装。

检查续训恢复、Adam 状态、累计步数、配置继承和历史链：

```powershell
.\.venv\python.exe -m unittest discover -s tests -p "test_mappo*.py"
```
