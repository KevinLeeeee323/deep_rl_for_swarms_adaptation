# 连续多追一：macOS ARM / PyTorch MAPPO

本分支仅适配课程选题三第二问 `pursuit_evasion.py`。原始粒子环境、prey 脚本、奖励和结束判定源码保持原样；通过外层适配加入新训练与评估。参考提交为 `2e51997`，工作分支为 `codex/adapt`。

## 安装

已验证 Python 3.10.22、arm64。首次使用可创建环境；已有 `swarm-rl` 时直接激活：

```bash
conda activate swarm-rl
python -m pip install -r requirements-macos.lock.txt
python -m pip install --no-deps -e .
python -m pip check
```

重建环境：`conda env create -f environment-macos.yml`。`requirements-macos.txt` 是直接依赖，lock 文件包含验收时的传递依赖。旧依赖保存在 `requirements-legacy.txt`；旧 TF1/TRPO 入口作为参考保留，新环境不能直接运行它。

## 先验证，再训练

在项目根目录执行：

```bash
MPLBACKEND=Agg python -m pytest -q
MPLBACKEND=Agg python -m deep_rl_for_swarms.experiments.pursuit.verify
```

第二条命令从 Git 固定参考提交导出原环境到临时目录，用独立进程运行，再与适配层在同一现代依赖栈对照。它不证明历史依赖栈数值完全等价。

## 训练

```bash
MPLBACKEND=Agg python -m deep_rl_for_swarms.experiments.pursuit.train   --config configs/pursuit/reference.json --output runs --seed 0
```

默认 200000 环境步；可用 `--steps 2048` 做短检查。`--output` 现在指定父目录，程序自动新建 `mappo-YYYYMMDD-HHMMSS-seedN` 子目录，例如 `runs/mappo-20261119-090000-seed0/`。时间来自 `time.time()`，按运行机器本地时间格式化到秒；启动时输出实际目录。父目录可以存在，但同秒、同 seed 的同名运行目录不得已存在，避免覆盖。以后接入其他算法时沿用同一命名方式、替换算法前缀；当前入口仅实现 MAPPO。CPU 是已验证后端，`--device mps` 需单独检查，本分支不承诺逐位可复现或加速。

每个 rollout 记录一个共享团队回报与全部追捕者动作，Actor 共享，Critic 使用归一化位置/朝向全局状态。连续动作采用 tanh 高斯策略，保留 latent 和 Jacobian 修正后的 log probability。真捕获不 bootstrap，超时 bootstrap 且 GAE 不跨 reset。训练不修改原奖励。

每个运行子目录包括 `config.json`、`manifest.json`、固定 `validation.json`、`metrics.jsonl`、`history.json`、`timing.json`、`untrained.pt`、`best.pt` 和 `latest.pt`。时间戳只加在运行目录上，模型文件名保持固定。`best.pt` 从已训练 checkpoint 按验证成功率、失败按 1024 计入的用时、原始回报依次选择；测试集不参与选择。当前入口从头训练，尚无断点续训参数。

`timing.json`、manifest 和每个 checkpoint 的 `timing` 字段保存整次运行的起止时间（带本地时区偏移）、持续秒数和状态（`running` / `completed` / `interrupted` / `failed`）。checkpoint 另保留 `checkpoint_saved_at` 和 `checkpoint_elapsed_seconds`，区分模型产生时刻与整次运行结束时刻；结束回填只更新元数据，不替换模型参数和步数。正常退出、Python 异常及 Ctrl+C 会回填，强制终止或断电无法保证。总计时从运行目录命名开始，包含初始化、基线验证、采样、更新、周期验证和常规保存，不含最终计时元数据回填。日志 `elapsed` 在每次 `model.update()` 刚完成时记录累计耗时，因此不含当轮随后执行的验证和保存，但包含之前轮次的验证和保存；初始行记录基线验证完成时的实际累计耗时。

## 独立测试与随机对照

先生成一份固定测试集。训练内部验证 seed 为 100000，测试默认 seed 为 200000：

```bash
python -m deep_rl_for_swarms.experiments.pursuit.evaluate   --suite runs/test100.json --generate 100 --split test
python -m deep_rl_for_swarms.experiments.pursuit.evaluate   --suite runs/test100.json --checkpoint runs/mappo-20261119-090000-seed0/best.pt   --output runs/test-mappo0 --trajectories
python -m deep_rl_for_swarms.experiments.pursuit.evaluate   --suite runs/test100.json --random --output runs/test-random
```

将上述示例运行目录替换为训练时打印的实际目录。评估、绘图等入口的 `--output` 仍表示精确输出位置，只有训练入口自动新建运行子目录。同一测试集可评估其他种子 checkpoint 和 `untrained.pt`。最终测试完成后不根据其结果挑选模型。结果位于 `results.json`；有轨迹时另存 `trace-XXXX.npz`。每个结果附场景集哈希、配置、checkpoint 哈希和代码来源。只加载可信的本地模型文件。

原演示配置：10 个追捕者、1 个脚本 prey、100×100 周期场地、unicycle 动力学，通信半径 `200*sqrt(2)`、观测半径为其一半。范围很大，不能省略后称为局部观测实验。具体值见配置文件。

评价沿用原始共享距离奖励；严格捕获阈值 `<1`，最多 1024 环境步。报告成功率、成功条件下平均捕获步数（无成功为 null / N/A）、失败按 1024 计入的平均用时及一份共享奖励累计的原始回报。`done` 不等于成功。

## 可视化

```bash
python -m deep_rl_for_swarms.experiments.pursuit.render   --evaluation runs/test-mappo0 --index 0 --video
python -m deep_rl_for_swarms.experiments.pursuit.plot   --runs runs/mappo-20261119-090000-seed0 --evaluations runs/test-mappo0 runs/test-random   --output runs/figures
```

导出轨迹图、距离曲线、MP4 及元数据。轨迹在周期边界断线，动画标明步数、距离和捕获/超时；`--index` 是显式场景编号，不自动挑选最佳案例。多个训练目录传给 `--runs` 可绘制对齐预算的跨种子均值和标准差带；单种子不画虚假的方差。

## 证据与限制

[适配记录](ADAPTATION.md)、[展示计划](PRESENTATION.md)、[验收清单](VERIFICATION.md)、[实际验收结果](VERIFICATION_RESULTS.md)。运行输出在 `runs/`，默认不加入 Git。原构造器中失效的默认值和未使用的动力学分支未重写，新入口只接受此次支持的显式配置。原 prey 对精确中线位置及退化几何有已知风险，参考与新环境均保留其行为，发生异常会报错而不静默重抽场景。


复算指标（需要对应的 `--trajectories` 评估输出）：

```bash
python -m deep_rl_for_swarms.experiments.pursuit.audit --suite runs/test100.json --evaluations runs/test-mappo0 --output runs/metric-audit.json
```

同初态并排视频（左 before、右 after；较短的一侧保持终帧，避免隐藏提前结束）：

```bash
python -m deep_rl_for_swarms.experiments.pursuit.compare --before runs/test-untrained --after runs/test-mappo0 --index 0 --output runs/comparison.mp4
```

先用同一测试集和 `untrained.pt` 生成 `runs/test-untrained`，并在两次评估时均加 `--trajectories`。视频编码使用 `imageio-ffmpeg` 自带 FFmpeg，无需系统 FFmpeg 含 `libx264`。
