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

## 一键训练、固定评估与可视化

在项目根目录激活 `swarm-rl` 后运行：

```bash
MPLBACKEND=Agg python -m deep_rl_for_swarms.experiments.pursuit.pipeline \
  --config configs/pursuit/reference.json --output runs --seed 0
```

默认依次执行：**从头训练 → 评估本次 best.pt → 创建或复用共享随机基线 → 绘制图表并导出全部20个场景的视频**。最后打印 `summary.md` 的位置。不会自动评估 `untrained.pt`。

默认训练200000环境步；`--steps 2048` 可做短检查。只有训练入口的 `--output` 表示父目录，会自动新建 `mappo-YYYYMMDD-HHMMSS-seedN`。目录时间戳来自运行机器本地开始时间，精确到秒；同秒同种子的同名目录存在时拒绝覆盖。`--seed` 是训练种子，不改变固定评估初态或默认随机对照动作。

| 参数 | 行为 |
| --- | --- |
| 不加跳过参数 | 训练、评估、随机对照、图表及全部视频 |
| `--skip-visualization` | 训练与评估，保存JSON和NPZ；不生成图片或视频 |
| `--skip-evaluation` | 训练并绘制学习曲线；不进行固定集评估、随机对照和场景渲染 |
| 两个跳过参数同时使用 | 只训练 |
| `--skip-videos` | 保留评估、轨迹和静态图表，跳过视频编码 |

`--device cpu` 默认已验收；`--device mps` 仍需单独检查，不承诺加速或逐位复现。`--stride 4 --fps 15` 控制视频采样和播放；视频编码增加耗时，日常调参可使用 `--skip-videos`。支持 `--suite`、`--random-output`、`--action-seed` 指定其他评估协议；必须保证各对照使用同一组初态。

## 固定评估集与共享随机对照

`configs/pursuit/evaluation20.json` 原样固定了旧 `runs/test20.json` 的20个初始场景（生成种子200000）。每次读取具体位置和朝向，不重新抽样。文件名是新的，JSON中的历史 `split: test` 与场景ID保留，以保持内容哈希一致。它现在是反复比较、调参用的**固定评估集**；正式泛化成绩应另用未参与调参的冻结测试集。

训练内部验证初态仍由种子100000生成，`best.pt` 仅按验证成功率、失败按1024计入的平均用时、原始回报依次选择，固定评估结果不参与自动选模型。评估使用确定性Actor，不更新参数。

共享随机基线默认保存到项目根目录的 `runs/eval_random/`，动作是 `RandomState(200000).uniform(-1, 1, (num_agents, 2))`。第一次需要时自动评估一次，之后校验场景集、环境配置/实现及依赖版本、动作种子、策略身份和轨迹哈希后复用。仅修改算法超参数或训练种子不要求重跑随机基线。改变环境、场景集、动作种子或相关环境依赖时，指定新的 `--random-output`；已有不兼容或损坏的结果会报错，不会静默复用或覆盖。

同一基线由目录锁保护，并发使用时会明确报忙，可稍后补做后处理；强制杀进程可能遗留 `.eval_random.lock`，应确认没有任务使用后再处理。随机评估数据完整生成后才发布；后续缺视频只补渲染，改变stride/fps也只重新渲染。一次固定动作序列是可复现参考，不是随机策略期望性能的精确估计。

## 结果保存在哪里，先看什么

```text
runs/
├── eval_random/
│   ├── results.json
│   ├── scene-0000/
│   │   ├── trace-0000.npz
│   │   ├── episode-0000.mp4
│   │   ├── render.json
│   │   ├── distance-0000.png
│   │   └── trajectory-0000.png
│   └── ...共20个场景目录
└── mappo-YYYYMMDD-HHMMSS-seedN/
    ├── config.json / manifest.json / validation.json
    ├── metrics.jsonl / history.json / timing.json
    ├── untrained.pt / best.pt / latest.pt
    ├── eval_best/
    │   ├── results.json
    │   └── scene-XXXX/       # 与随机基线相同的逐场景结构
    ├── figures/
    │   ├── learning.png
    │   ├── evaluation.png
    │   ├── table.json
    │   └── plot-manifest.json
    ├── summary.md
    └── pipeline.json
```

跳过步骤时对应产物不生成。建议先打开 **summary.md**，再看 `figures/learning.png`，最后选择场景目录里的视频和距离曲线。左侧学习曲线为验证捕获率，右侧为验证原始回报；回报越高越好，但回报改善不等于捕获成功。`evaluation.png` 比较固定集捕获率，详细回报见summary或table.json。

`results.json` 包含总体summary、逐场景episodes、场景哈希、配置、模型哈希、策略类型、环境实现标识及代码来源。报告成功率、成功条件平均捕获步数（无成功为null/N/A）、失败按1024计入的平均用时和一份共享奖励累计的原始回报。`done` 不等于成功。

轨迹在周期边界断线；视频标明场景、步数、距离和捕获/超时，不挑选最好案例。严格捕获阈值 `<1`，最多1024环境步。参考配置是10追捕者、1脚本prey、100×100周期场地、unicycle动力学，通信半径 `200*sqrt(2)`、观测半径为其一半；不能将其称为小范围感知实验。

## 只训练，或给已有模型补做后处理

原训练入口继续可用，它只训练，不自动评估/渲染：

```bash
MPLBACKEND=Agg python -m deep_rl_for_swarms.experiments.pursuit.train \
  --config configs/pursuit/reference.json --output runs --seed 0
```

已有训练目录可以直接补做，**不会重新训练**：

```bash
MPLBACKEND=Agg python -m deep_rl_for_swarms.experiments.pursuit.postprocess \
  --run runs/mappo-20261119-084855-seed0
```

postprocess支持与pipeline相同的评估/可视化开关。若原先跳过视频，之后不带 `--skip-videos` 运行就会补视频；重复运行会验证并复用已有轨迹和匹配的渲染产物，不重新评估。已有eval_best与当前best不匹配时拒绝覆盖，应保留旧结果并明确处理后再运行。

`timing.json` 仍只记录训练起止时间、持续秒数和状态。每个checkpoint保存运行计时，并通过 `checkpoint_saved_at`、`checkpoint_elapsed_seconds` 区分模型保存时刻与训练结束时刻；不改变模型参数和步数。日志elapsed记录在每次更新完成、该轮验证和保存之前，包含之前轮次的耗时。

`pipeline.json` 分别记录训练、模型评估、随机基线和可视化状态及耗时；补做后处理另记 `postprocess.json`，保留原流水线记录。后处理失败不会删除已完成模型或评估，summary会显示失败阶段。正常异常和Ctrl+C会记录状态；强制终止、断电无法保证。当前仍不支持训练断点续训，postprocess的补做只针对评估和渲染。

## 单独调用评估、渲染与指标复算

这些命令是可选的分步入口。评估输出目录必须不存在，`--output` 是精确位置，不自动添加时间戳。只加载可信的本地checkpoint。

```bash
python -m deep_rl_for_swarms.experiments.pursuit.evaluate \
  --suite configs/pursuit/evaluation20.json \
  --checkpoint runs/mappo-20261119-084855-seed0/best.pt \
  --output runs/manual-evaluation --trajectories
python -m deep_rl_for_swarms.experiments.pursuit.render \
  --evaluation runs/manual-evaluation --index 0 --video
python -m deep_rl_for_swarms.experiments.pursuit.audit \
  --suite configs/pursuit/evaluation20.json --evaluations runs/manual-evaluation \
  --output runs/manual-audit.json
```

新评估默认按 `scene-XXXX/` 分目录；`--layout flat` 可导出旧布局。render、audit、compare同时兼容没有layout字段的历史平铺结果和新布局。`--index` 是明确的场景编号。并排比较随机与模型的同一初态：

```bash
python -m deep_rl_for_swarms.experiments.pursuit.compare \
  --before runs/eval_random \
  --after runs/mappo-20261119-084855-seed0/eval_best \
  --index 0 --output runs/comparison.mp4
```

较短一侧保持终帧；编码使用imageio-ffmpeg自带FFmpeg。多个训练目录可传给plot的 `--runs`，按共同预算绘制均值和训练种子间样本标准差；单种子不画虚假方差。

## 证据与限制

[适配记录](ADAPTATION.md)、[展示计划](PRESENTATION.md)、[验收清单](VERIFICATION.md)、[历史验收结果](VERIFICATION_RESULTS.md)。2026-11-18验收中的旧目录、未训练对照和分步命令是历史记录，不代表当前默认流程。新流水线检查记录见 [PIPELINE_VERIFICATION.md](PIPELINE_VERIFICATION.md)。运行输出在 `runs/`，默认不加入Git；固定评估初态作为配置纳入版本管理。

原构造器中失效的默认值和未使用的动力学分支未重写，新入口只接受此次支持的显式配置。原prey对精确中线位置及退化几何有已知风险，发生异常会报错，不静默重抽场景。新流程不改变环境规则、奖励或MAPPO更新。
