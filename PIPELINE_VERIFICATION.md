# 一键实验流程验收（2026-11-19）

分支：`codex/adapt`。CPU、Python 3.10.22 / PyTorch 2.5.1，使用已有swarm-rl环境。本文记录新增流水线的检查，2026-11-18历史迁移与效果结果仍保留于VERIFICATION_RESULTS.md。本次短训练只检验流程，不作为策略效果实验。

## 自动检查

```bash
MPLBACKEND=Agg python -m pytest -q
```

**34 passed**。保留原23项检查，新增11项（含参数化用例）：

- 固定evaluation20初态与历史20场景完全一致。
- 缓存身份、环境/动作种子/策略不匹配和损坏轨迹检测。
- 新分场景及历史平铺布局的渲染、指标复算和并排视频。
- 四种评估/可视化开关组合，流水线始终调用训练。
- 后处理失败保留checkpoint，补做复用轨迹且不再运行策略采样。
- 共享基线互斥锁和不完整缓存检测。
- 环境与场景不匹配在训练前报错。
- 训练中断单独记录，不继续后处理。

测试中的故障、缓存采样拦截等是显式测试夹具；生产流程未修改环境动力学、奖励或MAPPO更新。编译检查及git diff --check通过。仍有既有Gym/Matplotlib弃用提示，不影响上述检查。

## 实际命令与数据检查

在项目根目录实际执行：

```bash
MPLBACKEND=Agg python -m deep_rl_for_swarms.experiments.pursuit.pipeline \
  --steps 8 --output runs/pipeline-check
python -m deep_rl_for_swarms.experiments.pursuit.audit \
  --suite configs/pursuit/evaluation20.json \
  --evaluations runs/pipeline-check/mappo-20261119-105338-seed0/eval_best runs/eval_random \
  --output runs/pipeline-check/metrics-audit.json
```

训练、模型评估、随机基线数据生成均通过；模型与随机对照各20回合、各20480环境步。初态集SHA256为 `7dd4da741f62f2432b99f85d3e0367d08d32641f87284654f02d99cab5ae7f53`，与历史固定20场景相同。随机基线平均回报 -238.13326094872508，与旧随机评估一致。

从保存位置独立复算周期距离、奖励、捕获/超时和总体指标：两组均PASS，最大回报误差分别约3.13e-13、3.98e-13。

实际短训练还验证了 `--skip-evaluation` 和 `--skip-evaluation --skip-visualization`：运行目录位于runs/pipeline-check/skip-checks，前者生成学习曲线但无eval_best，后者只训练且无图表。两次状态均completed。

## 全量媒体与复用验收

**PASS。** 默认stride=4、fps=15，模型与随机各20个场景全部生成trace、MP4、render.json、距离图和轨迹图。40个视频均可解码首帧，格式640×640/15fps；80张PNG可读。首次全流程实际耗时557.33秒，其中训练29.60秒，视频编码占主要时间。

再次运行：

```bash
MPLBACKEND=Agg python -m deep_rl_for_swarms.experiments.pursuit.postprocess \
  --run runs/pipeline-check/mappo-20261119-105338-seed0
```

后处理内部计时约0.19秒（不含解释器启动），全部完成。复跑前后对205个文件比较SHA256和纳秒修改时间：两份results.json、40场景的全部轨迹/媒体/渲染元数据以及3个checkpoint均保持不变，确认未重新采样、编码或修改模型。学习曲线和对照图按日志重绘，summary.md更新为此次后处理的可读汇总，原pipeline.json保留首次流程计时。

小型检查证据保存于[docs/verification/pipeline-checks.json](docs/verification/pipeline-checks.json)。完整本机产物位于runs/pipeline-check/mappo-20261119-105338-seed0/和runs/eval_random/；运行目录是验收用途的8步模型，不能作为训练成绩。

## 限制

- 本次仅8步或4步训练检查，不代表策略已经学会捕获。
- 固定20场景用于开发比较；正式泛化测试应另设未见场景。
- MPS、强制断电/杀进程、跨设备一致性没有在本次验收；并发锁冲突有测试，但没有运行多条长训练进行压力测试。
- runs产物默认被Git忽略；固定evaluation20初态及本记录作为代码/文档保留。
