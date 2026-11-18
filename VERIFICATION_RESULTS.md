# 第二问迁移验收结果

日期：2026-11-18；分支：`codex/adapt`。本次实现尚未自动提交 Git，基于用户文档提交 `cefe68b` 的工作区修改。原始环境参考提交为 `2e51997b30b49c2d288f841ba648cc1d193fe3b8`。

## 结论

1. **迁移运行：通过本次 CPU 单环境验收。** Python 3.10.22 / ARM64，环境、MAPPO 更新、模型保存/加载、独立评估、图表和视频导出均已执行。
2. **任务与测量一致性：在明确范围内通过。** 原 `ma_envs/` 的 14 个文件字节不变；原提交环境在独立进程与适配层对照，5 组各 1024 步，共 5120 步，位置误差为 0；Actor 观测的最大差异为 2.98022068135e-08，来自 float64→float32。使用同一现代依赖栈，不是历史软件栈等价性证明，也不是所有初态的数学证明。
3. **策略效果：尚未学会稳定捕获。** 三种子各 8192 步是迁移后的短程检查，不是充分训练。固定 20 个独立测试初态下，各模型均为 0/20 成功；不应作为最终作业训练成绩。回报略有改善不能替代成功率。

## 环境与实现

- Conda：`swarm-rl`，解释器 `/opt/anaconda3/envs/swarm-rl/bin/python`。
- Python：3.10.22；系统：Darwin 25.6.0；架构：arm64。
- PyTorch 2.5.1、NumPy 1.26.4、SciPy 1.13.1、Gym 0.23.1、Shapely 1.8.5.post1；完整版本见 [锁文件](requirements-macos.lock.txt)。
- 原动力学、prey 脚本、奖励、距离、观测、捕获/超时等源码全部保留。修改安装元数据，另加 wrapper、PyTorch MAPPO、实验和可视化模块。
- Actor 使用原 503 维局部观测；共享 Actor 输出二维 tanh 高斯动作，概率包含 Jacobian 修正。集中式团队 Critic 使用 42 维训练特征，未给 Actor 注入全局状态。
- 使用参考文件演示配置：10 追捕者、1 prey、100×100 周期场地、unicycle、通信半径 `200*sqrt(2)`。不能把结果称为小通信半径实验。
- 原始依赖清单保留在 [requirements-legacy.txt](requirements-legacy.txt)。未迁移原 TF1/TRPO 训练入口，也未把随机策略称作 TRPO baseline。

## 已执行检查

| 检查项 | 状态 | 证据与范围 |
| --- | --- | --- |
| 依赖安装与本地 editable 安装 | PASS | `swarm-rl` 安装成功；`python -m pip check` 无冲突 |
| 语义与训练测试 | PASS | 21 项：捕获阈值、1023/1024 步、同时捕获/超时、随机性隔离、周期距离、原奖励、GAE、动作概率、参数更新、保存加载、独立评估、渲染及已知错误传播 |
| 完整随机动作回合 | PASS | 1024 步正常超时，初次实测约 1.1 秒；不是通用性能保证 |
| 新旧环境逐步对照 | PASS | 原始源码独立进程；随机动作 3 组、零动作 1 组、含越界裁剪动作 1 组；比较 reset/每步位置、prey、观测、距离、奖励和 done |
| 原环境源码一致 | PASS | 14 个文件相对原始提交字节一致，SHA256 列表已保存 |
| 三种子短训练 | PASS | seeds 0/1/2，各 8192 环境步；无非有限损失/梯度；5 个固定验证初态、每 4096 步评估 |
| checkpoint 与独立测试 | PASS | best checkpoint 仅由验证集选择，另生成测试集 seed 200000；每种方法 20 个相同初态 |
| 指标独立复算 | PASS | 5 种策略共 100 回合，从位置重算周期距离、原奖励与成功/超时/回报；最大回报复算误差小于 7e-13 |
| 静态图与视频 | PASS | 场景 0 的未训练/训练后图表与视频；跨边界断线，明确标记超时，无成功视频伪装 |
| 历史依赖栈对照 | NOT RUN | 未安装旧 NumPy/SciPy/TensorFlow 运行栈；新旧源码使用相同现代依赖 |
| 全新 Conda 环境重建 | NOT RUN | 提供锁文件与 yml，本次在已有环境安装；没有另建第二个干净环境 |
| MPS / 多进程 | NOT RUN | 验收设备为 CPU；当前实现为单环境采样 |
| 充分训练与 100 场景正式测试 | NOT RUN | 当前仅各 8192 步 / 20 场景；没有达到建议的完整效果实验规模 |

已知 Gym 停维护提示、Matplotlib/Pyparsing 弃用提示及 Agg 非交互提示未影响这些检查；没有用升级 NumPy 2 或替换任务环境来消除提示。

## 短程效果结果

| 方法 | 训练环境步 | 测试回合 | 成功率 | 成功条件平均步数 | 失败按上限计入用时 | 原始平均回报 |
| --- | --- | --- | --- | --- | --- | --- |
| random | 0 | 20 | 0% | N/A | 1024 | -238.133261 |
| untrained | 0 | 20 | 0% | N/A | 1024 | -234.841977 |
| mappo0 | 8192 | 20 | 0% | N/A | 1024 | -225.493424 |
| mappo1 | 8192 | 20 | 0% | N/A | 1024 | -228.662763 |
| mappo2 | 8192 | 20 | 0% | N/A | 1024 | -226.047826 |

回报为一份原始共享奖励累计，未按追捕者数量乘十。全部失败时成功条件步数为 N/A。未训练对照来自 seed 0 的初始化模型，不是三种子未训练对照的平均。短训练的 best checkpoint 在已训练候选中按验证成功率、惩罚用时、原回报排序，未使用最终测试选择模型。

本批测试集已用于检查和报告，今后若根据这些结果调整超参数，应将它视为开发集，正式成绩另生成并冻结新的未见测试初态。不能反复用同一“最终测试集”调优后再宣称独立测试。

## 公平性与行为变化说明

- **任务源码未变**：原奖励为 `-min(d_min, obs_radius)/obs_radius`；参考配置捕获为严格 `d_min<1`；1024 步上限与原先 scripted-prey-first 更新顺序保留。
- **表示发生变化**：wrapper 观测转 float32，空间元数据改为无界，避免对负值角度特征错误裁剪；未改变原观测的含义。
- **随机性变得可控**：wrapper 隔离原全局 RNG，固定初态在原 World.reset 前注入；跨版本对比保留具体初态文件，不能仅依赖 seed。
- **训练过程改变**：从原 TRPO 思路改为新 MAPPO，集中式 Critic 与 timeout bootstrap 是新训练实现，不能声称算法更新过程与旧 TRPO 一样。可比较的是固定任务与统一测量。
- **旧问题保留并披露**：原默认 `2D_rbf`、direct 分支和 reward_mech 属性未重写；新入口只接受已支持的显式配置。prey 精确中线会报错的已有问题有回归测试，未静默替换 prey 或重抽场景。
- 对捕获临界状态使用预先设定的测试与容差。五组轨迹对照没有覆盖所有几何情况；不承诺依赖版本变化后所有轨迹逐位一致。

## 可复现证据与展示入口

- [环境逐步对照记录](docs/verification/environment.json)
- [原环境文件哈希](docs/verification/environment-source-hashes.json)
- [固定测试初态](docs/verification/test-suite.json)
- [指标独立复算](docs/verification/metrics-audit.json)
- [效果汇总及模型/场景哈希](docs/verification/experiment-summary.json)
- [原始学习曲线](runs/figures/learning.png)
- [固定场景 0 的轨迹](runs/eval-mappo0/trajectory-0000.png) 与 [距离曲线](runs/eval-mappo0/distance-0000.png)
- [场景 0 学习前后并排视频](runs/figures/comparison-0000.mp4)：左侧未训练，右侧短训练 seed 0；两边均超时，不作为成功演示。

`runs/` 不默认加入 Git。小型 JSON 证据复制到 `docs/verification/`，便于提交审核；模型、全轨迹和媒体在本机保留。图表输入列表和渲染元数据在对应输出目录。

## 核心命令

```bash
conda activate swarm-rl
python -m pip check
MPLBACKEND=Agg python -m pytest -q
MPLBACKEND=Agg python -m deep_rl_for_swarms.experiments.pursuit.verify
python -m deep_rl_for_swarms.experiments.pursuit.audit --suite runs/test20.json --evaluations runs/eval-random runs/eval-untrained runs/eval-mappo0 runs/eval-mappo1 runs/eval-mappo2 --output runs/verification/metrics-audit.json
```

训练与展示完整命令见 [README](README.md)。后续应在同一任务规则下增加训练预算、检查探索与学习稳定性，再在新冻结测试集报告成绩。奖励塑形、降低 prey 速度或调整观测半径应作为另行说明的实验条件，不并入本次等价性结论。
