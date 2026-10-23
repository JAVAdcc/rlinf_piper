# C50 online 训练操作手册

更新日期：2026-10-23。适用本机 `/home/descfly/code/rlinf_piper`，代码分支 `piper/warmup-shared-20261021`，已包含 C50、replay 修正与训练/推理异步执行。新运行默认启用异步；旧目录 resume 沿用其保存的配置。

## 1. 本次使用的权重与数据

云端新包已经下载并校验，启动时使用 **`piper_shared_c50_prejoint_20261022`**：actor 完成 BC5000，critic 完成 TD10000，离线联合更新为 0。两个网络都在 `actor/model_state_dict/full_weights.pt` 中，不需要单独指定 critic 路径。actor 与旧 C50 BC5000 包逐张量相同；新增训练只改变 critic。

| 用途 | 本地路径（相对仓库根目录） |
|---|---|
| 冻结 Stage1，H50 FP32 | `checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor` |
| C50 actor + 已 warmup 的 critic | `checkpoints/piper_shared_c50_prejoint_20261022/actor` |
| 与本次 warmup 匹配的 C50 离线 cache | `outputs/piper/offline_cache/cache_local_segments_c50_s10_20261022` |
| online 训练配置 | `examples/embodiment/config/piper_rlt_online.yaml` |
| 主从臂、相机与复位配置 | `examples/embodiment/config/piper_usb_hardware.yaml` |

离线 cache 已包含 latent、proprio、H50 reference、C50 动作和分段结果，**可以直接加载，不需要重新计算 latent，也不需要下载原始录像**。只导入 train split：40 条 episode、3,399 个 transition；10 条验证 episode、518 个 transition 不进入训练池。

该离线 cache 是云端 warmup 原本使用的 **C50 / stride=10 / pad** 数据，保留原始内容与指纹。本地新在线数据使用 **C50 / stride=2 / align**；两者均为 C50，允许在同一个 batch 中混采。此次不加载旧 C10 online run 的 replay。

原始 `readiness.json` 仍为 false：critic 已完成训练，但 Q 泛化和部分动作检查尚未通过。云端验证 TD RMSE 约 0.250，训练约 0.048。本手册用于实验性 online 测试，不表示真机成功率已经验收；命令保留动作检查。

## 2. 启动一轮新的 online 训练

先退出占用同一机械臂或相机的其他程序。在本机前台终端执行整段命令，不需要手动启动 Ray，也不要同时运行评测入口。

```bash
cd /home/descfly/code/rlinf_piper

export PIPER_PYTHON="$PWD/.venv-piper/bin/python"
export PIPER_STAGE1_PATH="$PWD/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"
export PIPER_ACTOR_PATH="$PWD/checkpoints/piper_shared_c50_prejoint_20261022/actor"
export PIPER_OFFLINE_CACHE="$PWD/outputs/piper/offline_cache/cache_local_segments_c50_s10_20261022"

piper_online_dir="/media/descfly/新加卷/value/data_inference/rlinf_online_c50_$(date +%Y%m%d_%H%M%S)"
printf '本次训练目录：%s\n' "$piper_online_dir"

bash toolkits/piper/train_online.sh \
  --teleop piper_leader \
  --num-episodes 5 \
  --steps 7200 --replan-steps 50 \
  --record-dataset --save-rlt-reference \
  --reset-after-episode initial \
  --output-dir "$piper_online_dir"
```

脚本会加载 Stage1、actor 和 critic，将离线训练数据导入独立 demo 池，建立新的 online optimizer 与 target Q，然后连接相机、主从臂，等待 `→`。Stage1 始终冻结。`version=0` 表示本轮尚未做 online 更新，不表示加载的是随机 critic。

`--num-episodes 5` 是本轮累计保留的 episode 目标；`--steps 7200` 是单条最多 7,200 个控制 tick，并非训练步数。名义控制频率为 30 Hz，推理与等待会影响实际耗时。`--replan-steps 50` 保持 C50 执行，按键和人工接管仍可中途打断。

`--save-rlt-reference` 记录每次实际给 actor 使用的完整 H50 reference，便于分析动作与松爪。`initial` 在第一条开始时记录复位目标，之后每条结束或丢弃后复位；首次启动不会先自动回到某个预设姿态。

## 3. 采集、标注与确认更新

保持运行终端处于焦点，按以下顺序操作：

| 时机 / 按键 | 行为 |
|---|---|
| 等待时 `→` | 第一条开始执行；后续确认保留上一条并提交后台，立即开始下一条；无需等训练完成 |
| 执行中 `b` | 开始 critical segment，切换到共享 C50 actor；已有活动段时不会再开始一个段 |
| 活动段中 `s` / `f` | 结束当前段并标为成功 / 失败，丢弃剩余 actor 队列，恢复 VLA；整条继续执行 |
| `i` | 切换人工接管 / 请求交还 policy；人工控制不会被旧 actor 结果覆盖 |
| 执行中 `→` | 停止整条 episode，进入整条评分 |
| 整条停止后 `s` / `f` | 为整条单独评分，保存数据并复位；不会覆盖各段结果 |
| `←` | 执行时立即停止并丢弃当前条；保存、复位或等待时可撤销尚未确认的上一条 |
| 等待确认时 `q` | 退出；尚未确认的数据保留为 pending，之后可 resume |

正常收尾顺序是：**段内 `s/f` → `→` 停止整条 → 再 `s/f` 评分整条 → 保存与复位 → 恢复场景 → `→` 确认保留并提交后台**。确认前仍可 `←` 撤销，确认后已经用于训练，不能再撤销。达到最后一条时也需要按 `→` 确认提交；程序会在机械臂停止后等待后台处理完已确认数据与训练预算，再正常退出。等待期间可按 `q`，保存剩余工作供 resume。

没有已评分 critical segment 的 episode 不产生训练样本。成功段的最后有效 tick 奖励为 1，失败段为 0，段与段之间独立终止。

## 4. 每次确认后的训练配方

以本地 `examples/embodiment/config/piper_rlt_online.yaml` 为准。云端包内说明针对旧代码，包含旧的 32 critic / 8 actor、stride50 配方，**不要照搬，也不要应用包内旧 `c50_online.patch`**；本地已包含 C50 支持及后续 replay 修正。

| 项目 | 当前配置 |
|---|---|
| 在线窗口 | 长度 50、stride=2，窗口由实际发送动作组成，不跨关键段 |
| 段尾 | 补一个对齐段末的完整窗口；不足 50 步的短段保留有效长度 mask |
| batch | 64 = 32 online + 32 offline，有放回采样 |
| 在线池 | 从最近 200 条有效 episode 的 transition 中均匀采样 |
| 离线池 | 保留全部 3,399 个训练 transition；validation 不入池 |
| 开始更新 | 累计至少 64 个在线 transition；导入离线数据本身不触发更新 |
| 更新预算 | 每个新增在线 transition 产生 5 次 critic 更新预算 |
| 后台每轮 | 最多 32 次 critic 更新并发布 checkpoint；有预算则自动继续下一轮 |
| actor 频率 | 每 2 次 critic 更新一次 actor |
| 学习率 | actor / critic 均为 `3e-4` |
| target / 折扣 | 每步软更新，`tau=0.005`；`gamma=0.995` 按实际控制 tick 计算 |
| actor 目标 | 前 1,000 次 critic 更新中，BC 权重 5→2，Q 权重由接近 0→0.1 |

确认后后台先复用已记录特征，按 batch=2 补算缺失状态，写入 `online_features.npz`，再送入独立 learner。learner 可与下一条推理同时运行；完整 checkpoint 发布后，rollout 只在下一条 episode 开始前加载最新版本；整条 episode 的所有 C50、critical phase 切换和人工交还都固定使用该版本。若上一条尚未补算/训练完成，本条会沿用此前版本，执行途中不会追赶新版本。

本机仍只有一份冻结 VLA。特征补算每个小 batch 之间让等待中的推理优先，但无法抢占正在执行的 GPU forward；训练与推理也共享单卡算力，因此异步不等于推理零延迟或严格 30 Hz。`trace.jsonl` 保留推理耗时、补算等待和实际动作版本，便于现场评估。

例如新入池 100 个 transition，对应 500 次 critic 更新预算；后台以每轮最多 32 次连续消耗这 500 次预算，不需要等待下一次确认。`version` 记录本轮累计 critic 更新数，不是 episode 数，也不含离线 warmup 的 10,000 步。

## 5. 退出后继续同一轮训练

优先在等待场景恢复时按 `q` 退出。重新打开终端后，把下面目录换成第一次启动时打印的**实际目录**：

```bash
cd /home/descfly/code/rlinf_piper
export PIPER_PYTHON="$PWD/.venv-piper/bin/python"
piper_online_dir="/media/descfly/新加卷/value/data_inference/rlinf_online_c50_实际时间戳"

bash toolkits/piper/train_online.sh \
  --resume "$piper_online_dir" \
  --num-episodes 100
```

这里的 100 是累计保留目标，不是额外采集 100 条。resume 从该目录恢复模型、target、optimizer、离线/在线两个数据池、更新预算、随机数和已提交清单。已经入池的数据不会重复入池；已确认但未处理完的数据会自动恢复后台处理，尚未确认的数据继续等待确认。退出时只完成当前 learner 小轮并保存，不强制耗尽全部预算。异步调度下恢复计数、池与随机数状态，但不保证真实轨迹或推理/训练时序逐位复现。

续训不传新的 actor、offline cache 或 online 配置，不需要再次导入离线 cache。不要用此命令续训旧 C10 目录；更换模型或训练配方时创建新目录。`checkpoints/piper_shared_c50_prejoint_20261022` 是初始化权重目录，不能作为 `--resume` 的参数。

## 6. 查看结果与排查

| 运行目录内文件 | 内容 |
|---|---|
| `online.yaml` | 本轮实际训练配置、初始化权重与离线 cache 绑定 |
| `episode_XXX/result.json` | 分段 / 整条评分、是否 accepted |
| `episode_XXX/trace.jsonl` | 请求目标、实际命令、反馈和 H50 reference |
| `episode_XXX/online_features.json` | 特征复用 / 补算数量与耗时 |
| `episode_XXX/online_update.json` | 该条入池的第一轮更新结果；后续预算消耗查 async_metrics |
| `online/latest.json` | 最后完整发布的 checkpoint；learner version 可能领先当前 rollout |
| `online/async_metrics.jsonl` / `async_status.json` | 每轮后台更新 / 最新一轮的计数、预算、耗时和最后推理版本 |
| `native_logs/` | 原生 actor / rollout worker 日志 |

沿用当前终端的 `piper_online_dir`，可查看最后一次提交：

```bash
cat "$piper_online_dir/online/latest.json"
```

确认是非阻塞的；后台尚未完成、或下一条 episode 尚未开始时，rollout version 保持原值。持续为 0 时检查是否已有 64 个在线 transition，或该条是否有已评分的 critical segment。旧 BC-only 包会报 `warmed-up critic` 错误，应核对 `PIPER_ACTOR_PATH` 是否指向本手册的新包。

动作检查报错时保留 trace 与报错。云端曾记录某输入首目标 J4 偏差 8.761°，超过当前 5° 门限，本手册没有关闭该检查。相机 RGB 节点缺失时依照 `order.md` 的相机恢复说明处理，不手工换成红外节点。

2026-10-23 拉取时，系统盘可用约 11 GB，数据盘约 19 GB。录像和 replay 会持续占用空间；出现 Ray `/tmp/ray` 磁盘超过 95% 的提示时，应处理对应文件系统空间，不要把提示当作训练已经失败或屏蔽掉。

## 7. 下载与核验记录

本次通过 `ssh yiming-aliserver` 使用本机已配置的 **25 端口**下载。云端来源：

- 权重包：`/home/yiming/workspace/code/rlinf_piper_warmup_20261019/deployment/piper_shared_c50_prejoint_20261022.tar.gz`
- 离线 cache：`/home/yiming/workspace/code/rlinf_piper_warmup_20261019/prepared/cache_local_segments_c50_s10_20261022`
- 原始部署说明：`/home/yiming/workspace/code/rlinf_piper_warmup_20261019/DEPLOY_C50_ONLINE.md`

权重 SHA256：`fb26f7d3e607ff221dff309f6cbb0a1299fe7207cb2a83098ec0457b9e2385d0`。

cache fingerprint：`0cda0bdb431bf28f641edf468c40a5da8ae0d393603977dab0c894ef0e47d7f2`。

Stage1 checkpoint ID：`b1f2800f7b294b4c95834ad60e0ff23d`。复用现有 Stage1，没有重新下载大基座。

本地收据与原始压缩包位于 `outputs/piper/deployment_c50_online_20261023/`。`local_validation.json` 记录权重与全部 cache shard 校验、训练/验证划分及配置兼容性；`native_initialization.json` 记录 GPU 模型初始化、离线池导入和权重同步验证。上述检查均已通过，初始化后保存的全部 40 个 actor/critic 张量与下载源逐位相同，online 更新数为 0。验证没有连接相机或机械臂。

## 8. episode 内固定权重与异步实现

参考 [RLToken §5](https://arxiv.org/html/2604.23073v2#S5) 的异步 rollout/learning 以及 [openpi-RLT LearnerService](https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/src/rlt_online_rl/trainer.py) 和 [推理快照轮询](https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/src/rlt_online_rl/inference.py)。保留 RLinf 原生训练、replay 与 checkpoint，用发布完整快照的方法解耦 learner 和 rollout；不是直接移植上游的 JAX 服务。

入口命令与之前相同，`piper.async_training: true` 已是默认配置。若需要做同步对照，可复制配置并设为 false，用 `--online-config /path/to/profile.yaml` 新建运行；不要修改已有运行的配置。异步模式保留按 episode 确认入池的撤销窗口，不会在本条尚可撤销时提前训练。

2026-10-23 异步验证记录：在单张 RTX 5090 上使用真实 H50/C50 模型和合成数据，通过首次运行与恢复测试。首次确认提交耗时约 0.6 ms，退出保存 version 8 / 剩余 192 次预算；恢复期间 97 次推理请求中有 48 次与训练 RPC 重叠，最终 learner 到 version 200、预算为 0，episode 只入池一次。恢复期间推理含调度等待的中位数约 207 ms、P95 约 215 ms。测试使用小 batch=4、每轮更新上限 4，不能直接视作正式 batch=64 的性能，且没有连接机械臂。报告保存在 `outputs/piper/async_smoke_20261023/async_first.json` 和 `async_resume.json`。

2026-10-23 后续修正：权重安装从“每次 C50 推理前”改为“整条 episode 开始前”，使用显式 begin_episode 调用；训练和特征准备仍异步。此前运行的 trace 可能包含多个版本，修改不会重写历史数据。resume 也从下一条开始采用固定版本规则，运行中的旧进程需退出并重新启动才能生效。

checkpoint 仍按每轮最多 32 次 critic 更新保存完整训练状态；总预算仍为新增 transition 数乘 5。默认保留最近 3 代，恢复进程另外保留正在引用的加载代。该频率影响恢复粒度和磁盘写入，不再决定执行中换权重的频率。
