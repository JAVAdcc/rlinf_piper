# C50 online 训练操作手册

更新日期：2026-10-23。适用本机 `/home/descfly/code/rlinf_piper`，代码分支 `piper/warmup-shared-20261021`，已包含 `cab89606` 的 C50 与 replay 修正。

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
| 等待时 `→` | 第一条开始执行；后续先确认保留上一条、完成训练和权重同步，再开始下一条 |
| 执行中 `b` | 开始 critical segment，切换到共享 C50 actor；已有活动段时不会再开始一个段 |
| 活动段中 `s` / `f` | 结束当前段并标为成功 / 失败，丢弃剩余 actor 队列，恢复 VLA；整条继续执行 |
| `i` | 切换人工接管 / 请求交还 policy；人工控制不会被旧 actor 结果覆盖 |
| 执行中 `→` | 停止整条 episode，进入整条评分 |
| 整条停止后 `s` / `f` | 为整条单独评分，保存数据并复位；不会覆盖各段结果 |
| `←` | 执行时立即停止并丢弃当前条；保存、复位或等待时可撤销尚未确认的上一条 |
| 等待确认时 `q` | 退出；尚未确认的数据保留为 pending，之后可 resume |

正常收尾顺序是：**段内 `s/f` → `→` 停止整条 → 再 `s/f` 评分整条 → 保存与复位 → 恢复场景 → `→` 确认保留并更新**。确认前仍可 `←` 撤销，确认后已经用于训练，不能再撤销。达到最后一条时也需要按 `→` 确认提交；不要在保存结束后直接关闭终端。

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
| 单次确认上限 | 最多 256 次 critic 更新，剩余预算保留到后续确认 |
| actor 频率 | 每 2 次 critic 更新一次 actor |
| 学习率 | actor / critic 均为 `3e-4` |
| target / 折扣 | 每步软更新，`tau=0.005`；`gamma=0.995` 按实际控制 tick 计算 |
| actor 目标 | 前 1,000 次 critic 更新中，BC 权重 5→2，Q 权重由接近 0→0.1 |

模型更新发生在两条 episode 之间。确认后先复用已记录特征，批量补算缺失状态，写入 `online_features.npz`，再训练、保存 checkpoint、同步权重。等待终端提示 `Online update complete` 后才会进入下一条。

例如新入池 100 个 transition，对应 500 次 critic 更新预算；本次执行上限 256 次，剩余 244 次留待后续确认。`version` 记录本轮累计 critic 更新数，不是 episode 数，也不含离线 warmup 的 10,000 步。

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

这里的 100 是累计保留目标，不是额外采集 100 条。resume 从该目录恢复模型、target、optimizer、离线/在线两个数据池、更新预算、随机数和已提交清单。已经提交的数据不会重复入池；pending 数据会继续等待确认。

续训不传新的 actor、offline cache 或 online 配置，不需要再次导入离线 cache。不要用此命令续训旧 C10 目录；更换模型或训练配方时创建新目录。`checkpoints/piper_shared_c50_prejoint_20261022` 是初始化权重目录，不能作为 `--resume` 的参数。

## 6. 查看结果与排查

| 运行目录内文件 | 内容 |
|---|---|
| `online.yaml` | 本轮实际训练配置、初始化权重与离线 cache 绑定 |
| `episode_XXX/result.json` | 分段 / 整条评分、是否 accepted |
| `episode_XXX/trace.jsonl` | 请求目标、实际命令、反馈和 H50 reference |
| `episode_XXX/online_features.json` | 特征复用 / 补算数量与耗时 |
| `episode_XXX/online_update.json` | 实际 actor / critic 更新数、预算余量、loss、version 与耗时 |
| `online/latest.json` | 最后完整发布的 checkpoint |
| `native_logs/` | 原生 actor / rollout worker 日志 |

沿用当前终端的 `piper_online_dir`，可查看最后一次提交：

```bash
cat "$piper_online_dir/online/latest.json"
```

首次确认后若 version 仍为 0，先检查是否已有 64 个在线 transition，或该条是否有已评分的 critical segment。旧 BC-only 包会报 `warmed-up critic` 错误，应核对 `PIPER_ACTOR_PATH` 是否指向本手册的新包。

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
