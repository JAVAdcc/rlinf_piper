**RLinf critical phase 数据采集、离线 warmup 与共享 actor 评测（2026-10-21）**

**最新 C30/C50 dropout 权重：使用本文末尾的 [reference dropout 启动命令](#2026-10-23-reference-dropoutc30--c50-新权重)，包含固定权重评测、online 训练及续训。**

**C50 actor + critic 的完整 online 启动与续训命令见 [ONLINE_C50.md](ONLINE_C50.md)。** 2026-10-23 已拉取 `piper_shared_c50_prejoint_20261022`（BC5000 + critic10000）及匹配的 C50 离线 cache。

沿用前面的 SFT 验收流程：global_step_50000 模型自动执行，人工按 `b` 标记关键阶段、用 `s/f` 结束并评分当前关键段，整条 episode 停下后另行评分。新标注为 v3，warmup 已接通局部终止训练。critical phase 指轨迹中的关键动作区间。第 1 节是纯 SFT 验收；需要主臂人工接管时，使用第 5 节的命令，入口和标注方式相同。中断后继续采集或增加目标条数，使用第 8 节的 `--resume`。共享 RLT actor 的本机路径、评测和续采命令见第 9 节，原部署流程见 [共享网络部署说明](docs/piper_shared_deployment.rst)。

**1. 启动采集**

在本机前台终端执行整段命令。先测试 20 条，正式采集时将 `piper_episodes` 改为 `100`。使用包含 ``critical-segment-v1`` 采集协议的代码：b 开始、s/f 结束并评分局部，右键结束 episode 后独立评分整条；下文说明按键和 v3 数据的训练边界。

```bash
cd /home/descfly/code/rlinf_piper

piper_episodes=20
piper_run_dir="/media/descfly/新加卷/value/data_inference/rlinf_sft50k_critical_$(date +%Y%m%d_%H%M%S)"

export PIPER_PYTHON="/home/descfly/code/rlinf_piper/.venv-piper/bin/python"
export PIPER_STAGE1_PATH="/home/descfly/code/rlinf_piper/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"

printf '本次保存目录：%s\n' "$piper_run_dir"
bash toolkits/piper/eval_stage1.sh rollout \
  --num-episodes "$piper_episodes" \
  --steps 3600 \
  --replan-steps 50 \
  --record-dataset \
  --streaming-video \
  --save-observations \
  --play-sounds \
  --no-action-limits \
  --reset-after-episode initial \
  --output-dir "$piper_run_dir"
```

脚本固定加载 H50 FP32 Stage 1 权重、三路相机和双 USB 任务提示词，名义控制频率为 30 Hz，每次执行 50 个目标后重新推理；每条最多执行 3,600 个目标。新采集使用 `--output-dir` 指定新目录，程序自动创建；续采使用 `--resume` 指向原目录。采集前先退出占用同一机械臂或相机的其他程序。

前视相机按序列号和 RGB 接口 `04` 选择，不能用红外节点替代。Orbbec SDK 使用后偶尔会留下 USB 在线、RGB 视频节点缺失的状态；配置中的 `reset_on_connect: true` 现在也会在启动解析不到节点时，检查占用并定向 reset 该相机、重新解析一次。仍缺失或被其他进程占用会停止启动。视频编号可因恢复而变化，无需手动修改硬件配置。该恢复同样适用于第 9、10 节的评测和 online 入口。

`--no-action-limits` 和 `--reset-after-episode initial` 沿用原文件设置：前者关闭额外的 rollout 动作范围/步幅检查，驱动行程限制仍保留；后者在每条评分并保存后，张开夹爪、返回第一条开始时的姿态，再恢复夹爪位置。若本次希望手动复位，将后者改为 `--reset-after-episode none`。

复位在整条 episode 独立评分并保存完成后，或 `←` 撤销并释放录制资源后开始；首次启动程序不会先复位。`initial` 会将第一条开始时的关节/夹爪读数限制到驱动的合法目标范围，并在 `reset_pose.json` 中同时记录原始读数和实际复位目标，避免零位附近的小幅反馈偏差使复位被拒绝。若要每回合返回零位，使用 `--reset-after-episode zero`。

**2. 按键与标注**

模型加载、相机连接和从臂使能完成后，程序等待右方向键。保持终端焦点，按键无需回车。

| 按键 | 行为 |
| --- | --- |
| `→` | 就绪时开始；运行中停止下发新动作、保持最后目标并进入评分。 |
| `b` | 进入 critical phase；段内重复 b 只提示需要 s/f 评分。 |
| `s` / `f` | 运行中只关闭并评分当前关键段，继续普通阶段；整条任务停止后的独立评分界面才保存全局结果。 |
| `←` | 直接停止并丢弃当前回合，无需评分；保存、复位或等待阶段也能撤销刚结束的回合。清理和复位完成后按 `→` 重采。 |
| `r` | 复位失败时重试；续采发现上次复位未完成时，确认工作区后按此键完成复位。 |
| `q` / `Ctrl-C` | 退出；此前完成的回合保留，当前未评分回合不进入训练标注清单。 |

正常顺序是 `b → s/f` 标第一段，再 `b → s/f` 标第二段；按 `→` 结束本回合后，另按 `s/f` 评分整条 episode。局部结果不自动决定全局结果。

如果结束时段仍开启，按 `→` 停止下发新动作、先 `s/f` 补评局部，再 `→` 确认进入整体评分，最后 `s/f`。达到步数上限后同样先补评打开段，再右键确认整体评分。重复 s/f 不能越过确认键。零动作空段会删除；等待评分时不会恢复执行。左键、q 退出优先于同批评分。

全回合由选定的 policy 执行，b 和局部 s/f 只改标注，不切换 actor/reference；右键结束不是硬件急停。视频随采集编码，整回合终止观测在等待评分前捕获，时间戳使用捕获时刻。保存和已配置复位完成后，再恢复场景、按右键开始下一条。采够指定条数后仍可左键撤销、右键确认结束。

**3. 采集后检查数据**

在同一终端运行下面的读取命令。若换了终端，先将 `piper_run_dir` 设置为第 1 步打印的实际目录。

```bash
python3 - "$piper_run_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
labels = json.loads((root / "annotations.json").read_text())
print("保存目录：", root)
for key in ("status", "completed_episodes", "successes", "failures", "success_rate",
            "autonomous_success_rate", "intervention_rate",
            "recorded_episodes", "phase_labeled_episodes", "discarded_episodes"):
    print(f"{key}: {summary.get(key)}")
for episode in labels["episodes"]:
    print(episode["dataset"], episode["outcome"],
          "critical_phases=", episode["critical_phases"],
          "terminal_frame=", episode["terminal_frame"])
PY
```

本次保存目录中：

- `summary.json`：成功率、完成数、录制数和有效阶段标签数。
- `annotations.json`：各回合的多段 `[start, end)`、终止帧及成功/失败标签。
- `episode_001/dataset/` 等：LeRobot 数据，含三路视频、状态、实际下发动作、逐帧 critical 标记和终止观测。
- 每条回合的 `result.json`、`trace.jsonl`：回合结果与执行记录；`observations/` 为每次推理时的快照，完整逐帧数据在 `dataset/`。
- `policy.json`、`hardware.yaml`、`run.json`：本次权重元数据、硬件配置和启动参数。
- `reset_pose.json`：整次采集固定使用的复位目标；各回合的 `reset.jsonl` 记录实际复位过程。复位出错时先查看 `result.json` 的 `reset.error`，程序保持连接并等待 `r` 重试或 `q` 退出。
- `discarded/`：左键撤销的采集尝试，保留结果、trace、已写入的文件和撤销时间供检查；未评分的尝试保留 `original_outcome=unscored`，视频可能不完整。这些目录不进入 `annotations.json`，也不计入成功率。

`start` 包含该帧动作，`end` 不包含；局部段在 episode 停止后补评分时，`end` 等于 `terminal_frame`。每段独立的 `outcome` 保存在 v3 annotations/result 中，LeRobot 的 episode_success 仍只表示全局结果。终止帧只有观测，`action_valid=False`，不会作为训练动作。

**4. 数据确认后运行离线 warmup**

本节命令按格式区分 v2 全局回报与 v3 局部终止回报。不得在 v2 内补写局部 outcome 或混用版本。v3 共享 actor 可通过显式手动评测入口加载；默认入口及专用网络仍拒绝，见上方共享网络部署说明。

先用 `init` 固定训练/验证划分，这一步不启动训练或机械臂。需要有足够的独立回合用于划分；正式训练同时保留成功和失败。完全漏标的回合默认报错：补标后再执行，或明确加 `--skip-unlabeled` 排除这些回合。已有输出文件不会被覆盖。

在完成采集的同一终端执行以下命令，复用 `piper_run_dir` 和 `PIPER_STAGE1_PATH`。如果换了终端，先将这两个变量设为实际采集目录和对应的 Stage 1 权重。下面的路径数组供后续步骤共用，缓存和训练输出都保存在本次采集目录下。

```bash
cd /home/descfly/code/rlinf_piper

piper_warmup_args=(
  "stage1=$PIPER_STAGE1_PATH"
  "annotations=$piper_run_dir/episodes.json"
  "cache=$piper_run_dir/rlt_cache"
  "output=$piper_run_dir/rlt_warmup"
)

USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup init \
  --collection "$piper_run_dir/annotations.json" \
  "${piper_warmup_args[@]}"
```

合并多次采集时，给 `init` 重复传入 `--collection /实际采集目录/annotations.json`。同一回合的所有 critical 区间会进入同一个 split，训练和验证不会混用同一回合。

然后计算一次冻结 Stage 1 的 latent/reference 缓存，再启动小型 RLT actor-critic 训练。这两步使用 GPU，不连接机械臂；采集和 SFT 验收本身不需要提前计算 latent cache。

```bash
USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup cache \
  "${piper_warmup_args[@]}"

USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup train \
  "${piper_warmup_args[@]}"
```

默认配置在 `examples/offline_rl/config/piper_rlt_warmup.yaml`：5,000 步 actor BC、10,000 步 critic、5,000 步联合更新，batch size 为 64，联合阶段每 4 步更新一次 actor。训练只读缓存，按 episode 等概率、再按 chunk 等概率采样；验证池独立。正式数据门槛包括训练集成功/失败各至少 20 条、验证集各至少 10 条，以及至少 600 条训练 transition。

缓存中断后重新执行相同的 `cache` 命令即可校验并继续。更换权重、标注或抽帧配置时使用新的缓存目录。训练中断后保留同一组路径和配置，使用下面的恢复命令；若只想先跑到第 1,000 步，首次训练时在 `train` 后加 `--until-step 1000`。

```bash
USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup train --resume \
  "${piper_warmup_args[@]}"
```

训练后查看验证报告：

```bash
python3 -m json.tool "$piper_run_dir/rlt_warmup/readiness.json"
```

`metrics.jsonl` 记录训练和验证指标，`latest.json` 指向恢复 checkpoint，`actor/` 保存导出的 Stage 2 actor。固定预算结束且最后 3 次定期验证全部通过，才会得到 `ready_for_deterministic_eval=true`。当前 `ready_for_online_exploration` 始终为 `false`，离线 warmup 完成后仍需确定性真机验收和后续 online 接入。详细设计见 `docs/piper_offline_warmup.rst`。

**5. 启用主臂 human intervention**

HIL 已接入同一个采集入口；在原命令上增加 `--teleop piper_leader` 即可。当前配置沿用 Evo-RL USB 任务的主臂序列号 `004F003C4648571020363032`，从臂为 `003F00334648571220363032`。按序列号解析 CAN 通道，不依赖 `can0/can1` 编号。更换主臂时可加 `--leader-usb-serial 序列号`。

本机 `.venv-piper` 已安装 HIL 依赖。下面命令会连接两臂；按右方向键后，主臂先用约 3 秒对齐从臂，再开始回合。建议先验收 2 条，再把回合数改为 20 或 100。

```bash
cd /home/descfly/code/rlinf_piper

piper_run_dir="/home/descfly/code/rlinf_piper/data/rlinf_sft50k_hil_$(date +%Y%m%d_%H%M%S)"
export PIPER_PYTHON="/home/descfly/code/rlinf_piper/.venv-piper/bin/python"
export PIPER_STAGE1_PATH="/home/descfly/code/rlinf_piper/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"

bash toolkits/piper/eval_stage1.sh rollout \
  --teleop piper_leader \
  --num-episodes 50 \
  --steps 7200 --replan-steps 50 \
  --record-dataset \
  --streaming-video \
  --play-sounds \
  --no-action-limits \
  --reset-after-episode initial \
  --output-dir "$piper_run_dir"
```

这条命令每回合最多执行 7,200 个控制 step，按名义 30 Hz 约为 4 分钟，实际时长还受取帧、推理和写盘影响；它不是训练步数。`--record-dataset` 已保存完整的逐帧训练数据和视频。需要额外保存每次推理的 RGB/state/task 快照用于排查时，再加 `--save-observations`。

| 操作 | 行为 |
| --- | --- |
| `→` | 主臂对齐从臂，然后开始本回合。对齐中可按 `q` 退出。 |
| `i` | 接管：主臂进入重力补偿，从臂跟随主臂；reference / actor 都不能覆盖人工动作。 |
| 再按 `i` | 交还：两臂先保持最后的人工目标，等当前观测的新预测返回；接管前剩余动作作废。 |
| `b` | 开始关键段，用 s/f 结束并评分；接管不自动更改 critical 标签。 |
| `s/f` | 运行中只评当前关键段；右键结束整回合后独立评分，随后保存、复位。 |
| `←` | 运行中或人工接管时直接停止并丢弃当前回合，无需 `s/f`；保存、复位和等待下一条时也可撤销。就绪后按 `→` 重采。 |
| `q` / `Ctrl-C` | 退出，已完成回合保留。 |

若习惯 Evo-RL 的 `o` 键，可加 `--intervention-key o`。交还后的等待保持仍记为人工段；普通推理等待记为 `hold`。采集线程在推理期间继续响应按键，但实际周期还受相机、CAN 和写盘影响，30 Hz 是名义配置。

采集后用第 3 节检查数据；第 4 节按 v2/v3 分别使用全局/局部回报。数据会保存逐帧 `is_intervention`、`critical_phase`、实际下发目标与请求目标；`action_source` 的编码为 `0=reference、1=actor、2=human、3=hold`，另存于 `dataset/meta/piper_control.json`。`policy_action_valid=False` 表示该帧没有有效 policy 输出，不能把占位零值当作预测。

`summary.json` 中 `success_rate` 是最终成功率，`autonomous_success_rate` 只把未接管的成功计入分子，分母仍为全部已评分回合；`intervention_rate` 是发生接管的回合比例。每回合另存 `takeovers` 和 `human_steps`。验收 SFT 自主效果时看自主成功率，避免把人工救回算成模型独立成功。

HIL 采集前不需要 latent cache；新 v3 使用 `warmup init -> cache -> train` 的局部终止路径；相同边界的旧特征可用 `relabel-cache` 在新目录复用。当前 critical-only warmup 只取标注区间，区间外的人工动作保留在原始数据中；混合回合不要整条改成 `behavior=human`。

原生在线 `PiperPolicyEnv` 已接入接管动作和标签回传，交还到新 chunk 才生效；它在分布式推理等待期间仍不能像本地入口一样连续响应接管。在线有效长度、跨阶段折扣和数据池恢复仍需后续接通，不能直接把 HIL 采集通过视为在线探索就绪。实现细节见 `/home/descfly/code/rlinf_piper/docs/piper_human_intervention_design.rst`。本次开发使用 CPU 和 mock 验证，尚未启动真机验收。

**6. 采集语音播报**

第 1、5 节的本地交互采集入口均已接入语音，默认开启英文播报，与 Evo-RL 使用同一个本地 Speech Dispatcher 后端。原有命令不加参数也会播报；上面的 `--play-sounds` 显式开启。需要中文时增加 `--speech-language zh`，关闭时把 `--play-sounds` 换成 `--no-play-sounds`。这些参数不影响数据、标注或动作来源，也不启动额外的模型服务。

语音使用表示当前操作的短句，英文为 1–3 个单词，中文为 2–5 个字。回合编号、按键和完整说明保留在终端，语音按下面的短句播报。

| 操作 / 状态 | 英文（默认） | 中文 |
| --- | --- | --- |
| 等待 / 开始 / 主臂对齐 | Ready / Start / Aligning | 就绪 / 开始 / 对齐 |
| `b` / 局部 `s/f` | Critical on / Segment succeeded or failed | 关键段开始 / 关键段成功或失败 |
| 补评局部 / 空段 | Score the segment / Empty phase | 请评关键段 / 空段 |
| `i` 接管 | Human control | 人工接管 |
| 交还后等待预测 / 新 policy 动作生效 | Predicting / Policy control | 等待预测 / 模型接管 |
| 进入整体评分 / 等待整体结果 | Right to score the episode / Score the whole episode | 按右键进入整条评分 / 请评整条任务 |
| 整体 `s/f` | Episode succeeded / Episode failed | 整条任务成功 / 整条任务失败 |
| 保存中 / 保存完成 | Saving / Saved | 保存中 / 已保存 |
| 漏标 critical phase | Unmarked phase | 未标关键段 |
| 复位开始 / 轨迹结束 / 失败 | Resetting / Reset finished / Reset failed | 复位中 / 复位结束 / 复位失败 |
| `←` / 无可撤销回合 | Discarded / Nothing to discard | 已丢弃 / 无可丢弃 |
| 最后一条等待确认 | Confirm finish | 确认结束 |
| 续采 / 等待恢复复位 | Resuming / Reset pending | 续采 / 待复位 |
| 完成 / 停止 / 故障 | Complete / Stopped / Error | 采集完成 / 已停止 / 故障 |

评分和标注写入成功后播报，保存完成在文件关闭并更新清单后播报；“模型接管”仍等新的 policy 动作实际下发后才播报。“复位结束”表示轨迹发送完成，实际姿态和误差继续查看终端。

播报在后台串行播放，不在控制循环中等待音频。快速连续操作时，尚未播放的同类状态提示会合并为最新状态，过期提示会丢弃，终端日志保留完整事件。正常结束会保留最后的评分、保存和复位结果；退出或故障则优先播报停止。程序先清理相机和机械臂，再有限等待收尾播报：正常结束最多 8 秒，停止 / 故障最多 3 秒，随后回收语音进程。已经交给系统播放的短句可能继续读完。

本机已有 `spd-say`。以下命令只测试扬声器，不连接机械臂：

```bash
spd-say --wait --language en-US 'Piper collection voice test.'
spd-say --wait --language cmn '采集语音播报测试。'
```

听不到声音时，先检查桌面的音量和输出设备是否选中了实际使用的扬声器 / 耳机。其他机器需要安装系统包 `speech-dispatcher`；若找不到命令、播放失败或单句超过 8 秒，程序只警告一次并关闭本次语音，采集、评分和保存继续运行。语音目前接在本地交互入口，原生分布式在线环境和离线 warmup 不播报。

**7. 流式视频与左键撤销**

原命令默认启用流式视频，上面已显式写入 `--streaming-video`。三路 RGB 在采集时分别由后台线程直接编码成 H.264，省去逐帧 PNG 的写盘和回读；评分后仍需要写入终止观测、排空短队列和保存 LeRobot 元数据。视频与动作保持逐帧对应，包含最后的无动作终止帧。每条 `result.json` 的 `save_duration_s` 记录评分后的实际保存耗时，`recording.video_encoding` 标记使用 `streaming` 还是 `png`。若要对照旧流程，把参数换成 `--no-streaming-video`。

运行中、人工接管中或达到步数上限等待评分时，都可以直接按 `←` 停止并丢弃当前回合，无需先按 `s/f`。收到左键后停止继续下发本回合动作，取消录制，跳过终止帧补拍和完整视频保存，播报“已丢弃”。若同一批输入里还有 `s/f`、接管键或右键，以撤销为准；同时按 `q` 则撤销后退出。录制资源释放后按原设置复位，再进入下一条等待界面。

评分后的保存、复位和等待阶段仍可用 `←` 撤销刚结束的回合。编码线程与文件释放后，整条尝试移到 `discarded/episode_XXX/`，同步从评分、成功率、自主成功率、接管比例和训练清单中排除。未评分撤销的回合只保留排查所需的已有文件，不生成可训练的完整回合。

撤销后恢复场景，再按一次 `→` 重采；不会自动启动动作。`--num-episodes` 表示要保留的已评分回合数，撤销的回合不占名额。目录编号按实际尝试递增，例如撤销 `episode_002` 后，新数据写入 `episode_003`，终端仍显示正在补采第 2 条。连续按左键只撤销最近这一条，不会继续删除更早的有效数据；正在执行新回合时左键只丢弃当前回合。

最后一条保存并复位后，程序等待确认：`→` 完成采集并退出，`←` 丢弃最后一条并等待重采，`q` 退出并保留未撤销的数据。保存和复位中按下的右键不会提前触发下一条，需要等待就绪后重新按。

每路编码队列最多 32 帧，CPU 编码使用 `veryfast`、每路 2 个编码线程。队列持续拥堵超过 1 秒或编码失败会停止本次采集并报告错误，避免丢帧后污染动作与 critical 标签的对应关系。当前环境支持所需的 LeRobot 流式接口；旧版环境若提示缺少该接口，可升级 LeRobot，或用 `--no-streaming-video` 继续原采集方式。训练数据继续从结束采集后的 `annotations.json` 导入。

**8. 中断后续采或增加目标条数**

仅新协议 `critical-segment-v1` 支持续采，旧协议目录须保留并另开新目录；入口在模型加载前拒绝旧 run.json。用 `--resume` 指向新协议的整次采集目录，自动恢复模型路径、任务、主臂接管、critical 标注规则、步数上限、录制和复位配置。纯 SFT 与 HIL 共用这个入口。下面用已有 HIL 目录示例；如要接着另一批数据采，将 `piper_run_dir` 改成对应目录。

```bash
cd /home/descfly/code/rlinf_piper
export PIPER_PYTHON="/home/descfly/code/rlinf_piper/.venv-piper/bin/python"
piper_run_dir="/home/descfly/code/rlinf_piper/outputs/piper/segment_scoring_collection"

bash toolkits/piper/eval_stage1.sh rollout \
  --resume "$piper_run_dir" \
  --num-episodes 50
```

`--num-episodes 100` 表示累计保留 100 条，包含原目录中已有的完整回合；已有 60 条时再采 40 条。省略此参数则沿用上次的目标条数。已经达到目标时进入最后一条确认界面，可按 `→` 结束或 `←` 撤销重采；要继续增加数据就把目标调大。

续采会读取原目录的 `run.json` 和 `hardware.yaml` 快照，所以原先的 `--teleop`、`--record-dataset`、`--reset-after-episode` 等参数自动生效。命令保留 `--resume` 和需要的覆盖项即可，不再传 `--output-dir`。可以覆盖累计目标、`--device`、语音开关 / 语言、流式编码开关。旧目录未记录编码开关时默认启用流式编码；需要 PNG 流程时加 `--no-streaming-video`。更换模型、任务或采集配置时使用新的采集目录。

已有完整回合的评分、critical 区间、接管记录和视频继续保留，成功率和训练清单按整个目录更新。撤销或未评分、未完整保存的尝试保留在磁盘，但不计入目标，也不进入 `annotations.json`。续采从下一条新回合开始，目录编号超过所有已有尝试编号，包括 `discarded/` 和未完成目录，避免覆盖。最后一次尝试若完整保留，开始新回合前仍可按 `←` 撤销；若最后一次已经撤销或未完成，左键不会追溯删除更早回合。

`initial` 始终使用第一次采集的 `reset_pose.json`，重启时不会把当前姿态替换为原复位点。如果上次在回合、保存或复位途中退出，且启用了自动复位，先确认工作区并按 `r` 返回原复位点，再恢复场景、按 `→` 开始。上次复位已经完成时直接等待 `→`。这两种状态均有终端和语音提示。

启动时检查已完成录制的动作表和视频帧数；已完成数据缺文件、元数据损坏或复位目标缺失时会报错，修复后再续采。CLI 对目录持有进程锁，防止两个新版本采集进程同时写入；退出后锁自动释放，无须删除 `.collection.lock`。原 `run.json`、`hardware.yaml` 和 `policy.json` 保留，每次续采的启动快照另存于 `resumes/`。采集全部确认后再执行第 4 节；已有 warmup 清单和 cache 不会随续采自动扩充，新增数据需使用新的清单、cache 和训练输出路径。

**9. 共享 actor 在本机评测（2026-10-21）**

已拉取分支 `piper/warmup-shared-20261021` 的部署提交 `6378dc4686fdb28cdb305366a11be47277838b3d`。阿里云原始说明和压缩包保存在 `/home/descfly/deploy/piper_20261021/`；小网络解压到 `checkpoints/piper_shared_prejoint/actor/`，Stage1 继续使用下方的本地 50k 权重。压缩包、包内文件、完整 Stage1 权重及归一化文件均已核对 SHA256。部署包记录 BC 5,000 步、critic 10,000 步、joint 0 步，原始 readiness 保存在 `checkpoints/piper_shared_prejoint/original_readiness.json`。

以下设置供离线推理和真机评测共用；`PIPER_DATA_PATH` 是已有录像，只用于不连接硬件的推理检查。

```bash
cd /home/descfly/code/rlinf_piper

export PIPER_PYTHON="$PWD/.venv-piper/bin/python"
export PIPER_STAGE1_PATH="$PWD/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"
export PIPER_ACTOR_PATH="$PWD/checkpoints/piper_shared_prejoint/actor"
export PIPER_DATA_PATH="$PWD/data/rlinf_sft50k_hil_20261018_143541/episode_001/dataset"

bash toolkits/piper/eval_shared.sh infer-reference --frame 300
bash toolkits/piper/eval_shared.sh infer-actor --frame 300
```

本机默认使用 TorchCodec 解码；旧 `pyav` 路径依赖的 `torchvision.io.VideoReader` 在当前环境已不存在。2026-10-21 在 RTX 5090 上读取上述第 300 帧，两路完整 GPU 推理均通过：reference 为 `50×7`、约 0.50 秒，actor 为 `10×7`、约 0.42 秒，输出均为有限值。这是单帧加载与推理检查，实际循环频率和成功率需由 rollout 测量。249 项 Piper 回归通过；结果及日志位于 `/home/descfly/deploy/piper_20261021/checks/`。

准备好场景后，在前台终端启动一条带主臂接管的评测。脚本默认录制、流式编码，保留动作范围/步幅检查；下面选择手动复位。需要自动回到第一条起点时，将 `none` 改为 `initial`。

```bash
cd /home/descfly/code/rlinf_piper
piper_run_dir="$PWD/outputs/piper/shared_prejoint_$(date +%Y%m%d_%H%M%S)"

bash toolkits/piper/eval_shared.sh rollout \
  --teleop piper_leader \
  --num-episodes 1 \
  --steps 3600 \
  --replan-steps 50 \
  --speech-language zh \
  --reset-after-episode none \
  --output-dir "$piper_run_dir"
```

| 操作 | 共享 actor 评测中的行为 |
| --- | --- |
| 就绪时 `→` | 开始整条 episode，先执行 H50 reference。 |
| `b` | 开始关键段并切到共享 C10 actor；段内重复 `b` 不结束该段。 |
| 段内 `s/f` | 独立评分该段并切回 reference，整条任务继续运行。 |
| 运行中 `→` | 停止下发新动作，进入整条任务的评分流程。 |
| `i` | 人工接管 / 交还，接管优先于 actor/reference；critical 标注保持独立。 |
| `←` | 直接丢弃当前 episode；评分后的保存、复位、等待阶段也可撤销。 |

按 `→` 停止时若没有未评分段，另按一次 `s/f` 评分整条；若还有未评分段，则先 `s/f` 补评段，再 `→` 进入整体评分，最后 `s/f`。达到步数上限时也需要右键确认整体评分。每段保存 `[start, end)` 和独立的 `outcome`，整条 `outcome` 另存；段评分不会推断整条结果。切换会清空旧 chunk、拒绝过期异步预测，人工交还等到新命令真正下发才结束人工归属。保存与已配置的复位完成后，右键开始下一条或确认全部结束。

共享 actor 的续采使用公共入口，从原目录恢复权重身份、接管和采集设置。例如将累计保留目标增加到 5 条：

```bash
cd /home/descfly/code/rlinf_piper
# 换终端后，将 piper_run_dir 设为上次实际输出目录。
USE_TF=0 JAX_PLATFORMS=cpu "$PIPER_PYTHON" -m toolkits.piper.run_policy rollout \
  --resume "$piper_run_dir" \
  --num-episodes 5
```

不能将旧版 v2 数采目录续采成 v3，也不能在同一目录替换 actor 权重；更换协议或模型时新建输出目录。续采直接调用上述公共入口，`eval_shared.sh` 的新建运行默认参数不适用于 `--resume`。

`eval_shared.sh` 执行共享 actor 的手动切换评测和数据采集，不启动在线更新。在线训练使用下面的独立配置，actor horizon 从权重读取并沿用逐段评分。原 readiness 仍为 false；脚本的实验选项不会改写报告。二分类 phase monitor 目前只提供建议状态，未参与自动切换。

## 10. 原生 RLT online 训练与续训

`train_online.sh` 在每条 episode 确认保留后提交后台，下一条可立即开始；原生 RLT actor/critic 独立更新，rollout 在下一条 episode 开始前切换完整快照，整条执行期间固定版本。完整 C50 启动命令和已下载的 critic 权重见 [ONLINE_C50.md](ONLINE_C50.md)。Stage1 保持冻结。新运行必须显式指定已经完成 critic warmup 的 actor，以及与其 Stage1、任务、动作维度和 horizon 一致的离线 cache。新下载的 C50 prejoint 包已完成 critic TD10000；第 13 节旧 BC5000 包仍只用于固定权重评测。以下命令使用已下载的权重和配套离线 cache，不读取 C10 rollout 旧数据：

```bash
cd /home/descfly/code/rlinf_piper
export PIPER_PYTHON="$PWD/.venv-piper/bin/python"
export PIPER_STAGE1_PATH="$PWD/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"
export PIPER_ACTOR_PATH="$PWD/checkpoints/piper_shared_c50_prejoint_20261022/actor"
export PIPER_OFFLINE_CACHE="$PWD/outputs/piper/offline_cache/cache_local_segments_c50_s10_20261022"
piper_online_dir="/media/descfly/新加卷/value/data_inference/rlinf_online_c50_$(date +%Y%m%d_%H%M%S)"

bash toolkits/piper/train_online.sh \
  --teleop piper_leader --num-episodes 5 \
  --output-dir "$piper_online_dir"
```

也可直接传 `--offline-cache /path/to/cache`，优先于环境变量。脚本默认录制数据、`--steps 7200 --replan-steps 50 --reset-after-episode initial`。actor horizon 从 metadata 读取，C50 在关键段执行完整 50 步；按键和人工接管仍可随时中断队列。online 入口拒绝把实际执行长度设得比 actor horizon 更短。

需要从新 C50 数据准备离线池时，使用独立配置 `examples/offline_rl/config/piper_rlt_c50_warmup.yaml`，默认 `chunk_length=50 stride=2 tail_policy=align`，输入标签和输出均在 `outputs/piper/c50/`。已有 v3 分段标签时可运行 `"$PIPER_PYTHON" -m toolkits.piper.warmup cache --config examples/offline_rl/config/piper_rlt_c50_warmup.yaml annotations=/path/to/c50_annotations.json`。该步骤只准备 cache，不会训练或调用机器人；现成的相容 C50 warmup cache 可直接传给 online。

操作为 `→` 开始、`b` 进入关键段、`s/f` 结束并评分该段并恢复 VLA、`i` 人工接管。运行中 `→` 停止整条，再单独 `s/f` 评分。`←` 在执行、评分、编码、复位或等待下一条时丢弃当前或上一条。**保存和复位结束后的 `→` 确认保留并启动训练，此后不能撤销该条；最后一条也必须确认。** `q` 退出前未确认的数据保持可撤销，续训时再确认。

配置集中在 `examples/embodiment/config/piper_rlt_online.yaml`：

| 项目 | 当前默认 |
|---|---|
| replay 窗口 | 长度等于 actor horizon，stride=2，与执行 chunk 独立 |
| 段末 | 补一个对齐段末的完整窗口；整段不足 C 时只保留一个带 mask 的短窗口 |
| 在线池 | 最近 200 条有效 episode 内均匀、有放回采样；原始数据持续保留 |
| 离线池 | 导入 warmup train split 全部数据，不受在线 200 条窗口限制；validation 不入池 |
| batch | 固定 64：32 online + 32 offline；少量数据通过有放回采样保持比例 |
| 人工接管 | 实际执行动作和逐帧 human mask 进入在线池，BC 对应帧模仿人工动作 |
| 开始更新 | 至少 64 个在线 transition；按 transition 数判断，不按 episode 条数 |
| 训练预算 | 每个新增在线 transition 产生 5 次 critic 更新预算；每轮完成全部预算后保存一次；每 32 次检查退出，正常退出时保存进度与余量 |
| actor 频率 | 每 2 次 critic 更新一次；version 仍是累计 critic 更新数 |
| 学习率 / target | actor、critic 均为 3e-4；每步软更新 target Q，tau=0.005 |

UTD 在本入口明确按“新增 replay transition”计数，不按帧数、batch 样本曝光次数或离线导入数计数。上限可能令实际更新数低于预算，查看 `rlt/pending_update_budget`，不会悄悄丢弃余量。Q 权重在前 1,000 次 critic 更新中由近 0 升至 0.1，BC 权重从 5 降至 2；这是一份可调训练配置，不代表任务成功率已经通过验证。

每段独立终止，成功段最后有效 tick 奖励为 1，失败为 0，`gamma=0.995` 按实际 tick 折扣。窗口不跨 critical phase 边界；整条 episode 的评分不覆盖分段奖励。保存的是实际发出的连续动作，包括 human 和 hold tick，而不是把网络预测的未执行尾部当作动作。

推理过程中会在 `replay_anchors/` 保存请求帧的 normalized latent、proprio 和完整 H50 reference。确认后复用这些特征，对缺失的起点、下一状态去重，并按默认 batch=2 补算，再保存 `online_features.npz`。`online_features.json` 记录复用数量、补算数量和耗时；不需要每次梯度更新都运行 VLA。增密窗口会增加编码开销；后台按小 batch 补算，等待中的推理优先，但正在执行的 forward 无法抢占。单 GPU 的训练也会竞争算力。最后一条确认后等待后台完成，期间 q 可退出并保留余量。

续训恢复模型、target、optimizer、两个 replay 池、累计新增数据量、训练计数和随机数状态：

```bash
bash toolkits/piper/train_online.sh --resume "$piper_online_dir" --num-episodes 100
```

`--num-episodes` 是累计目标。续训使用运行目录内的 `online.yaml` 和硬件快照，不替换模型或采样配置；新的 C50 运行应使用新目录。离线数据已包含在 learner checkpoint 中，续训不再依赖原离线 cache 目录。确认后中断会从最后完整 checkpoint 重做；已提交的数据不会重复入池。

`online_update.json` 的 `round_critic_updates_run`、`round_actor_updates_run` 记录这轮累计更新数，`rlt/critic_updates_run`、`rlt/actor_updates_run` 仅表示最后一个小批次；另有剩余预算、两个池大小和新权重版本。`online/latest.json` 指向完整 checkpoint；默认保留最近 3 份，及续训进程正在使用的旧 checkpoint。原始录像和 replay 文件持续保留。无硬件 GPU 复测说明见 `docs/piper_online_training.rst`。

## 11. 记录 C10 对应的完整 H50 reference

排查关键段内不松手时，在 `train_online.sh` 或 `eval_shared.sh rollout` 的命令后加 `--save-rlt-reference`。默认关闭；开启后，每次预测将**同一次 VLA 前向生成、实际提供给 actor 的完整 H50 reference**写入该条 `episode_XXX/trace.jsonl`，不会另做一次随机采样。关键段外也记录 reference，便于比较切换前后。

以下命令使用固定的共享 warmup actor 做诊断，episode 之间不训练。`b`、`s/f`、`→`、`←` 和人工接管按键沿用第 10 节的规则：

```bash
cd /home/descfly/code/rlinf_piper
export PIPER_PYTHON="$PWD/.venv-piper/bin/python"
export PIPER_STAGE1_PATH="$PWD/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"
export PIPER_ACTOR_PATH="$PWD/checkpoints/piper_shared_prejoint/actor"
piper_reference_dir="/media/descfly/新加卷/value/data_inference/rlinf_reference_$(date +%Y%m%d_%H%M%S)"

bash toolkits/piper/eval_shared.sh rollout \
  --teleop piper_leader --num-episodes 2 --steps 7200 \
  --no-action-limits --reset-after-episode initial \
  --save-rlt-reference \
  --output-dir "$piper_reference_dir"
```

若要观察 online 更新后的 actor，把第 10 节的新开训练命令加上 `--save-rlt-reference` 即可。也允许续训时单独启用：

```bash
bash toolkits/piper/train_online.sh \
  --resume "$piper_online_dir" --num-episodes 100 --save-rlt-reference
```

`--no-save-rlt-reference` 可在续训时关闭记录；已经采完的旧 episode 无法补出当时的随机 reference。开关只增加诊断数据传输和 JSON 写入，不改变 H50/C10 执行方式、actor 采样或训练行为。

每个 `event="prediction"` 日志内：

- `actions`：实际推理的 actor C10（关键段内）或 reference H50（关键段外）。
- `rlt_reference.reference_actions`：与上述预测配对的完整 H50，每行按 J1–J6、夹爪排列，单位为度、毫米；不是已执行轨迹。当前 actor 只读取它的前 10 步。
- `rlt_reference.observation_state`：该次推理输入的关节角和夹爪反馈。`mode` 区分固定 actor 的 `eval` 和 online 探索的 `train`；online 还记录 `policy_version`。
- `step`：输入观测对应的执行步号。遥操模式下 `accepted_at_step` 是结果开始可用的步号，`discarded=true` 表示预测已因接管或路由变化作废，不代表执行过。
- `rlt_reference.inference_call_s`：推理调用本身耗时（online 包含 worker 通信与诊断数据返回）；原 `inference_s` 在遥操模式下还包含控制循环发现结果的等待时间。

同文件内 `event="action"` 的 `target`、`command`、`state` 分别表示请求目标、实际发送目标和反馈。先比较 reference 前 10 步与后 40 步的夹爪目标，再对照 actor C10 和实际反馈，才能区分延后松手、actor 输出偏差和执行问题。录像通过数据集帧号与 `step` 对齐；如需保留精确的推理输入 RGB，可另加 `--save-observations`，但会增加磁盘占用。

## 12. 纯 VLA：H50 规划、C10 执行对照

这组对照直接使用 Stage1 每次生成的 H50 reference，只执行前 10 步就重新观测、推理。全程不加载 RLT actor、不训练；保留同样的 RLT feature 提取、遥操和控制循环，以便比较短片段执行本身的影响。已有 `PIPER_ACTOR_PATH` 环境变量不会被 `eval_stage1.sh` 自动加载。

```bash
cd /home/descfly/code/rlinf_piper
export PIPER_PYTHON="$PWD/.venv-piper/bin/python"
export PIPER_STAGE1_PATH="$PWD/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"
piper_vla_c10_dir="/media/descfly/新加卷/value/data_inference/rlinf_vla_c10_$(date +%Y%m%d_%H%M%S)"

bash toolkits/piper/eval_stage1.sh rollout \
  --teleop piper_leader \
  --num-episodes 2 --steps 7200 --replan-steps 10 \
  --record-dataset --save-rlt-reference \
  --no-action-limits --reset-after-episode initial \
  --output-dir "$piper_vla_c10_dir"
```

这条命令从抓取到插入、松手都是 C10。`b` 开始关键段标注，`s/f` 结束并评分该段；这些按键不会切换 actor，也不会恢复 C50。`→` 停止整条后再 `s/f` 评分，`←` 丢弃，`i` 遥操接管，`q` 退出。为比较自主动作，使用遥操或用手帮助插入的时刻应注明。

每次 `prediction.actions` 和 `rlt_reference.reference_actions` 都保留完整 H50，`replan_steps=10` 表示实际计划执行前 10 步；真实发送情况以同文件 `action` 事件为准。中途停止或接管时可能少于 10 步。这样既能检查后半段是否规划松手，也能确认 C10 重规划时松手是否持续留在未来。

对照纯 VLA C50 时，新建一个输出目录，把 `--replan-steps 10` 改为 `--replan-steps 50`，其余参数保持一致。C10/C50 应分别重复几条、尽量保持初始场景一致；全程 C10 可能改变到达孔口时的状态，不能把一次前后对比视为相同初始条件。

当前 warmup 的 BC 目标：非人工接管步模仿 cached reference 的前 10 步；人工接管步模仿记录的人工动作。启用联合或 online actor 更新后，还叠加 Q 目标。因此纯 VLA C10 能检验 teacher 在这种执行节奏下是否本身就会停滞，也能帮助区分 actor 的模仿误差。

## 13. 共享 C50 actor：BC5000 执行对照

云端新包 `piper_shared_c50_bc5000_20261022` 已下载到本地 `checkpoints/`，包内 SHA256 清单全部通过。它仍使用同一份 Stage1 H50，但 actor 现在读取完整 H50 reference，输出 50×7；本地现有代码按 metadata 加载，无需回退代码。新 actor 已做 5,000 次 BC，critic 与联合更新均为 0，**只用于 `eval_c50.sh` 固定权重执行对照，不用于 `train_online.sh`、Q 评估或联合训练**。

云端部署说明和训练收据保存在 `/home/descfly/deploy/piper_20261022/c50_bc5000/`；权重包内 `README.md`、`EXPERIMENT.md` 和检查 JSON 也保留原样。`readiness.json=false` 表示完整 actor/critic warmup 门槛尚未满足，实验入口仍显式使用 `--allow-offline-actor`。

```bash
cd /home/descfly/code/rlinf_piper
export PIPER_PYTHON="$PWD/.venv-piper/bin/python"
export PIPER_STAGE1_PATH="$PWD/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"
export PIPER_C50_ACTOR_PATH="$PWD/checkpoints/piper_shared_c50_bc5000_20261022/actor"
piper_c50_dir="/media/descfly/新加卷/value/data_inference/rlinf_shared_c50_bc5000_$(date +%Y%m%d_%H%M%S)"

bash toolkits/piper/eval_c50.sh rollout \
  --teleop piper_leader \
  --num-episodes 2 --steps 7200 \
  --record-dataset --save-rlt-reference \
  --reset-after-episode initial \
  --output-dir "$piper_c50_dir"
```

`eval_c50.sh` 默认加载本节 C50 BC5000 包，独立于终端中旧的 `PIPER_ACTOR_PATH`；更换 C50 包时设置 `PIPER_C50_ACTOR_PATH`。入口校验 C50/H50，并固定 `replan_steps=50`。普通阶段执行 VLA H50；`b` 开始关键段并切换共享 C50 actor，`s/f` 结束并评分该段、丢弃未执行的 actor 队列并恢复 VLA。`→` 停止整条后另行 `s/f` 评分；`←` 丢弃，`i` 人工接管。C50 也可中途被按键或接管打断，不需要等完整 50 步走完。

专用入口固定 `--replan-steps 50`；若直接使用通用 `eval_shared.sh`，也需保持 50。日志中的关键段 `prediction.actions` 应为 50×7，`rlt_reference.reference_actions` 同样为 50×7，`replan_steps=50`。本次 actor 固定不更新，便于对比新 C50 是否能够推进微调和松手阶段。

这份权重尚未通过全部动作检查。云端收据记录过某个实录输入的首目标 J4 与观测相差 8.761°，超过当前 5° 步幅门限；长动作尾部的人工监督误差也较大。以上命令保留当前动作检查，未附加历史命令中的 `--no-action-limits`。若触发检查，保留该条 trace 和报错用于定位，不能把“模型能加载、输出50步”当作动作验收或任务成功率已通过。

只读取录像的加载检查可用以下命令，不连接机械臂：

```bash
export PIPER_DATA_PATH="/path/to/new_c50_run/episode_001/dataset"
bash toolkits/piper/eval_c50.sh infer-actor --frame 0
```

此前部署验收曾完成第 2350 帧的真实 GPU 推理：输出 `(50,7)`、全为有限值，Stage1 ID、norm 与小网络权重 SHA256 全部匹配。该次 C50 actor 在第 34 步给出超过 20 mm 的张开目标，最大 41.73 mm；首目标最大关节偏差为 3.91°。这只说明该录制观测下的输出与契约，不代表整条轨迹或全部动作检查已通过。收据位于 `outputs/piper/c50_bc5000_infer_20261022/local_validation.json`，没有连接机械臂。

## 2026-10-23 实验备份与清理

最近的 RLinf C10/C50 对照、SFT/遥操数据、online 训练记录与震荡分析已备份至阿里云。压缩包为 1,320,592,012 bytes（1.32 GB），包含 3.28 GB 原始文件；云端已核验整个压缩包的 SHA256，以及包内 2,445 个文件或链接的内容/目标。备份目录：

```text
/home/yiming/workspace/code/rlinf_piper_warmup_20261019/deployment/local_backups/rlinf_local_20261023_162042
```

本地已清理 2.92 GB 旧实验、重复 checkpoint 和旧合成验证的训练状态。保留 `rlinf_online_c50_20261023_145843` 的录像、动作、replay、日志和 `online/checkpoint_000075`；部署权重、C50 离线 cache、分析结果也保留。新实验继续按本文命令使用新目录。该历史运行最新 actor 的震荡问题尚未解决，保留 checkpoint 用于诊断和恢复，不代表推荐从它开始新实验。

本地清单、云端校验结果和具体删除记录位于 `/home/descfly/code/rlinf_piper/outputs/backups/rlinf_local_20261023_162042/` 下的 `manifest.json`、`VERIFIED.json`、`cleanup_receipt.json`。本地临时备份包和上传分片已移除，云端保留完整压缩包。以下命令在云端校验并解压到独立目录：

```bash
ssh yiming-aliserver
cd /home/yiming/workspace/code/rlinf_piper_warmup_20261019/deployment/local_backups/rlinf_local_20261023_162042
sha256sum -c SHA256SUMS
piper_restore_dir="$HOME/piper_restore_20261023"
mkdir -p "$piper_restore_dir"
zstd -dc --long=29 piper_local_experiments.tar.zst | tar -xf - -C "$piper_restore_dir"
```

解压后的 `data_inference/` 对应本地 `/media/descfly/新加卷/value/data_inference/`，`workspace_outputs/piper/` 对应 `/home/descfly/code/rlinf_piper/outputs/piper/`。每个文件的原始绝对路径在 `manifest.json` 中。

## 2026-10-23 reference dropout：C30 / C50 新权重

这一组用于比较 reference dropout 后的 actor，并可继续做 online 更新。两版均完成 BC5000 + critic TD10000，联合更新为 0，部署包 SHA256 已与云端核对，包内校验清单全部通过。原权重保留，新包位于 `checkpoints/piper_reference_dropout_c30_c50_20261023/`。

[RL Token 论文附录 B](https://arxiv.org/html/2604.23073v1#A2) 明确规定：训练时遮挡 50% reference，推理始终提供 reference。[openpi-RLT 的 actor 更新](https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/src/rlt_online_rl/trainer.py#L162) 和 [遮挡函数](https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/src/rlt_online_rl/networks.py#L193) 采用按样本独立抽样、整 chunk 置零、保留样本不缩放的实现；这是第三方复现。BC 监督目标仍保留，critic bootstrap 使用完整 reference。

本地新建 online run 默认 `reference_dropout_prob=0.5`，从第一次 actor 更新开始，对 online/offline 混合 batch 同样生效。只遮 actor 的 reference 输入；RL token、proprio、监督目标、valid mask、replay/cache 原数据均保留。人工 tick 先按现有逻辑替换训练 reference 和 BC 目标，再遮 actor 输入。实际 rollout、固定权重评测和 critic 下一状态估值不做遮挡；actor 的 Gaussian 探索采样是另一项机制。

选择 C50 或 C30 后，执行以下公共设置。两者普通阶段均执行 VLA H50；关键段分别执行 actor C50 或 C30，保持 `--replan-steps 50`。C30 actor 读取 H50 的前 30 步，C50 读取全部 50 步。两套离线 cache 各有 3,399 个训练 transition 和 518 个验证 transition，已校验全部 shard；只将 train split 入池。

```bash
cd /home/descfly/code/rlinf_piper
piper_variant=c50  # 测 C30 时改为 c30
export PIPER_PYTHON="$PWD/.venv-piper/bin/python"
export PIPER_STAGE1_PATH="$PWD/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"
export PIPER_ACTOR_PATH="$PWD/checkpoints/piper_reference_dropout_c30_c50_20261023/$piper_variant/actor"
case "$piper_variant" in
  c30) export PIPER_OFFLINE_CACHE="$PWD/outputs/piper/offline_cache/cache_local_segments_c30_s10_20261023" ;;
  c50) export PIPER_OFFLINE_CACHE="$PWD/outputs/piper/offline_cache/cache_local_segments_c50_s10_20261022" ;;
esac
```

仅评测离线权重，用以下命令；该进程不更新网络：

```bash
piper_eval_dir="/media/descfly/新加卷/value/data_inference/rlinf_${piper_variant}_refdrop_eval_$(date +%Y%m%d_%H%M%S)"
bash toolkits/piper/eval_shared.sh rollout \
  --teleop piper_leader --num-episodes 5 \
  --steps 7200 --replan-steps 50 \
  --record-dataset --save-rlt-reference \
  --reset-after-episode initial --output-dir "$piper_eval_dir"
```

要边采集边训练，退出评测进程后，用新目录启动：

```bash
piper_online_dir="/media/descfly/新加卷/value/data_inference/rlinf_${piper_variant}_refdrop_online_$(date +%Y%m%d_%H%M%S)"
bash toolkits/piper/train_online.sh \
  --teleop piper_leader --num-episodes 5 \
  --steps 7200 --replan-steps 50 --reference-dropout 0.5 \
  --record-dataset --save-rlt-reference \
  --reset-after-episode initial --output-dir "$piper_online_dir"
```

使用 `--reference-dropout 0` 可做关闭 online 遮挡的对照；它不改变所加载权重的离线训练历史。参数仅接受有限的 `[0, 1]` 概率，仅适用于 online 训练。`online.yaml` 保存实际配置，训练指标 `actor/reference_dropout_prob` 记录生效概率。`--resume` 沿用原配置，不允许用该参数更改已有运行；历史 dropout=0 的 run 仍为 0。每条 episode 固定一个 checkpoint，后台每轮完成全部预算后保存的规则继续生效。

```bash
bash toolkits/piper/train_online.sh --resume "$piper_online_dir" --num-episodes 100
```

云端结果说明位于包内 `REFERENCE_DROPOUT_C30_C50.md`。dropout 明显改善了缺失 reference 的离线表现，但完整 reference 下的验证 RMSE 没有改善，尚不能判断真机效果更好。包内 readiness 仍未通过全部门槛；本地已完成 C30/C50 的 GPU 推理与更新、dropout 随机状态恢复和 C50 异步续训验证，尚未执行真机任务；收据见 `outputs/piper/refdrop_deployment_20261023/validation.json`。动作约束按现有配置执行。离线训练补丁无需应用即可加载这些权重；本次启用的是本地已有的 online actor 遮挡路径。
