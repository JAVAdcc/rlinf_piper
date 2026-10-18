**RLinf critical phase 数据采集与离线 warmup（2026-10-18）**

沿用前面的 SFT 验收流程：global_step_50000 模型自动执行，人工按 `b` 标记关键阶段、按 `s/f` 评分，数据用于离线 actor-critic warmup。critical phase 指轨迹中的关键动作区间。第 1 节是纯 SFT 验收；需要主臂人工接管时，使用第 5 节的命令，入口和标注方式相同。

**1. 启动采集**

在本机前台终端执行整段命令。先测试 20 条，正式采集时将 `piper_episodes` 改为 `100`。使用合并后的 `rlinf_piper`，其中已经包含多段 `b` 标注与奇数次自动补终点的改动。

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

脚本固定加载 H50 FP32 Stage 1 权重、三路相机和双 USB 任务提示词，名义控制频率为 30 Hz，每次执行 50 个目标后重新推理；每条最多执行 3,600 个目标。每次启动使用新的输出目录，程序自动创建。采集前先退出占用同一机械臂或相机的其他程序。

`--no-action-limits` 和 `--reset-after-episode initial` 沿用原文件设置：前者关闭额外的 rollout 动作范围/步幅检查，驱动行程限制仍保留；后者在每条评分并保存后，张开夹爪、返回第一条开始时的姿态，再恢复夹爪位置。若本次希望手动复位，将后者改为 `--reset-after-episode none`。

复位在 `s/f` 评分、数据和视频保存完成后开始，首次启动程序不会先复位。`initial` 会将第一条开始时的关节/夹爪读数限制到驱动的合法目标范围，并在 `reset_pose.json` 中同时记录原始读数和实际复位目标，避免零位附近的小幅反馈偏差使复位被拒绝。若要每回合返回零位，使用 `--reset-after-episode zero`。

**2. 按键与标注**

模型加载、相机连接和从臂使能完成后，程序等待右方向键。保持终端焦点，按键无需回车。

| 按键 | 行为 |
| --- | --- |
| `→` | 恢复场景后开始当前回合。 |
| `b` | 第一次进入 critical phase，第二次退出；第三、四次标记下一段，可以继续添加。 |
| `s` / `f` | 停止下发动作，分别记为成功 / 失败并保存。 |
| `←` | 在评分后的保存、复位或等待阶段，丢弃刚结束的回合；恢复场景后按 `→` 重采。 |
| `q` / `Ctrl-C` | 退出；此前完成的回合保留，当前未评分回合不进入训练标注清单。 |

如果结束时 `b` 为奇数次，按 `s/f` 会自动将最后一段延长到终止帧，前面已配对的段保持原边界。完全没按 `b` 的回合仍保存录像和评分，但需要补标才能用于 warmup。刚按 `b` 就评分、尚未执行动作的空段会被忽略。

全回合由 SFT 模型执行，`b` 只改变标签。达到步数上限后程序停止下发动作并等待 `s/f`。视频默认随采集流式编码，评分后只需收尾并保存元数据；等待保存和复位完成，再恢复场景并按 `→`。采够指定条数后，最后一条仍可按 `←` 撤销，按 `→` 确认结束。

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
- `discarded/`：左键撤销的采集尝试，保留原始评分和撤销时间供检查；这些目录不进入 `annotations.json`，也不计入成功率。

`start` 包含该帧动作，`end` 不包含；最后一段自动延长时，`end` 等于 `terminal_frame`。终止帧只有观测，`action_valid=False`，不会作为训练动作。

**4. 数据确认后运行离线 warmup**

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

piper_run_dir="/media/descfly/新加卷/value/data_inference/rlinf_sft50k_hil_$(date +%Y%m%d_%H%M%S)"
export PIPER_PYTHON="/home/descfly/code/rlinf_piper/.venv-piper/bin/python"
export PIPER_STAGE1_PATH="/home/descfly/code/rlinf_piper/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"

bash toolkits/piper/eval_stage1.sh rollout \
  --teleop piper_leader \
  --num-episodes 2 \
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
| `b` | 独立进入 / 退出 critical phase；接管本身不自动更改 critical 标签。 |
| `s/f` | 停止、评分、保存，再复位；复位时主臂跟随，复位帧不录入训练数据。 |
| `←` | 保存中、复位中或下一条开始前撤销刚结束的回合，随后按 `→` 重采。 |
| `q` / `Ctrl-C` | 退出，已完成回合保留。 |

若习惯 Evo-RL 的 `o` 键，可加 `--intervention-key o`。交还后的等待保持仍记为人工段；普通推理等待记为 `hold`。采集线程在推理期间继续响应按键，但实际周期还受相机、CAN 和写盘影响，30 Hz 是名义配置。

采集后仍用第 3、4 节检查并生成 warmup 清单。数据会保存逐帧 `is_intervention`、`critical_phase`、实际下发目标与请求目标；`action_source` 的编码为 `0=reference、1=actor、2=human、3=hold`，另存于 `dataset/meta/piper_control.json`。`policy_action_valid=False` 表示该帧没有有效 policy 输出，不能把占位零值当作预测。

`summary.json` 中 `success_rate` 是最终成功率，`autonomous_success_rate` 只把未接管的成功计入分子，分母仍为全部已评分回合；`intervention_rate` 是发生接管的回合比例。每回合另存 `takeovers` 和 `human_steps`。验收 SFT 自主效果时看自主成功率，避免把人工救回算成模型独立成功。

HIL 采集前不需要 latent cache，后续继续用 `warmup init -> cache -> train`。当前 critical-only warmup 只取标注区间，区间外的人工动作保留在原始数据中；混合回合不要整条改成 `behavior=human`。

原生在线 `PiperPolicyEnv` 已接入接管动作和标签回传，交还到新 chunk 才生效；它在分布式推理等待期间仍不能像本地入口一样连续响应接管。在线有效长度、跨阶段折扣和数据池恢复仍需后续接通，不能直接把 HIL 采集通过视为在线探索就绪。实现细节见 `/home/descfly/code/rlinf_piper/docs/piper_human_intervention_design.rst`。本次开发使用 CPU 和 mock 验证，尚未启动真机验收。

**6. 采集语音播报**

第 1、5 节的本地交互采集入口均已接入语音，默认开启英文播报，与 Evo-RL 使用同一个本地 Speech Dispatcher 后端。原有命令不加参数也会播报；上面的 `--play-sounds` 显式开启。需要中文时增加 `--speech-language zh`，关闭时把 `--play-sounds` 换成 `--no-play-sounds`。这些参数不影响数据、标注或动作来源，也不启动额外的模型服务。

| 操作 / 状态 | 播报内容与时机 |
| --- | --- |
| 等待 `→`、开始回合 | 提示回合编号和开始键；HIL 另提示正在对齐主臂。 |
| `b` 进入 / 退出 | 提示关键阶段编号；标签写入成功后播报。 |
| 奇数次 `b` 后评分 | 提示最后一段自动延长到回合结束；空段被忽略时另有提示。 |
| `i` 接管 | 接管成功后提示“人工接管”。 |
| 再按 `i` 交还 | 先提示“等待模型的新预测”；新的 policy 动作实际下发后才提示“模型控制已恢复”。 |
| 到达步数上限 | 提示等待 `s/f` 评分；此时再开新的 critical phase 会提示无效。 |
| `s/f` | 评分写入成功后提示该回合标记成功 / 失败。 |
| 保存数据 | 提示正在保存；数据、结果和标注清单写入成功后提示保存完成。漏标 critical phase 会单独提示。 |
| 自动复位 / `r` 重试 | 提示开始复位、轨迹执行完成或复位失败及重试键；“轨迹执行完成”仍需现场确认姿态。 |
| `←` / 最后一条确认 | 提示该回合已丢弃、可以重采，或提示右键结束 / 左键撤销。 |
| 完成、`q` / Ctrl-C、故障 | 分别提示采集完成、采集停止或故障停止。 |

播报在后台串行播放，不在控制循环中等待音频。快速连续操作时，尚未播放的同类状态提示会合并为最新状态，过期提示会丢弃，终端日志保留完整事件。正常结束会保留最后的评分、保存和复位结果；退出或故障则优先播报停止。程序先清理相机和机械臂，再有限等待收尾播报：正常结束最多 8 秒，停止 / 故障最多 3 秒，随后回收语音进程。已经交给系统播放的短句可能继续读完。

本机已有 `spd-say`。以下命令只测试扬声器，不连接机械臂：

```bash
spd-say --wait --language en-US 'Piper collection voice test.'
spd-say --wait --language cmn '采集语音播报测试。'
```

听不到声音时，先检查桌面的音量和输出设备是否选中了实际使用的扬声器 / 耳机。其他机器需要安装系统包 `speech-dispatcher`；若找不到命令、播放失败或单句超过 8 秒，程序只警告一次并关闭本次语音，采集、评分和保存继续运行。语音目前接在本地交互入口，原生分布式在线环境和离线 warmup 不播报。

**7. 流式视频与左键撤销**

原命令默认启用流式视频，上面已显式写入 `--streaming-video`。三路 RGB 在采集时分别由后台线程直接编码成 H.264，省去逐帧 PNG 的写盘和回读；评分后仍需要写入终止观测、排空短队列和保存 LeRobot 元数据。视频与动作保持逐帧对应，包含最后的无动作终止帧。每条 `result.json` 的 `save_duration_s` 记录评分后的实际保存耗时，`recording.video_encoding` 标记使用 `streaming` 还是 `png`。若要对照旧流程，把参数换成 `--no-streaming-video`。

按 `s/f` 后即可按 `←`，即使仍在视频收尾、保存元数据或自动复位。程序会立即把该回合标为丢弃，保存线程释放文件后将整个采集尝试移到 `discarded/episode_XXX/`。已经完成保存、正在等待下一条的回合也可以撤销；评分、成功率、自主成功率、接管比例和训练标注清单同步排除它。撤销会播报，复位仍按原配置完成。

撤销后恢复场景，再按一次 `→` 重采；不会自动启动动作。`--num-episodes` 表示要保留的已评分回合数，撤销的回合不占名额。目录编号按实际尝试递增，例如撤销 `episode_002` 后，新数据写入 `episode_003`，终端仍显示正在补采第 2 条。连续按左键只撤销最近这一条，不会继续删除更早的有效数据；正在执行新回合时左键不撤销上一条。

最后一条保存并复位后，程序等待确认：`→` 完成采集并退出，`←` 丢弃最后一条并等待重采，`q` 退出并保留未撤销的数据。保存和复位中按下的右键不会提前触发下一条，需要等待就绪后重新按。

每路编码队列最多 32 帧，CPU 编码使用 `veryfast`、每路 2 个编码线程。队列持续拥堵超过 1 秒或编码失败会停止本次采集并报告错误，避免丢帧后污染动作与 critical 标签的对应关系。当前环境支持所需的 LeRobot 流式接口；旧版环境若提示缺少该接口，可升级 LeRobot，或用 `--no-streaming-video` 继续原采集方式。训练数据继续从结束采集后的 `annotations.json` 导入。
