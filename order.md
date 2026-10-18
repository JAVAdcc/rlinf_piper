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
  --save-observations \
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
| `q` / `Ctrl-C` | 退出；此前完成的回合保留，当前未评分回合不进入训练标注清单。 |

如果结束时 `b` 为奇数次，按 `s/f` 会自动将最后一段延长到终止帧，前面已配对的段保持原边界。完全没按 `b` 的回合仍保存录像和评分，但需要补标才能用于 warmup。刚按 `b` 就评分、尚未执行动作的空段会被忽略。

全回合由 SFT 模型执行，`b` 只改变标签。达到步数上限后程序停止下发动作并等待 `s/f`。评分后会编码视频；等待保存和复位完成，再恢复场景并按 `→`。

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
            "recorded_episodes", "phase_labeled_episodes"):
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
| `q` / `Ctrl-C` | 退出，已完成回合保留。 |

若习惯 Evo-RL 的 `o` 键，可加 `--intervention-key o`。交还后的等待保持仍记为人工段；普通推理等待记为 `hold`。采集线程在推理期间继续响应按键，但实际周期还受相机、CAN 和写盘影响，30 Hz 是名义配置。

采集后仍用第 3、4 节检查并生成 warmup 清单。数据会保存逐帧 `is_intervention`、`critical_phase`、实际下发目标与请求目标；`action_source` 的编码为 `0=reference、1=actor、2=human、3=hold`，另存于 `dataset/meta/piper_control.json`。`policy_action_valid=False` 表示该帧没有有效 policy 输出，不能把占位零值当作预测。

`summary.json` 中 `success_rate` 是最终成功率，`autonomous_success_rate` 只把未接管的成功计入分子，分母仍为全部已评分回合；`intervention_rate` 是发生接管的回合比例。每回合另存 `takeovers` 和 `human_steps`。验收 SFT 自主效果时看自主成功率，避免把人工救回算成模型独立成功。

HIL 采集前不需要 latent cache，后续继续用 `warmup init -> cache -> train`。当前 critical-only warmup 只取标注区间，区间外的人工动作保留在原始数据中；混合回合不要整条改成 `behavior=human`。

原生在线 `PiperPolicyEnv` 已接入接管动作和标签回传，交还到新 chunk 才生效；它在分布式推理等待期间仍不能像本地入口一样连续响应接管。在线有效长度、跨阶段折扣和数据池恢复仍需后续接通，不能直接把 HIL 采集通过视为在线探索就绪。实现细节见 `/home/descfly/code/rlinf_piper/docs/piper_human_intervention_design.rst`。本次开发使用 CPU 和 mock 验证，尚未启动真机验收。
