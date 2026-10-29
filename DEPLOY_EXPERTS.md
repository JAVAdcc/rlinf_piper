# 两段 C50 网络的固定权重评测与独立在线更新

这份交接说明用于部署两套独立的 C50 actor–critic：操作员明确选择第一或第二操作，普通阶段仍执行冻结 H50 VLA。每套初始权重完成 Actor BC 5,000 步和 Critic TD 10,000 步，未做联合更新，BC reference dropout 为 0.5。两套网络只共用冻结 VLA，训练参数、target、优化器、replay 和更新计数相互独立。

训练数据中的第一、第二段来自人工核查的时间顺序，不是视觉识别出的某根 USB 线。现场按任务操作选择 `phase1`／`phase2`，同一操作失败后可以重复选择同一网络。自动识别进入关键段尚未接入。

## 获取代码和权重

代码分支：`piper/separated-c50-deploy-20261029`，基于现场已有 `be934f03`，保留共享入口。现场有未提交修改时，应在新 checkout 部署，或先保留这些修改再切换分支。

```bash
git fetch origin
git switch --track origin/piper/separated-c50-deploy-20261029
```

从 25 拷贝以下独立数据包；权重不存入 Git：

`/home/yiming/workspace/code/rlinf_piper_warmup_20261019/deployment/piper_separated_c50_prejoint_20261029.tar.gz`

包内包含 `phase1/actor`、`phase2/actor`、`experts.yaml`、原 C50 缓存和校验清单。缓存保留原 40/10 父 episode 划分，在线只读取 train，按专家过滤后分别为 1,944／1,455 个窗口。缓存中的原数据绝对路径仅作溯源，使用缓存时不要求现场存在那些原视频目录。

| 初始权重 | `model_state_dict/full_weights.pt` SHA256 |
|---|---|
| phase1 | `1fe32c0d707fcfcc4055106a3b2754d5f676dca893cd982b2468cd8d4787365b` |
| phase2 | `ba718154392ba06a27847bf4b8733787432a331e236e66866fcae7b4c0225cba` |

保留现场原 Stage1：checkpoint ID `b1f2800f7b294b4c95834ad60e0ff23d`，FP32 H50、state、三路相机。沿用已能运行共享 C50 的依赖环境。下面在仓库根目录配置路径：

```bash
mkdir -p checkpoints
tar -xzf /path/to/piper_separated_c50_prejoint_20261029.tar.gz -C checkpoints
export PIPER_EXPERT_PACKAGE="$PWD/checkpoints/piper_separated_c50_prejoint_20261029"
(cd "$PIPER_EXPERT_PACKAGE" && sha256sum -c SHA256SUMS)
export PIPER_EXPERTS_PATH="$PIPER_EXPERT_PACKAGE/experts.yaml"
export PIPER_OFFLINE_CACHE="$PIPER_EXPERT_PACKAGE/offline_cache"
export PIPER_STAGE1_PATH="$PWD/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"
export PIPER_PYTHON="$PWD/.venv-piper/bin/python"
```

`experts.yaml` 中的相对路径相对该文件解析。不要对调两套权重或修改 `phase_filter` 来绕过检查；加载器会核对阶段、Stage1、精度、动作长度和实际权重。

## 先运行无硬件推理，再评测固定权重

给出现场已有 LeRobot episode 目录和一个关键段内的真实帧号，分别检查两个网络；这些 `infer` 命令不连接机器人：

```bash
export PIPER_DATA_PATH=/path/to/episode/dataset
bash toolkits/piper/eval_experts.sh infer-phase1 --frame 200 --output-dir outputs/piper/expert1_probe
bash toolkits/piper/eval_experts.sh infer-phase2 --frame 200 --output-dir outputs/piper/expert2_probe
```

`--frame 200` 是示例，现场应改成对应操作的有效帧。两套都输出 50×7 动作。固定权重推理使用 actor 均值，reference dropout 关闭。单次 H50 VLA 仍可能因去噪起始噪声而改变 reference，不能用两次独立 VLA 调用差异判断 actor 是否偷偷更新。

由现场操作员核对硬件配置、相机和动作保护后运行：

```bash
bash toolkits/piper/eval_experts.sh rollout \
  --hardware examples/embodiment/config/piper_usb_hardware.yaml \
  --num-episodes 10 --output-dir outputs/piper/separated_c50_eval
```

固定评测不会训练。按键顺序：

1. `Right` 开始 episode，普通阶段使用 VLA。
2. `1` 选择第一操作，或 `2` 选择第二操作；再按 `b` 进入该网络。没有选择时 `b` 被拒绝。
3. `s/f` 结束并只评分当前关键段，回到 VLA。活动段中 `1/2` 被拒绝。
4. 下一次进入前重新选择 `1/2`。重试不自动递增网络编号。
5. `Right` 停止 episode，再单独 `s/f` 评分整条；有未评分的活动段时先评分该段，再进入整体评分。

人工接管继续使用既有 `i/o` 配置，与网络选择和结果标签独立。`Left` 丢弃、`q` 退出保留原语义。默认回合后复位到初始姿态，可显式用 `--reset-after-episode none`；保留现有角度、速度、反馈新鲜度和动作幅度检查。`result.json` 每段写入 `expert_id` 和版本，`trace.jsonl` 保存请求与实际动作的专家身份，晚到的旧推理结果会被丢弃。

## 每条 episode 后分别在线更新

固定权重现场行为检查后，可在一个新的输出目录启用在线入口：

```bash
bash toolkits/piper/train_experts.sh \
  --hardware examples/embodiment/config/piper_usb_hardware.yaml \
  --output-dir outputs/piper/separated_c50_online
```

采集按键相同。整条评分、保存和复位后，下一次 `Right` 确认保留才进入 replay；训练期间没有下一条机器人 episode。两头本回合预算完成后，保存两套完整状态并提交同一个指针，下一条开始前启用整对权重。若其中一头出错，停止继续采集，恢复最后完整提交；不在半更新状态下继续运行。

配置为 `examples/embodiment/config/piper_rlt_experts_online.yaml`。它沿用既有算法：C50、在线 stride 2、末尾 align；每新增 transition 获得 5 次 critic 更新，critic/actor 比 2；每头先积累至少 64 条在线 transition；batch 64（32 offline＋32 online），FP32、Adam LR 3e-4、gamma 0.995/控制 tick、tau 0.005。每头独立前 1,000 次 critic 更新内 BC 权重 5→2、Q 权重 0→0.1；actor reference dropout 0.5，推理和 TD bootstrap 不遮挡。在线 actor 使用既有 std 0.002 的探索采样。动作投影、HIL 目标、局部终止和 TD loss 沿用原实现。

这个双专家入口按用户确认的“回合结束后更新”同步运行，不使用旧共享入口的后台边采边训。每个原生调用最多执行 32 次 critic 更新，入口会循环排空预算；未达到 64 条启动门槛时，保存 replay 与累计计数，达到门槛后按累计数据计算更新预算。没有新数据的专家不更新、不推进日程。旧共享网络的数据不会因来源相同就自动流入新在线池；原离线 50 条 train 缓存则按对应原始段序号进入各自 demo 池。

`online_experts/phase1_metrics.jsonl`、`phase2_metrics.jsonl` 保存每轮原生 loss 和更新数；每条 `episode_*/online_update.json` 汇总两头新增 transition、实际 actor/critic 更新数及版本。`online_experts/latest.json` 指向整对 model、target、optimizer、replay、RNG 和已确认 episode 清单。

## 恢复运行

```bash
bash toolkits/piper/train_experts.sh --resume outputs/piper/separated_c50_online --num-episodes 100
```

恢复使用该运行保存的配置和两套状态；保持 Stage1、初始专家目录、缓存与任务不变。不能把旧共享 run 或固定评测目录当作双专家在线 run。已提交 episode 不重复入池；确认后但整对提交前中断的 episode 在恢复时重新处理。

## 验证边界

这些是实验候选权重，既有 offline readiness 门槛未通过。分离网络并没有证明能消除 Q 驱动的动作粗糙化，因此首先交付联合前权重；部署包不含此前粗糙化后的联合回放权重。现场成功率、真实控制延迟、两操作的匹配以及更新后的动作质量仍需现场检查。自动 critical-phase 识别和自动选择网络不在这个手动入口中。

本次验证收据随包提供；CPU 路由、分流与真实 GPU 检查的实际结果以收据为准。无硬件检查不等于真机动作验收。
