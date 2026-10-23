# Piper 常用命令

先执行公共设置，再选一个运行命令。所有新运行均带 `--no-action-limits`，关闭额外动作范围/步幅检查；驱动行程限位仍保留。

## 1. 公共设置

```bash
cd /home/descfly/code/rlinf_piper
export PIPER_PYTHON="$PWD/.venv-piper/bin/python"
export PIPER_STAGE1_PATH="$PWD/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"
piper_data_dir="/media/descfly/新加卷/value/data_inference"

piper_variant=c50  # 切换 C30 时改为 c30，并重新执行下面的设置
export PIPER_ACTOR_PATH="$PWD/checkpoints/piper_reference_dropout_c30_c50_20261023/$piper_variant/actor"
case "$piper_variant" in
  c30) export PIPER_OFFLINE_CACHE="$PWD/outputs/piper/offline_cache/cache_local_segments_c30_s10_20261023" ;;
  c50) export PIPER_OFFLINE_CACHE="$PWD/outputs/piper/offline_cache/cache_local_segments_c50_s10_20261022" ;;
esac
```

## 2. SFT 权重推理

```bash
piper_run_dir="$piper_data_dir/rlinf_sft_$(date +%Y%m%d_%H%M%S)"
bash toolkits/piper/eval_stage1.sh rollout \
  --teleop piper_leader --num-episodes 5 \
  --steps 7200 --replan-steps 50 --no-action-limits \
  --reset-after-episode initial --output-dir "$piper_run_dir"
```

## 3. Actor 权重推理（不训练）

普通阶段执行 SFT H50；按 `b` 切到 C30/C50 actor，段内 `s/f` 后回到 SFT。

```bash
piper_run_dir="$piper_data_dir/rlinf_${piper_variant}_eval_$(date +%Y%m%d_%H%M%S)"
bash toolkits/piper/eval_shared.sh rollout \
  --teleop piper_leader --num-episodes 5 \
  --steps 7200 --replan-steps 50 --no-action-limits \
  --record-dataset --save-rlt-reference \
  --reset-after-episode initial --output-dir "$piper_run_dir"
```

## 4. Online 完整训练

使用所选 actor + critic 和匹配的离线池。训练 dropout=0.5，推理保留 reference；每条 episode 固定一份权重，后台一轮训练完成后保存。

```bash
piper_online_dir="$piper_data_dir/rlinf_${piper_variant}_online_$(date +%Y%m%d_%H%M%S)"
printf '本次训练目录：%s\n' "$piper_online_dir"
bash toolkits/piper/train_online.sh \
  --teleop piper_leader --num-episodes 100 \
  --steps 7200 --replan-steps 50 --no-action-limits \
  --reference-dropout 0.5 --record-dataset --save-rlt-reference \
  --reset-after-episode initial --output-dir "$piper_online_dir"
```

## 5. Resume 续训

填写上次目录；`--num-episodes` 是累计目标。模型、数据池、训练配置均自动恢复。以下命令用于原本就带 `--no-action-limits` 的运行。

```bash
piper_online_dir="/media/descfly/新加卷/value/data_inference/填写上次训练目录"
bash toolkits/piper/train_online.sh \
  --resume "$piper_online_dir" --num-episodes 100 --no-action-limits
```

## 6. 数采（SFT + 人工接管）

右键开始后按 `i` 遥操接管，再按 `i` 交回 SFT；用 `b`、`s/f` 标注关键段。

```bash
piper_run_dir="$piper_data_dir/rlinf_collect_$(date +%Y%m%d_%H%M%S)"
bash toolkits/piper/eval_stage1.sh rollout \
  --teleop piper_leader --num-episodes 100 \
  --steps 7200 --replan-steps 50 --no-action-limits \
  --record-dataset \
  --reset-after-episode initial --output-dir "$piper_run_dir"
```

续采或恢复固定权重评测，填写原目录：

```bash
piper_run_dir="/media/descfly/新加卷/value/data_inference/填写上次数采或评测目录"
bash toolkits/piper/eval_stage1.sh rollout \
  --resume "$piper_run_dir" --num-episodes 100 --no-action-limits
```

## 按键

| 按键 | 操作 |
|---|---|
| `→` | 等待时开始；运行中停止整条；保存后确认保留，online 同时提交后台训练 |
| `b` | 开始关键段 |
| `s` / `f` | 段内标记该段成功/失败；整条停止后另行评分整条 |
| `i` | 人工接管 / 交还 |
| `←` | 丢弃当前或刚完成、尚未确认的 episode |
| `q` | 退出，之后可 resume |

常用收尾：段内 `s/f` → `→` 停止整条 → `s/f` 评分整条 → 保存复位 → `→` 确认并继续。
