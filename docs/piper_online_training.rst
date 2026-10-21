Piper 的同步 online 训练
========================

``toolkits/piper/train_online.sh`` 让当前 Piper 数采流程直接驱动原生 RLT 训练。Stage1 为冻结的 H50 FP32 VLA，共享 actor 执行 C10；``b`` 进入关键段，``s/f`` 给该段局部结果并恢复 reference。人工接管、流式视频、立即丢弃和复位继续由本地 controller 管理。现场启动和续训命令集中在仓库根目录 ``order.md`` 第 10 节。

确认保留后再训练
----------------

整条 episode 停止、评分、保存和复位后，操作员仍可用 ``Left`` 撤销。只有 ``Right`` 确认保留时，才提交 replay、执行 learner 更新、保存 checkpoint 并广播新权重；这些步骤完成后下一条才开始。最后一条同样需要 ``Right`` 确认。确认后不能再撤销已经用于训练的数据，``q`` 退出前尚未确认的数据则继续留在待确认状态。

训练发生在两条 episode 之间。终端和 CAN 操作留在主进程，``PiperRLTRollout`` worker 持有一个冻结 VLA 和一个 actor，``PiperRLTActor`` 继承原生 ``RLTACFSDPPolicy``。replay ingestion、DataLoader、optimizer 更新、target 更新、checkpoint 和权重广播复用原生组件；``PiperOnlineRunner`` 负责操作员确认边界与提交事务，不另写优化器循环。当前配置支持单机单 GPU，尚未提供边执行边学习的异步模式。

局部奖励和数据池
----------------

每个关键段独立终止。成功段的最后一个有效 tick 给 1，失败段给 0，``gamma=0.995`` 按实际执行 tick 折扣；整条任务的 success/failure 只用于整体统计。局部终止既不跨到下一个关键段 bootstrap，也不触发机械臂复位。不到 C10 的末尾 chunk 保留真实长度，补齐部分不进入 BC 或 Q 输入。人类监督使用记录中实际发送的动作；actor 的 Q 输入按驱动行程投影，BC 仍针对原始 actor 输出。

确认后复用 rollout worker 已加载的 VLA，为关键段每 10 帧及边界提取一次特征，保存为该 episode 的 ``online_features.npz``。这与离线 warmup 共用 transition 和损失语义；同一确认数据重试时复用已校验的 shard。无需预先计算新轨迹的 cache，代价是长关键段在回合之间会有提取等待。GPU smoke 中仅有两个短测试段，耗时不能外推到完整任务。

默认 online 池保存所有确认的有效段，均匀采样最近 200 个 episode 中的 chunk。可设置 ``piper.offline_cache`` 接入已有 warmup cache：只导入 train split，使用原生 demo 池，online 和 demo 各采样半个 batch，人工接管 chunk 也加入 demo 池。validation 不参与更新，cache 必须绑定相同 Stage1、任务、归一化及 H50/C10。未提供 cache 时不自动拉取旧数据，云端小网络导出本身不含 replay。

训练参数与恢复
--------------

``examples/embodiment/config/piper_rlt_online.yaml`` 默认每个确认 episode 更新 critic 32 次、actor 8 次，batch 上限 64，actor 和 critic 的 Adam 学习率均为 ``3e-4``。前 1,000 次 critic 更新逐渐把 Q 权重从 0 提至 0.1，并把 BC 权重从 5 降至 2。在线不重跑 BC／critic warmup。actor 使用导出网络的固定标准差采样，reference dropout 首测设为 0。配置并不保证任务成功率会提高。

新建运行从 warmup actor 和 critic 初始化，target 从当前模型初始化，optimizer 从头建立。``--resume`` 则恢复整个 online 状态：模型、target、optimizer、学习率、replay、demo 池、更新计数和各随机数状态。训练配置和硬件配置由运行目录的快照恢复。``online/latest.json`` 仅在完整 checkpoint 写好之后发布；确认过但没有提交成功的 episode 会重试，已经提交的目录不会重复入池或重复更新。

``episode_*/result.json`` 保存采集时的 policy 版本，``online_update.json`` 给出实际更新后的 rollout 版本、权重摘要、loss 与耗时。完整状态位于 ``online/checkpoint_XXXXXX/actor``，其中 ``model_state_dict/full_weights.pt`` 和 ``piper_rlt_actor.json`` 可交给共享 actor 评测入口，仍需手动 critical routing。默认保留最近 3 份 checkpoint，加上续训进程仍在读取的一份旧 checkpoint；replay 和原始数据持续保留。

不用机械臂检查完整链路
----------------------

以下检查需要本机 GPU、真实 Stage1／warmup 权重和一条至少包含 10 个有效动作的现有录制 episode。它读取录像并使用合成的成功／失败短段标签，仅验证程序和恢复，不连接机械臂、不更改原采集目录，也不测任务成功率。测试产物不能作为正式训练权重或人工标注使用。

.. code-block:: bash

   cd /home/descfly/code/rlinf_piper
   export USE_TF=0 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH="$PWD"
   piper_test_args=(
     --stage1 checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor
     --actor checkpoints/piper_shared_prejoint/actor
     --source-episode /path/to/recorded/episode_001
     --output /tmp/piper-online-smoke
   )
   .venv-piper/bin/python tests/e2e_tests/embodied/piper_online.py "${piper_test_args[@]}"
   .venv-piper/bin/python tests/e2e_tests/embodied/piper_online.py "${piper_test_args[@]}" --resume
   .venv-piper/bin/python tests/e2e_tests/embodied/piper_online.py "${piper_test_args[@]}" --resume

第一遍检查 H50/C10 推理、录制数据提特征、入池、真实 actor／critic 参数变化和新权重广播。后两遍核对 checkpoint 恢复、重复提交保护，并从相同状态继续更新，比较后续权重摘要。没有上述权重、数据或 GPU 的 CI 只运行 CPU 单元测试，不执行此脚本。
