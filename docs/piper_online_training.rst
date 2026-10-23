Piper 的异步 online 训练
========================

``toolkits/piper/train_online.sh`` 让 Piper 数采流程驱动原生 RLT 训练。Stage1 为冻结的 H50 FP32 VLA，actor 的执行长度 C 从权重 metadata 读取，支持 C10 和 C50。``b`` 进入关键段，``s/f`` 给该段局部结果并恢复 reference。现场启动和续训命令集中在 ``order.md`` 第 10 节；当前 C50 BC5000 的固定权重测试使用第 13 节 ``eval_c50.sh``，它的 critic 未训练，不能直接用于 online。

确认保留后再训练
----------------

整条 episode 停止、评分、保存和复位后，操作员仍可用 ``Left`` 撤销。只有 ``Right`` 确认保留时才提交后台队列；下一条可以立即开始。后台准备特征、提交 replay、执行 learner 更新并发布完整 checkpoint。最后一条同样需要确认。确认后不能撤销已经用于训练的数据，``q`` 退出前未确认的数据继续留在待确认状态。

默认 ``piper.async_training=true``，训练可以和下一条的推理同时进行。终端和 CAN 留在主进程，``PiperRLTRollout`` 持有冻结 VLA 和 actor，``PiperRLTActor`` 继承 ``RLTACFSDPPolicy``。replay、DataLoader、optimizer、target、checkpoint 和权重广播复用原生组件；``PiperOnlineRunner`` 管理确认和提交事务。当前支持单机单 GPU。后台准备线程和 learner 线程相互独立，准备完成的内存队列最多暂存一条 episode；接受队列只保存路径与结果元数据，原始数据留在磁盘。

执行长度与 replay 窗口
----------------------

机器人执行 C 步后重规划，replay 则独立使用 ``piper.replay_stride=2``：例如 C50 的窗口为 ``[0:50]、[2:52]、[4:54]``。窗口内是逐 tick 实际发送的动作，可以跨过多次推理的边界，包括 hold 和人工接管。下一状态位于窗口结束的真实帧，不是下一个采样起点；折扣仍按实际 tick 数计算。增密 replay 不会缩短 C50 的实际执行 chunk。

每段独立终止，窗口不能跨 critical phase、reset 或未执行动作。默认 ``replay_tail_policy=align`` 保留完整 C 步窗口，并补一个对齐段末的窗口；整段短于 C 时只保留一个带有效长度 mask 的短窗口。成功段最后有效 tick 奖励为 1，失败为 0，``gamma=0.995`` 按 tick 折扣。整条 episode 的评分不覆盖分段奖励。人工帧的 BC target 使用实际人工动作，其他帧使用当前状态的 VLA reference；target Q 在段末不 bootstrap。

warmup 的 cache 命令也支持独立 ``chunk_length``、``stride`` 和 ``tail_policy``。为新 C50 数据创建 cache 时可使用 ``chunk_length=50 stride=2 tail_policy=align``，输出到新目录。旧 C10 transition 不能 reshape 成 C50；本次 C50 测试不读取 C10 rollout 旧数据。

新 C50 数据使用 ``examples/offline_rl/config/piper_rlt_c50_warmup.yaml``，它将标签、cache 和训练输出隔离在 ``outputs/piper/c50/`` 下。该配置不自动读取旧 C10 collection，也不修改现有部署包；正式训练仍需要准备符合训练/验证划分要求的 v3 分段标签。

特征复用与补算
--------------

online 推理使用同一前向结果，把 normalized ``z_rl``、``proprio``、H50 ``ref_chunk`` 和 raw state 写入 episode 的 ``replay_anchors/``。锚点绑定发起请求的观察帧，而不是异步预测返回的帧。即使动作随后被人工接管打断，这个已录制观察帧的冻结特征仍可复用；未执行就取消的终止帧请求不作为 terminal observation 的缓存。

确认后选择所有窗口的起点和下一状态，去重并校验锚点。对缺失状态按 ``piper.replay_feature_batch_size=2`` 补算，使用该状态自身的图像生成 reference，不平移前一个状态的 reference。补算在独立 RNG 上下文中运行，不消耗 rollout 的探索随机数。完成后原子保存 ``online_features.npz``，并在 ``online_features.json`` 中记录输入摘要、复用数量、补算数量、batch 数和耗时。重试会复用已校验的 shard。

离线与在线两个数据池
--------------------

新运行必须通过 ``--offline-cache``、``PIPER_OFFLINE_CACHE`` 或配置 ``piper.offline_cache`` 指定 warmup cache。只导入 train split；validation 留作评估。cache 必须匹配 Stage1、任务、归一化和 actor/reference 的 horizon。只加载小网络权重不会自动加载数据。

online 池均匀、有放回采样最近 200 条有效 episode 的 chunk；demo 池保存全部离线训练数据，``sample_window_size=0`` 不设置最近窗口。人工接管 chunk 保留在在线池，并携带逐 tick human mask，不重复加入离线池。两个池分别采样 32 个 transition，组成固定 batch 64；池较小时仍通过有放回采样维持比例。磁盘 replay 和原始数据持续保留，不随 200 条采样窗口自动删除。

训练预算与恢复
--------------

``examples/embodiment/config/piper_rlt_online.yaml`` 启用原生 transition 计数调度。至少积累 64 个在线 transition 才开始更新；导入的离线样本不产生 online 更新预算。每个新增 transition 产生 5 次 critic 更新预算，后台每轮最多执行 32 次并发布快照，有剩余预算就继续，不等待下一次确认。每 2 次 critic 更新一次 actor。这个入口的 UTD 分母明确为新增 replay transition，不是原始帧或 batch 样本曝光次数。

例如第一条生成 100 个 transition，产生 500 次预算，后台分轮执行直至预算耗尽或程序退出。每轮实际更新量和预算写入 ``online/async_metrics.jsonl``，最新一轮写入 ``async_status.json``。episode 的 ``online_update.json`` 仅记录其入池轮次，不代表随后全部预算。最终 episode 确认后等待后台排空，期间 q 允许保存余量并退出。

actor 和 critic 学习率均为 ``3e-4``，每步更新 target Q，``tau=0.005``。前 1,000 次 critic 更新逐渐把 Q 权重由接近 0 提至 0.1，把 BC 权重从 5 降至 2。actor 使用导出网络的固定标准差采样。初始化要求 critic 已 warmup，当前 BC-only C50 包会在模型和硬件启动前被拒绝；测试 actor 的命令不受影响。

新建运行从 warmup heads 初始化 target，并建立新的 optimizer。``--resume`` 恢复模型、target、optimizer、学习率、两个 replay 池、累计新增 transition、更新计数及随机数状态。离线池已经包含在 checkpoint 内，续训不再需要原离线 cache 路径。配置与权重绑定由运行快照恢复，新 C50 数据使用新目录；不在旧 run 中更换 stride 或模型。

``online/latest.json`` 只在完整 checkpoint 发布后更新。确认但未提交成功的 episode 重试，已提交的目录不会重复入池。异步退出会完成当前更新小轮并保存；未完成特征补算的已接受 episode 依靠 result.json 恢复。恢复不重复入池，也不丢失已保存的训练预算；并发时序与真实轨迹不保证逐位复现。旧运行未配置 async_training 时继续同步执行。默认保留最近 3 份 checkpoint，以及续训进程正在读取的一份旧 checkpoint；旧 checkpoint 清理不删除在线原始数据。产出的 ``actor/model_state_dict/full_weights.pt`` 可用于独立评测。

异步快照与推理优先
------------------

参考 `RLToken §5 <https://arxiv.org/html/2604.23073v2#S5>`_ 和 openpi-RLT 提交 ``c1e40ac360185778c98cf20da2820e22d2d415e7`` 的 `LearnerService.run_forever / export_actor_snapshot <https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/src/rlt_online_rl/trainer.py>`_、`推理端 snapshot poller <https://github.com/Yyshadow/openpi-RLT/blob/c1e40ac360185778c98cf20da2820e22d2d415e7/rlt_online_rl/src/rlt_online_rl/inference.py>`_。迁移的是独立更新和不可变快照发布机制，训练仍使用本仓库 PyTorch/RLinf 原生 RLT 链路。

初始和同步模式沿用 WeightSyncer；异步模式 learner 完整保存后原子替换 latest.json。rollout 在下一次推理检查版本、SHA256、张量键/形状/dtype/finite，验证完成才安装，并在同一个串行 RPC 内执行 forward。拒绝回退版本和不同运行的快照。当前排队动作继续使用原预测；trace、parquet 记录每个 tick 实际使用的 policy_version，human/hold 为 -1。跨版本 replay 窗口的版本为 -1，原逐帧数据仍保留版本。

为适配单张 5090，只保留一份冻结 VLA。InferenceGate 序列化推理与特征补算，补算按 batch=2 分割，等待中的推理优先；无法抢占一个已经开始的 forward。特征补算在 VLA worker 的独立 RNG 上下文中执行。后台 learner 属于另一 worker，可与推理并发，但共享 GPU 仍可能增加推理延迟。trace 的 background_wait_s 可分辨等待特征补算的部分；这不是硬实时保证。

后台错误在控制循环及评分、复位、等待按键期间通过健康检查上报，不会静默使用旧版本继续采集。q 退出停止新后台工作并保存当前 learner 小轮；最后一次正常确认则排空已确认工作再退出。语音“已提交”表示接受到后台，不误报“已更新”。

不用机械臂检查完整链路
----------------------

以下 GPU 集成测试默认生成新合成观察、控制 tick 和离线 cache，不读取旧 C10 rollout，也不连接相机或机械臂。它单独关闭测试配置的 critic warmup 检查，用于验证 C50 网络的更新与恢复，测试输出不能用于真实训练或任务成功率评估。

.. code-block:: bash

   cd /home/descfly/code/rlinf_piper
   export USE_TF=0 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH="$PWD"
   piper_test_args=(
     --stage1 checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor
     --actor checkpoints/piper_shared_c50_bc5000_20261022/actor
     --output /tmp/piper-c50-online-smoke
   )
   .venv-piper/bin/python tests/e2e_tests/embodied/piper_online.py "${piper_test_args[@]}"
   .venv-piper/bin/python tests/e2e_tests/embodied/piper_online.py "${piper_test_args[@]}" --resume
   .venv-piper/bin/python tests/e2e_tests/embodied/piper_online.py "${piper_test_args[@]}" --resume

第一遍检查 H50/C50 推理、锚点复用、批量提取、离线池导入、入池、真实 actor/critic 梯度更新和权重广播。后两遍从相同 checkpoint 继续消耗预算，比较得到的新权重摘要，覆盖 replay、optimizer、RNG 与计数恢复。CPU 单元测试另外覆盖段末对齐、短段 mask、人工动作混合、固定 batch、离线池保留及 C50 的中途切换和 HIL。

异步 GPU 检查使用同一个脚本加 ``--asynchronous``，输出到新的测试目录，随后运行一次 ``--resume``。它在训练 RPC 期间持续真实 H50/C50 推理，检查快照热更新、退出时未耗尽预算与恢复后总更新数；合成数据只用于验证并发和恢复，不能衡量任务成功率。CPU 测试另覆盖推理优先、后台异常、完整快照校验和逐 chunk 版本归属。
