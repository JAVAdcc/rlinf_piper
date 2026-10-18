Piper 人工接管与训练数据
========================

当前 ``rlinf_piper`` 已接入 Evo-RL 的 Piper 主臂跟随、MIT 重力补偿、人工接管和交还控制。本地入口仍是 ``toolkits/piper/eval_stage1.sh rollout``，增加 ``--teleop piper_leader`` 即可启用；默认关闭，原来的单从臂 SFT 验收流程继续可用。以下说明本地操作、数据进入 warmup 的方式，以及原生在线环境的接口和限制。

启动与操作
----------

本机 ``.venv-piper`` 已安装可选 HIL 依赖。重建环境时使用以下命令，无须替换现有 LeRobot 或把 Evo-RL 加入 ``PYTHONPATH``：

.. code-block:: bash

   cd /home/descfly/code/rlinf_piper
   uv pip install --python .venv-piper/bin/python -r requirements/embodied/piper-hil.txt

在前台终端启动带接管功能的 SFT 验收与数据采集：

.. code-block:: bash

   bash toolkits/piper/eval_stage1.sh rollout \
     --teleop piper_leader \
     --num-episodes 20 --record-dataset --save-observations \
     --no-action-limits --reset-after-episode initial \
     --output-dir "outputs/piper/hil_$(date +%Y%m%d_%H%M%S)"

硬件配置位于 ``examples/embodiment/config/piper_usb_hardware.yaml``：从臂 USB-CAN 序列号为 ``003F00334648571220363032``，主臂沿用 Evo-RL USB 任务的 ``004F003C4648571020363032``。2026-10-17 只读检查时分别解析到 ``can3`` 和 ``can1``；程序始终按序列号解析，不依赖 CAN 编号。换主臂时使用 ``--leader-usb-serial``；同一通道用于两臂会在连接前报错。

相机就绪后连接两臂并等待右方向键。按右方向键后，主臂用 3 秒对齐从臂当前姿态，再开始回合；这一步会移动主臂，对齐时可按 ``q`` 或 Ctrl-C 退出。主臂跟随从臂最终下发目标；接管前检查两臂实际反馈差，默认关节最大 10 度、夹爪 15 毫米，超限则停止并报告。

.. list-table:: 本地前台按键
   :header-rows: 1
   :widths: 20 80

   * - 按键
     - 行为
   * - 右方向键
     - 恢复场景后，对齐主臂并开始回合。
   * - ``i``
     - 接管 / 交还，可用 ``--intervention-key o`` 改为 ``o``。
   * - ``b``
     - 进入 / 退出 critical phase，可标注多段，与接管独立。
   * - ``s`` / ``f``
     - 停止动作、评分、保存数据，然后按已选配置复位。
   * - 左方向键
     - 评分后撤销刚结束的回合，保存中、复位中和等待下一条时均有效；完成复位后按右键重采。
   * - ``q`` / Ctrl-C
     - 结束运行，保留此前已完成的回合。

接管时主臂保持硬件角色 ``0xFC``，切入 200 Hz MIT 重力补偿，夹爪解除驱动；从臂按任务名义频率 30 Hz 跟随主臂反馈，human 优先于 reference 和 actor。再次按接管键，主臂恢复位置控制，从臂保持最后的人为目标，直到当前观测的新预测返回。交接等待段仍标记为 human。

本地 HIL 在独立线程推理，控制循环继续处理按键、人工动作和记录。每次接管或交还都会使之前的预测失效，即使旧推理稍后返回，也不执行其剩余动作。普通推理等待期间记录 ``hold``，不作为人工示范。相机、CAN 和磁盘仍可能影响周期，30 Hz 是配置目标，不代表已经完成真机延迟验收。

``b`` 的配对区间仍为 ``[start, end)``，奇数次标记在评分时延长到终止帧。复位先结束接管、关闭本回合记录，再按现有轨迹控制从臂并同步主臂；复位帧不进入训练数据。``--no-action-limits`` 沿用此前命令，关闭额外的 rollout 步幅检查，驱动行程限制继续生效。

视频编码与撤销
--------------

本地录制默认使用 ``--streaming-video``，参照 Evo-RL 将三路 RGB 在采集过程中直接交给后台 H.264 编码线程。每路最多缓存 32 帧，使用 ``veryfast``、CRF 30、GOP 2 和 2 个编码线程。无需逐帧 PNG 回读，回合结束只排空剩余帧、写入终止观测和元数据；``--no-streaming-video`` 可切回原 PNG 流程。

``PiperEpisodeRecorder`` 复用当前 LeRobot 的流式保存、统计归一化和视频元数据接口，并在 ``rlinf/data/storage/lerobot/video.py`` 提供严格逐帧的编码器。上游默认队列满时丢帧，本地编码器则最多等待 1 秒后报告失败，失败回合不发布到训练清单；成功保存前校验每路 MP4 帧数。图像统计在后台对每帧降采样后增量计算，编码队列与统计内存不随回合长度增长。

评分后的保存由单一 worker 持有 recorder，主线程继续读取左键和退出请求。左键只设置取消事件并更新回合状态，不从主线程并发关闭文件。编码线程、视频文件和 parquet writer 释放后，整条尝试移到 ``discarded/``；成功率与 ``annotations.json`` 立即排除它。撤销窗口一直保留到右键开始下一条，连续左键不能撤销更早回合。保存完成后只涉及文件移动与清单更新，不重新编码。

``--num-episodes`` 按保留回合计数，尝试目录编号始终递增。最后一条也等待确认：右键正常结束、左键补采、``q`` 保留当前结果并退出。复位期间主线程每个 tick 处理撤销 / 退出，撤销数据不打断复位轨迹。若编码器未能停止，保留原目录并报告清理错误，不移动仍被使用的文件。详见 ``order.md`` 第 7 节。

2026-10-18 在本机采集磁盘上对照三路 640×480、30 Hz、180 步合成视频：旧 PNG 流程评分后保存耗时 3.512 秒，流式流程 0.168 秒；三路均为 181 帧，含终止观测。录制调用的 P95 耗时分别为 0.271 毫秒和 0.677 毫秒。该测试验证磁盘与编码路径，没有同时运行模型或机械臂，实际长回合耗时以现场运行记录为准。

中断后继续采集
--------------

使用 ``bash toolkits/piper/eval_stage1.sh rollout --resume <原采集目录> --num-episodes 100`` 继续采集到累计 100 条。入口从原 ``run.json`` 恢复参数，并使用原 ``hardware.yaml``；原始启动快照保持不变，续采快照追加到 ``resumes/``。可覆盖目标条数、计算设备、语音和视频编码开关，其他采集配置沿用原设置。完整可执行命令见 ``order.md`` 第 8 节。

CLI 从模型加载到硬件、录制和语音清理结束一直持有目录文件锁。``rlinf/robotics/piper_collection.py`` 只读扫描各回合的 ``result.json`` 和目录编号，以回合结果恢复可能滞后的汇总，兼容旧目录没有 ``attempt`` 字段的情况。已发布录制核对动作表、视频头和元数据帧数；未评分或录制未提交的尝试保持原文件，恢复后的汇总将其排除。已发布文件损坏时在连接硬件前报错。``summary.json`` 和 ``annotations.json`` 从完整保留的回合重建，编号从所有尝试的最大值继续。

复位目标从原 ``reset_pose.json`` 恢复，重启不重新采样 initial。最后一次尝试复位未完成时先等待 ``r``，复位完成后才允许右键开始新回合；旧复位日志保留，新尝试写入下一个 retry 日志。最后一次完整保留的回合继续支持左键撤销，尚未开始下一条前有效；未完成或已撤销的尝试不会使左键跳过它们删除更早回合。

本地语音播报
------------

SFT 和 HIL 的本地交互入口默认开启英文语音，可用 ``--speech-language zh`` 切换中文、``--no-play-sounds`` 关闭。完整命令和播报事件表见仓库根目录 ``order.md`` 第 6 节。播报覆盖准备、对齐、开始、critical 标记、接管 / 交还、评分、数据保存、撤销、最后一条确认、复位 / 重试及退出；不接入分布式在线 worker 或离线 warmup。

每次只播报当前操作的短句：英文 1–3 个单词，中文 2–5 个字，例如 ``Success`` / “成功”、``Discarded`` / “已丢弃”、``Resetting`` / “复位中”。回合编号、按键说明和详细状态留在终端。``Reset finished`` / “复位结束”仍表示轨迹发送完成，实际反馈误差由终端报告。

``rlinf/robotics/piper_announcements.py`` 直接使用 Evo-RL 同款本地 ``spd-say``，不引入额外 Python 依赖。一个后台线程串行等待播放，队列最多保留 8 个提示；同主题的未播放状态只保留最新值，短期状态 3 秒后过期、结果 10 秒后过期。语音缺失、非零退出或单句超过 8 秒时只警告一次，继续终端交互和采集。

``evaluate_piper`` 在实际操作成功后发出事件：critical 和评分先写结果，保存完成先关闭 recorder 并更新清单，交还后的恢复事件由 ``rollout_piper`` 在新预测动作第一次成功下发后发出。旧预测返回、仍在等待或再次接管时不会误报恢复。评分成功与数据保存成功分开，编码失败不播报保存成功。

CLI 持有唯一的播报器上下文，先退出 ``PiperSession`` 和终端按键上下文，再清理语音。正常完成保留最后的评分、保存和复位结果并最多等待 8 秒；中断 / 故障清掉待播历史，最多等待 3 秒。之后终止并回收本实例的子进程，回收最多再等 0.75 秒。不发送影响其他应用的全局 stop / cancel；系统已接收的短句可能继续读完。播报不参与控制权仲裁、动作记录或 critical 标签计算。

动作、标签和成功率
------------------

每个回合保持原有 LeRobot 目录、三路视频和终止观测约定。物理动作字段统一为六个关节角度加夹爪毫米值；测量状态另存，不把命令目标描述成机械臂实际到达的位置。

* ``action``：从臂执行接口确认的下发目标，包含执行端限位处理。
* ``complementary_info.requested_action``：本帧选中的请求目标，可能来自 policy、主臂或等待保持。
* ``complementary_info.command_state``：下发前再次读取的从臂反馈。
* ``complementary_info.is_intervention``：本帧由人工控制，包括交还后的保持等待。
* ``complementary_info.critical_phase``：用户用 ``b`` 标注的关键阶段，与人工标签独立。
* ``complementary_info.action_source``：``0=reference, 1=actor, 2=human, 3=hold``，映射另存 ``meta/piper_control.json``。
* ``complementary_info.policy_action``、``policy_action_valid``：当前执行的 policy 预测及有效性。human / hold 段无有效预测，零值只是占位。
* ``complementary_info.action_valid``：是否有实际动作；末尾终止观测为 False，不能作为训练动作。

Evo-RL 的 ``action_values`` 和 ``_sent_action`` 表示选中目标与发送接口返回目标，并非主臂和从臂各自的 action。默认无 smoothing 时，两者之间主要执行 SDK 单位量化。本实现沿用 Evo-RL 的控制方式，保留 RLinf 录制确认命令的约定，同时保存请求目标和测量状态。

``summary.json`` 同时记录最终成功率 ``success_rate``、自主成功率 ``autonomous_success_rate`` 和接管回合比例 ``intervention_rate``。自主成功率的分母是全部已评分回合，分子只计未发生接管的成功回合。每条回合的 ``takeovers`` 和 ``human_steps`` 分别记录接管次数与人工帧数。

继续使用离线 warmup
-------------------

采集时不需要 latent cache。后续继续通过 ``python -m toolkits.piper.warmup init -> cache -> train`` 处理，具体命令见 ``piper_offline_warmup.rst``。缓存由固定 Stage 1 生成，数据池保持为不可变 CPU 数据集；新增回合需要生成新清单和缓存。

混合回合的默认 behavior 仍为 reference 或 actor，逐帧人工标签决定 BC 的 human target，不能把整条混合回合标为 human。现有 critical-only warmup 只使用标注区间内的数据，区间外人工动作保留在原始数据中。

原生在线接口与边界
------------------

``PiperPolicyEnv`` 复用同一个 ``PiperSession`` 和 ``PiperIntervention``。把它读取的硬件 YAML 中 ``teleop.type`` 改为 ``piper_leader`` 即可启用；外层 ``env.train.teleop`` 保持 ``none``，不要额外叠加通用 teleop wrapper。原生环境使用 evdev 键盘：``a`` 开始、``b`` 切换下一 chunk 的 critical phase、``i`` 接管 / 交还、``c/q`` 成功 / 失败，与本地采集的评分键不同。

人工目标经过与 policy 相同的从臂执行接口，返回的 ``intervene_action`` 是 ``codec.encode(session.last_command)``；超出训练分位数的人工动作不会被额外裁到 ``[-1, 1]``。``executed_action`` 单独回传全部确认命令，经过 ``RealWorldEnv -> EnvWorker -> EnvTransition -> TrajectoryStep`` 进入 replay，human mask 单独保留。原生环境的动作来源区分 policy / human / hold；actor 与 reference 的选择由上游 route 元数据说明，避免把 warmup 期间的 reference 错标成 actor。

``on_action_chunk_begin`` 只在新 chunk 到达时接受交还。旧 chunk 内请求交还后，其剩余动作持续被保持目标覆盖，并保留人工标记。评分后，剩余 slot 返回终止占位，不再运动，不重复给成功奖励，也不追加人工标签。

在线 worker 在 chunk 之间仍等待分布式推理，按键事件在下一个执行 tick 处理，尚不具备本地独立推理线程的连续响应。在线 ``b`` 按下一 chunk 生效，粒度不同于本地逐帧标注。原生 learner 尚未接通离线缓存那套完整的逐帧有效长度、跨关键阶段折扣和数据池恢复协议。因此当前可验收本地 SFT / actor HIL 采集及 warmup 数据，不能据此宣布整套在线 RLT 探索就绪；离线报告的 ``ready_for_online`` 仍为 False。

实现和验证
----------

本次修复了人工动作写入 ``curr_obs.ref_chunk`` 的问题：冻结 reference 是 actor 的输入，必须与推理时一致，人工动作只更新执行动作及 BC mask。环境 human mask 与模型 expert mask 取并集，模型返回 False 不会抹掉真实接管记录。模拟环境的 expert route 同样保留冻结 reference。

主臂驱动位于 ``rlinf/robotics/parts/teleop/piper_leader.py``，控制权状态机位于 ``rlinf/robotics/piper_intervention.py``，本地执行位于 ``rlinf/robotics/piper_runtime.py``。主臂使用 Evo-RL 固定提交的 ``piper_sdk 0.6.1``，从臂保留 ``pyAgxArm``。两臂各有一个连接所有者，通过 ``DeviceClaim`` 防止同通道重复占用，退出时停止重力补偿并恢复位置保持，再释放连接。

可选依赖固定 ``pin==2.7.0``，兼容当前 Python 3.12 / NumPy 1.26；Evo-RL 的 Pinocchio 3.9 依赖 NumPy 2，因此未覆盖现有推理环境。URDF 原样保留，来源与哈希见 ``rlinf/robotics/parts/teleop/assets/README.rst``。CPU 对照覆盖 4 组负载 / 基座朝向与每组 32 个姿态，力矩最大绝对差约 ``2.2e-19 Nm``。

review 记录保存在 ``outputs/review/piper_human_intervention_20261017/review.json``，实现验证记录位于同级 ``implementation/``。测试覆盖模式切换与失败清理、推理中的接管、交还后的旧 chunk 丢弃、录制后数据重读、critical 标签独立性、实际归一化命令回传和冻结 reference。真机对齐、重力补偿手感、CAN 时序与成功率需要操作员现场验收，开发过程中未连接或驱动机械臂。
