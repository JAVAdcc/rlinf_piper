Piper SFT 验收、标注与离线 Warmup
=================================

用同一份 Stage 1 权重完成 SFT 验收时，可以同步录制带关键段局部结果和整回合独立结果的 LeRobot 数据。采集入口是 ``toolkits/piper/eval_stage1.sh``，离线入口是 ``python -m toolkits.piper.warmup``；两者共享 Piper 的动作单位、相机映射和 Stage 1 元数据。

需要主臂人工接管时，在采集命令中增加 ``--teleop piper_leader``，用 ``i`` 接管 / 交还，``b`` 仍独立标注 critical phase。主臂配置、交接行为和自主成功率的统计方式见 ``piper_human_intervention_design.rst``。下面未启用 teleop 的命令继续执行纯 SFT 验收；采集前无须计算 latent。新采集采用 v3 局部结果格式；离线入口按格式区分 v2 全局回报与 v3 局部终止回报，禁止混用。

采集 100 条 SFT 回合
--------------------

在本机前台终端执行以下命令。它加载已经核对的 H50 FP32 global_step_50000 权重，保持每次执行 50 个目标的协议，按 30 Hz 名义控制频率运行；每条最多执行 3,600 个目标，达到上限后等待人工评分。

.. code-block:: bash

   cd /home/descfly/code/rlinf_piper
   bash toolkits/piper/eval_stage1.sh rollout \
     --num-episodes 100 --record-dataset \
     --output-dir outputs/piper/sft50k_100_20261013

新采集的输出目录必须是新目录。模型加载完成后会连接相机、使能 follower，并等待右方向键；这条命令会启动真机客户端。每条恢复场景后再按右方向键。默认不执行自动复位轨迹，已有的 ``--reset-after-episode initial|zero`` 可按原来的验收流程显式选择。

只有 ``scoring_protocol=critical-segment-v1`` 的新目录可以续采，旧协议目录会在加载模型前被拒绝，需保留原件并使用新输出目录。中断后可使用 ``bash toolkits/piper/eval_stage1.sh rollout --resume outputs/piper/sft50k_100_20261013 --num-episodes 100`` 续采。配置从原目录恢复，目标 100 条包含已有完整回合；撤销和未完成尝试不计数，新目录编号继续递增。原复位点保持不变，上次复位未完成时先等待 ``r`` 完成复位。完整说明见根目录 ``order.md`` 第 8 节。续采结束后再初始化数据池；已有清单和 cache 保持不变，新增数据使用新的输出路径生成清单和缓存。

自动复位在整条 episode 独立评分、数据和视频保存完成后执行；局部关键段评分不触发复位。选择 ``initial`` 时，第一条开始的实测姿态会限制到当前驱动的合法目标范围，``reset_pose.json`` 同时记录原始读数和实际目标，避免零位附近的反馈偏差被当作越界复位目标；所有下发动作仍遵守原有行程限制。选择 ``zero`` 则返回零位，首次启动程序不会提前执行复位。

.. list-table:: 前台终端按键，无需回车
   :header-rows: 1
   :widths: 20 80

   * - 按键
     - 行为
   * - 右方向键
     - 就绪时开始下一条；运行中停止下发本回合新动作并进入评分；采够条数后确认结束。
   * - ``b``
     - 开始一个关键段；段内重复按不会关闭。按键边界对应下一次下发动作的帧号。
   * - ``s`` / ``f``
     - 运行中仅关闭并评分当前关键段，继续执行普通阶段；回合停止后的独立评分界面才记录整条结果。段外 s/f 不停止回合。
   * - 左方向键
     - 运行中或等待评分时直接停止并丢弃当前回合；评分后至下一条开始前也可撤销，无需先按成功或失败。
   * - ``q`` / ``Ctrl-C``
     - 提前退出；保留已经完成的回合。当前未评分回合不进入 warmup 标注清单。

一回合可有多个关键段，例如 ``b → s`` 完成第一段，再 ``b → f`` 完成第二段。随后按右方向键停止下发新动作、保持最后目标，再按 ``s/f`` 独立评整条任务。两段的结果不自动决定整条结果。``b`` 和局部 ``s/f`` 仅改变标注，当前 reference 的 chunk 执行与 HIL 控制权保持原有逻辑；这些键没有实现 actor/reference 的关键段路由。可使用 ``--phase-definition '...'`` 写明一致的进入和成功判据。

如果右键停止时还有未关闭的关键段，先按 ``s/f`` 补评该段，再按右键确认进入整条评分，最后另按 ``s/f``。达到步数上限后也先补评未关闭段，再右键进入整条评分。等待期间不发送新动作；重复 ``s/f`` 不会越过右键确认，同一批输入不能完成两级评分。局部段尚无有效动作时删除空标记，不能生成零长度成功或失败样本。整回合没有关键段也可保存录像和整体结果，局部训练前必须补标。

标注格式为 ``piper-rlt-episodes-v3``，顶层 ``reward_scope=critical_segment``，每条保留独立的全局 ``outcome``：

.. code-block:: json

   {
     "critical_phases": [
       {"start": 120, "end": 180, "outcome": "success"},
       {"start": 300, "end": 360, "outcome": "failure"}
     ],
     "terminal_frame": 420,
     "outcome": "failure"
   }

每段有效动作区间为 ``[start, end)``：起点属于该段，终点是段内最后一条动作之后的观测。``terminal_frame`` 独立表示整个任务最后的观测，不要求等于最后一段的终点。普通段仍会执行动作，因此局部 end 行只复用其观测，不属于前段有效动作。整回合终止观测在停止后、等待评分前捕获，其 wall_time 保留捕获时刻。整回合终止观测的 action 是 hold 占位值，``complementary_info.action_valid=False``；缓存导入会排除它。

每条回合单独保存到 ``episode_001/dataset/`` 等 LeRobot v3 目录，包含三路 RGB 视频、原始七维状态、实际下发目标、原始预测目标、下发前反馈、帧时间和逐帧 critical 标记。角度/夹爪单位是 degree/mm。默认随采集直接进行后台 H.264 编码，结束时排空队列并保存元数据；可用 ``--no-streaming-video`` 切回 PNG 流程。保存完成前不会进入下一条或开始复位，保存期间可以按左方向键撤销。视频时间轴按名义 FPS，真实采样墙钟时间在 ``complementary_info.wall_time``，推理停顿也保留在 trace 中。

运行中或等待评分时，左方向键直接停止并丢弃当前回合，跳过终止观测补拍和完整视频保存；评分后的保存、复位和等待阶段也可撤销。文件释放后移入 ``discarded/``，同步从统计与 ``annotations.json`` 排除。完成已配置的复位、恢复场景后按右方向键补采，撤销回合不占 ``--num-episodes`` 名额。最后一条仍等待右方向键确认结束，或左方向键撤销。尝试目录编号递增，不覆盖旧文件。新 v3 数据使用局部终止训练入口，具体按键和流式编码说明见仓库根目录 ``order.md`` 第 7 节。

``summary.json`` 保存成功率、完成回合数、成功/失败数、录制数及有效阶段标签数；``policy.json`` 保存实际加载的 Stage 1 元数据，``hardware.yaml`` 保存本次硬件配置。每条的 trace、结果、数据集相互独立，上一条结束后已完成文件关闭。控制或录制异常会退出运行，已经完成的回合保持可读。

固定标注与训练/验证划分
-----------------------

下面的命令接受 v2 或完整 v3 标注；不同版本不能合并，在 v2 内补写局部 outcome 仍被拒绝。该命令按父 episode 建立固定 split，输出到配置的 ``annotations`` 路径。它拒绝覆盖已有标注，拒绝漏标、未配对、空区间、重叠或越界阶段；若明确要排除这些回合，可加 ``--skip-unlabeled``，排除名单也会写入结果。排除和训练/验证划分均以完整 episode 为单位，同一回合的多段不会分到不同池，也不会重复计为多个成功样本。

.. code-block:: bash

   USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup init \
     --collection outputs/piper/sft50k_100_20261013/annotations.json

多次采集可以重复传入 ``--collection``。默认按成功/失败分层留出完整 episode，优先留出各 10 条；如果采集时传入 ``--session-id``，同一个 session 会整体留在同一 split。只有一个 session 时不能同时建立训练与验证，需要独立 session。未设置 session 时只是 episode 留出，不能把它解释成跨场景泛化。

如果是已有的普通 LeRobot 数据，使用 ``init --dataset /absolute/dataset`` 生成模板，然后填写关键阶段、终止帧、behavior 和阶段定义。不会自动把单 USB 任务改写成双 USB。缓存入口检查原始任务文本、FPS、边界、结果标签、重复 episode 和 session 泄漏。

缓存与训练使用同一个入口
------------------------

``examples/offline_rl/config/piper_rlt_warmup.yaml`` 集中设置四个主要路径：Stage 1、annotations、cache、output。常规流程只有以下两条命令：

.. code-block:: bash

   USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup cache
   USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup train

``cache`` 加载冻结的 Stage 1，为标注阶段所需的当前状态、动作窗口末端和下一决策状态提取 ``z_rl``、proprio、H50 reference 和原始状态。C10、stride 10 为默认值，动作窗口不会跨越任一阶段终点；同一 episode 的重复状态只提取一次，终止观测也提取特征。每个 episode shard 原子写入，重启相同命令会校验并跳过已经完成的 shard。缓存清单绑定权重、norm、数据源、标注、预处理代码和抽帧规则，改变绑定必须换一个缓存目录。

``video_backend`` 默认仍为 ``torchcodec``，也支持显式设置 ``pyav``。若 LeRobot 的文件句柄接口与本机 torchcodec 版本不兼容，可先用同一批数据核验 PyAV 的帧号、时间戳和 RGB 输入，再设置 ``video_backend=pyav``；不会自动切换解码器。解码器选择、关键依赖版本、外部预处理源码以及 OpenPI 特征提取/RLT 源码一并绑定缓存，改变后使用新目录。

v2 全局标注对应 ``piper-rlt-cache-v2``，v3 局部标注对应 ``piper-rlt-cache-v3``，缓存版本与奖励契约必须一致。旧 v1 标注仍可读取，保留其原来“单一起点到任务终点”的含义；若希望改为分段标注，需要改成 v2 并明确填写各段边界。采集入口会在评分时为未关闭的最后一段补终点；磁盘上的 v2 标注需要包含明确的起止帧，导入时不推断未完整保存的标注。

``train`` 只加载缓存及小型 RLT MLP，使用本机单张 GPU。训练参数和模型尺寸分别来自同一 YAML 与缓存元数据，避免再次手写 H50、FP32 等 Stage 1 配置。训练池在 CPU 内存中按 episode 等概率、再按 chunk 等概率抽样；验证池独立且不可被训练采样。失败回合的动作供 critic 学习，非人工动作的 BC 目标仍是冻结 reference。

人工接管位置同时替换 actor 训练输入中的 reference 和 BC 监督目标，均采用已记录且有效的实际人工动作；混合窗口只替换人工 tick，padding 不生效。reference dropout 在替换后执行。缓存保留原 VLA reference，Bellman 下一状态动作和部署式验证继续使用它，避免把未来人工动作带入待评价的自主 policy；验证中的 ``human_rmse`` 因此仍衡量自主输入条件下对人工动作的接近程度。此训练输入修正会改变训练代码绑定，旧训练 checkpoint 不允许直接续接。

默认训练预算是 5,000 步 actor BC、10,000 步 critic、5,000 步联合更新；联合阶段每 4 步更新一次 actor，总计 6,250 次 actor 更新和 15,000 次 critic 更新。每段尾部不足 C10 的动作窗口按真实有效长度屏蔽。完整的参数与阈值保存到 ``output/config.json``。

``training.bc_reference_dropout`` 控制 BC 阶段每个样本整个 reference chunk 被置零的概率，默认 ``0.0`` 保留原配方。设为 ``0.5`` 时，每次 actor 更新独立遮掉约一半样本的 reference 输入，不缩放保留的输入；图像特征、proprio、原 reference／人工 BC 目标和有效长度 mask 不变。BC 仍使用确定性 actor 均值。该选项不影响 critic bootstrap、验证或部署，它们始终提供完整 reference；联合阶段继续使用独立的 ``dropout_end`` 调度。带遮挡的训练 loss 包含更难的无 reference 样本，应以同一验证集、完整 reference 输入的误差比较模型。它借鉴 openpi-RLT 的输入遮挡机制，但不复制其 Q＋BC warmup 配方。

对旧 v2 全局回报，退出关键阶段不等于任务终止：窗口结束仍在段内时，critic 的下一状态就是动作后的观测；退出该段时，下一决策状态是下一段的起点，若没有下一段则是实际任务终点。中间由 reference 执行的真实控制 tick 计入 ``discount_steps``，不计入 actor 的动作窗口。成功奖励只在任务终点出现，最后一段与任务结束之间的间隔同样折扣；不会给每一段都发一次成功奖励。缓存同时保留 ``action_end``，用于区分紧随动作的观测和下一决策状态。

可以先运行到固定预算中的某个位置，再恢复；``--until-step`` 不改变总预算或就绪要求：

.. code-block:: bash

   USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup train --until-step 1000
   USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup train --resume

支持 ``--config other.yaml`` 和 ``training.batch_size=32`` 这样的显式覆盖。恢复要求训练配置、缓存、设备及实现一致；checkpoint 包含 actor、critic、target critic、两个 optimizer、三个更新计数、验证历史及 CPU/GPU/采样 RNG。``latest.json`` 指向最后一个完整 checkpoint，默认保留最近 3 份。

恢复指纹同时覆盖 actor、critic、sampler、codec 和动作检查实现。中断写入导致的最后一条未完成 JSON 日志可被丢弃；完整行或中间行损坏仍报错，不把日志损坏视为训练通过。

读取训练输出
------------

``metrics.jsonl`` 每 1,000 步记录一次固定窗口验证。v2 按 episode 等权统计，v3 按父 episode 内的局部段分别评分并报告 ``by_phase``，不会把第一段成功归入整体失败；终止 critic 误差按成功/失败平衡，常数预测 baseline 只用训练集拟合。解码动作与 reference 的偏差采用各 episode 第 95 百分位的最大值，比把长轨迹混在一起统计更保守。关节/夹爪检查是缓存观测上的名义边界检查；后续 tick 用上一条预测作为比较基准，并不代表真实机器人已经通过检查。

含人工接管的验证窗口另报告 ``human_rmse`` 与有效 ``human_validation_ticks``，目标是实际人工动作，padding 和非人工帧不进入该统计；此诊断尚无单独放行阈值。更新指标中的 ``actor_grad_norm``、``critic_grad_norm`` 是裁剪前梯度范数，仍按既有验证间隔写日志。

``readiness.json`` 记录数据覆盖、actor 偏差、terminal/TD 误差、Q 分歧和范围，以及绑定权重的哈希。固定预算结束且最后三次定期验证全部通过，才会出现 ``ready_for_deterministic_eval=true``。测试用途缓存始终不能放行。v2 的 ``actor/`` 使用现有 ``PiperRLTPolicy`` 可加载的 Stage 2 格式；v3 局部导出采用离线专用格式，当前真机入口拒绝，``ready_for_deterministic_eval`` 保持 false。离线恢复 checkpoint 则供该 warmup runner 使用。

离线报告的 ``ready_for_online_exploration`` 始终为 false。Online replay 的容量管理、离线/在线混合采样、确定性真机试运行验收及回合边界锁存仍是后续 online 接入工作；不能仅将导出的 actor 传给旧 online worker，就宣称已经完成这些条件。

本机验证记录
------------

2026-10-19 新局部评分协议通过 85 项采集 CPU 回归，旧 warmup 与局部标签拒绝保护通过 31 项 CPU 回归。覆盖分段/整体评分隔离、重复键、右键混批、步数上限、撤销/续采、终止观测捕获时间，以及真实 LeRobot 表/视频保存和 pyav 回读。硬件和模型使用测试替身；此轮没有启动 GPU、真机或新的训练。以下保留旧全局协议的历史验证记录。

2026-10-13 多段标注修改后通过 202 项相关模型、数据、Piper 环境与控制回归；后续将奇数次 ``b`` 改为自动延长到终止帧，更新并通过全部 27 项采集评估测试。覆盖 100 条连续评分、多次进入/退出、奇数次标记自动补终点、到步数上限后关闭阶段、视频/状态/阶段对齐、录制故障时保留人工评分、缓存隔离、reference 间隔折扣、动作 mask、分阶段冻结、精确恢复及当前权重的验证门槛。测试均未连接机械臂；此次 GPU 正在运行另一项真机采集，未重复运行完整 Stage 1 权重的集成测试。

``outputs/review/warmup_integration_20261013.json`` 记录了此前单段版本在真实 Stage 1 权重上的测试链路：从既有观测构造明确标为 ``test_only`` 的四条 LeRobot fixture，提取 12 个冻结状态，完成短程三阶段更新与恢复，并用现有推理接口加载导出的 C10 actor。两种 readiness 标志均为 false；这不是多段版本的完整 GPU 集成验证，也不是正式数据训练或成功率验收。

三路 640×480 图像录制测试中，180 步按 30 Hz 运行用了 6.01 秒，图像复制与入队开销的中位数为 2.98 毫秒、95 分位为 4.66 毫秒，结束后的编码和文件关闭用了 3.96 秒。三路视频分别读回 181 帧，包含终止观测。测试使用已有 RGB 观测的平移副本，不包含真实相机与 CAN 延迟，也不代表长回合的编码时间。

局部回报与按段对照
------------------

v3 每个 ``critical_phases`` 条目必须有独立的 ``outcome``。窗口仍是 C10，末尾不足 10 步时只使用有效 K 步；段内 next 是执行 K 步后的观测，段末 next 是本段 end，terminated=true，next_valid_steps=0。成功段只在末窗口发 r=1，失败段 r=0；本段末尾不跨到下一段 bootstrap，也不把普通阶段长度计入局部折扣。全局 episode outcome 仅保留溯源和父 episode 覆盖统计。

TD 目标为 ``gamma**(K-1)*r + (1-d)*gamma**K*min(target_Q(next, action))``。gamma 仍按控制 tick 定义，默认 0.995；成功末窗口目标因此不是恒定 1，Q 也不是未经折扣的成功概率。不同段的局部价值不能直接当作整体任务价值。

``phase_filter: null`` 训练共享网络；``phase_filter: 0`` 或 ``1`` 只消费按时间排序的第一或第二关键段。过滤先选择包含该段的父 episode，再均匀抽父 episode 和段内窗口；不会改变既有 train/validation split，ordinal 也不声称识别了物理 USB 编号。筛选条件绑定 checkpoint。共享与分开网络使用相同每网络预算时，分开组总更新量翻倍，阶段采样曝光也不同，不能称为等预算架构消融。

复用原生 learner 做分段缓存回放时，调用 ``seed_offline_cache(path, phase_filter)`` 填充该网络的训练 demo pool，筛选条件必须与 ``piper.actor_metadata.phase_filter`` 一致。现场缓存先经 ``select_phase_transitions(arrays, phase_filter)`` 筛选，再转换为 trajectory；筛选保留完整 feature 表及原始 current/next 索引，不改写动作、结果、有效时长或人工标签。两头需要各自的 worker、replay、优化器和输出目录。默认单网络入口仍拒绝专用权重；双 C50 手动专家入口使用显式身份路由和成对发布，见仓库 ``DEPLOY_EXPERTS.md``。

每段验证单列局部成功/失败数量、TD 误差、末端误差及用对应训练段拟合的常数 baseline。缺少任一验证结果类别时，该段 balanced terminal RMSE 为 null，coverage 不通过；重叠窗口不能补充独立 episode 数。原覆盖阈值保持不变。

如果只有局部结果补标，原始数据、父 split 和边界均未改变，可在 CPU 上复用已经验证的冻结特征：

.. code-block:: bash

   USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup relabel-cache \
     --config local.yaml --source-cache /absolute/old_v2_cache

该命令输出到 local.yaml 指定的新 cache 目录，重新核验原始数据、Stage 1 权重/norm、模型预处理代码和环境，保留旧特征生成绑定，并加入来源 cache、标注及转换实现哈希。动作、human mask 和特征不改变，只重建局部 TD 边；输出已存在时拒绝覆盖。边界改变或来源不一致时必须重新提取特征。

``critic_action_projection: driver_limits`` 在 bootstrap、actor-Q 验证及联合 actor-Q loss 中先将 actor 提案转换为当前标准 Piper 驱动裁剪后的下发目标，再屏蔽 padding。使用仿射归一化后的上下界实现，与 decode、关节/夹爪限位、encode 等价；不会改变原始 actor 输出、BC 目标或驱动限位，也没有直通梯度。超界方向的 clamp 梯度为零。当前动作 Q 仍消费真实记录的下发动作，不重新裁剪历史数据。``none`` 保留旧行为。实际连接型号的限位/标定、拒绝动作后的 hold 和真机接管仍需独立验收。

局部训练导出 ``rlinf-piper-rlt-offline-local-actor-v1``，默认加载入口拒绝该格式。共享网络现可用 ``--actor-mode manual-critical --allow-offline-actor`` 显式进入实验评测；按 ``b`` 切到 actor，局部 ``s/f`` 评分后切回 reference。专用网络仍拒绝，未实现专家选择。完整命令、交接规则和测试边界见 ``piper_shared_deployment.rst``。首测权重设置 ``training.joint_steps=0`` 并使用 ``--until-step 15000``，仅完成 BC5000 和 critic10000，没有联合优化；后续联合实验权重须另行区分。
