Piper SFT 验收、标注与离线 Warmup
=================================

用同一份 Stage 1 权重完成 SFT 验收时，可以同步录制带关键阶段标签的 LeRobot 数据，再通过一个入口准备缓存和训练 RLT actor-critic。采集入口是 ``toolkits/piper/eval_stage1.sh``，离线入口是 ``python -m toolkits.piper.warmup``；两者共享 Piper 的动作单位、相机映射和 Stage 1 元数据。

需要主臂人工接管时，在采集命令中增加 ``--teleop piper_leader``，用 ``i`` 接管 / 交还，``b`` 仍独立标注 critical phase。主臂配置、交接行为和自主成功率的统计方式见 ``piper_human_intervention_design.rst``。下面未启用 teleop 的命令继续执行纯 SFT 验收；HIL 数据通过相同的 ``init -> cache -> train`` 入口处理，无须在采集前计算 latent。

采集 100 条 SFT 回合
--------------------

在本机前台终端执行以下命令。它加载已经核对的 H50 FP32 global_step_50000 权重，保持每次执行 50 个目标的协议，按 30 Hz 名义控制频率运行；每条最多执行 3,600 个目标，达到上限后等待人工评分。

.. code-block:: bash

   cd /home/descfly/code/rlinf_piper
   bash toolkits/piper/eval_stage1.sh rollout \
     --num-episodes 100 --record-dataset \
     --output-dir outputs/piper/sft50k_100_20261013

新采集的输出目录必须是新目录。模型加载完成后会连接相机、使能 follower，并等待右方向键；这条命令会启动真机客户端。每条恢复场景后再按右方向键。默认不执行自动复位轨迹，已有的 ``--reset-after-episode initial|zero`` 可按原来的验收流程显式选择。

中断后可使用 ``bash toolkits/piper/eval_stage1.sh rollout --resume outputs/piper/sft50k_100_20261013 --num-episodes 100`` 续采。配置从原目录恢复，目标 100 条包含已有完整回合；撤销和未完成尝试不计数，新目录编号继续递增。原复位点保持不变，上次复位未完成时先等待 ``r`` 完成复位。完整说明见根目录 ``order.md`` 第 8 节。续采结束后再初始化数据池；已有清单和 cache 保持不变，新增数据使用新的输出路径生成清单和缓存。

自动复位在每条 ``s/f`` 评分、数据和视频保存完成后执行。选择 ``initial`` 时，第一条开始的实测姿态会限制到当前驱动的合法目标范围，``reset_pose.json`` 同时记录原始读数和实际目标，避免零位附近的反馈偏差被当作越界复位目标；所有下发动作仍遵守原有行程限制。选择 ``zero`` 则返回零位，首次启动程序不会提前执行复位。

.. list-table:: 前台终端按键，无需回车
   :header-rows: 1
   :widths: 20 80

   * - 按键
     - 行为
   * - 右方向键
     - 开始下一条回合；采够指定条数后确认结束。
   * - ``b``
     - 成对切换关键阶段：第 1 次进入、第 2 次退出，第 3、4 次标记下一段。按键边界对应下一次下发动作的帧号；终端显示阶段编号和 ON/OFF。
   * - ``s`` / ``f``
     - 停止下发动作并标记成功 / 失败，拍摄终止观测，保存数据后等待下一条。
   * - 左方向键
     - 运行中或等待评分时直接停止并丢弃当前回合；评分后至下一条开始前也可撤销，无需先按成功或失败。
   * - ``q`` / ``Ctrl-C``
     - 提前退出；保留已经完成的回合。当前未评分回合不进入 warmup 标注清单。

一回合可有多段互不重叠的关键阶段。例如双 USB 的两次插入可分别按 ``b₁ → b₂`` 和 ``b₃ → b₄``，两段之间为普通阶段。整个验收回合仍由 SFT reference 执行，``b`` 只改变标注，不中断动作。先约定一致的进入/退出判据；不要因为某次动作看起来会失败就省略标注。可使用 ``--phase-definition '...'`` 写明本轮判据。

成功和失败都保留。``s/f`` 结束任务并评分；如果 ``b`` 为奇数次、最后一段仍开启，就自动把该段延长到回合终止帧，保存完整的起止标注。例如只按一次 ``b``，关键阶段就是该帧到任务结束；按三次则第一段正常配对，第二段延长到结束。完全未标注时 ``critical_phases`` 为空。达到步数上限后仍可按 ``b`` 关闭已开启的阶段，或直接按 ``s/f`` 自动补终点，但不能再开启新阶段。同一帧内重复按键造成的空区间会被忽略；若最后一次 ``b`` 后尚未执行动作便评分，该空段会被丢弃。

标注格式为 ``piper-rlt-episodes-v2``，每个 episode 可以包含：

.. code-block:: json

   {
     "critical_phases": [{"start": 120, "end": 180}, {"start": 300, "end": 360}],
     "terminal_frame": 420
   }

每段有效动作区间为 ``[start, end)``：起点属于该段，终点是段内最后一条动作之后的观测。``terminal_frame`` 独立表示整个任务最后的观测，不要求等于最后一段的终点。终止观测的 action 是 hold 占位值，``complementary_info.action_valid=False``；缓存导入会排除它。

每条回合单独保存到 ``episode_001/dataset/`` 等 LeRobot v3 目录，包含三路 RGB 视频、原始七维状态、实际下发目标、原始预测目标、下发前反馈、帧时间和逐帧 critical 标记。角度/夹爪单位是 degree/mm。默认随采集直接进行后台 H.264 编码，结束时排空队列并保存元数据；可用 ``--no-streaming-video`` 切回 PNG 流程。保存完成前不会进入下一条或开始复位，保存期间可以按左方向键撤销。视频时间轴按名义 FPS，真实采样墙钟时间在 ``complementary_info.wall_time``，推理停顿也保留在 trace 中。

运行中或等待评分时，左方向键直接停止并丢弃当前回合，跳过终止观测补拍和完整视频保存；评分后的保存、复位和等待阶段也可撤销。文件释放后移入 ``discarded/``，同步从统计与 ``annotations.json`` 排除。完成已配置的复位、恢复场景后按右方向键补采，撤销回合不占 ``--num-episodes`` 名额。最后一条仍等待右方向键确认结束，或左方向键撤销。尝试目录编号递增，不覆盖旧文件。采集结束后用更新后的 ``annotations.json`` 执行下面的 ``init``，具体按键和流式编码说明见仓库根目录 ``order.md`` 第 7 节。

``summary.json`` 保存成功率、完成回合数、成功/失败数、录制数及有效阶段标签数；``policy.json`` 保存实际加载的 Stage 1 元数据，``hardware.yaml`` 保存本次硬件配置。每条的 trace、结果、数据集相互独立，上一条结束后已完成文件关闭。控制或录制异常会退出运行，已经完成的回合保持可读。

固定标注与训练/验证划分
-----------------------

采集完成后，下面的命令从回合级标注建立固定 split，输出到配置的 ``annotations`` 路径。它拒绝覆盖已有标注，拒绝漏标、未配对、空区间、重叠或越界阶段；若明确要排除这些回合，可加 ``--skip-unlabeled``，排除名单也会写入结果。排除和训练/验证划分均以完整 episode 为单位，同一回合的多段不会分到不同池，也不会重复计为多个成功样本。

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

缓存格式升级为 ``piper-rlt-cache-v2``，旧缓存需要重新生成。旧 v1 标注仍可读取，保留其原来“单一起点到任务终点”的含义；若希望改为分段标注，需要改成 v2 并明确填写各段边界。采集入口会在评分时为未关闭的最后一段补终点；磁盘上的 v2 标注需要包含明确的起止帧，导入时不推断未完整保存的标注。

``train`` 只加载缓存及小型 RLT MLP，使用本机单张 GPU。训练参数和模型尺寸分别来自同一 YAML 与缓存元数据，避免再次手写 H50、FP32 等 Stage 1 配置。训练池在 CPU 内存中按 episode 等概率、再按 chunk 等概率抽样；验证池独立且不可被训练采样。失败回合的动作供 critic 学习，非人工动作的 BC 目标仍是冻结 reference。

默认训练预算是 5,000 步 actor BC、10,000 步 critic、5,000 步联合更新；联合阶段每 4 步更新一次 actor，总计 6,250 次 actor 更新和 15,000 次 critic 更新。每段尾部不足 C10 的动作窗口按真实有效长度屏蔽。完整的参数与阈值保存到 ``output/config.json``。

退出关键阶段不等于任务终止：窗口结束仍在段内时，critic 的下一状态就是动作后的观测；退出该段时，下一决策状态是下一段的起点，若没有下一段则是实际任务终点。中间由 reference 执行的真实控制 tick 计入 ``discount_steps``，不计入 actor 的动作窗口。成功奖励只在任务终点出现，最后一段与任务结束之间的间隔同样折扣；不会给每一段都发一次成功奖励。缓存同时保留 ``action_end``，用于区分紧随动作的观测和下一决策状态。

可以先运行到固定预算中的某个位置，再恢复；``--until-step`` 不改变总预算或就绪要求：

.. code-block:: bash

   USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup train --until-step 1000
   USE_TF=0 JAX_PLATFORMS=cpu .venv-piper/bin/python -m toolkits.piper.warmup train --resume

支持 ``--config other.yaml`` 和 ``training.batch_size=32`` 这样的显式覆盖。恢复要求训练配置、缓存、设备及实现一致；checkpoint 包含 actor、critic、target critic、两个 optimizer、三个更新计数、验证历史及 CPU/GPU/采样 RNG。``latest.json`` 指向最后一个完整 checkpoint，默认保留最近 3 份。

读取训练输出
------------

``metrics.jsonl`` 每 1,000 步记录一次固定窗口验证，按 episode 等权统计；终止 critic 误差按成功/失败平衡，常数预测 baseline 只用训练集拟合。解码动作与 reference 的偏差采用各 episode 第 95 百分位的最大值，比把长轨迹混在一起统计更保守。关节/夹爪检查是缓存观测上的名义边界检查；后续 tick 用上一条预测作为比较基准，并不代表真实机器人已经通过检查。

``readiness.json`` 记录数据覆盖、actor 偏差、terminal/TD 误差、Q 分歧和范围，以及绑定权重的哈希。固定预算结束且最后三次定期验证全部通过，才会出现 ``ready_for_deterministic_eval=true``。测试用途缓存始终不能放行。``actor/`` 使用现有 ``PiperRLTPolicy`` 可加载的 Stage 2 格式，离线恢复 checkpoint 则供该 warmup runner 使用。

离线报告的 ``ready_for_online_exploration`` 始终为 false。Online replay 的容量管理、离线/在线混合采样、确定性真机试运行验收及回合边界锁存仍是后续 online 接入工作；不能仅将导出的 actor 传给旧 online worker，就宣称已经完成这些条件。

本机验证记录
------------

2026-10-13 多段标注修改后通过 202 项相关模型、数据、Piper 环境与控制回归；后续将奇数次 ``b`` 改为自动延长到终止帧，更新并通过全部 27 项采集评估测试。覆盖 100 条连续评分、多次进入/退出、奇数次标记自动补终点、到步数上限后关闭阶段、视频/状态/阶段对齐、录制故障时保留人工评分、缓存隔离、reference 间隔折扣、动作 mask、分阶段冻结、精确恢复及当前权重的验证门槛。测试均未连接机械臂；此次 GPU 正在运行另一项真机采集，未重复运行完整 Stage 1 权重的集成测试。

``outputs/review/warmup_integration_20261013.json`` 记录了此前单段版本在真实 Stage 1 权重上的测试链路：从既有观测构造明确标为 ``test_only`` 的四条 LeRobot fixture，提取 12 个冻结状态，完成短程三阶段更新与恢复，并用现有推理接口加载导出的 C10 actor。两种 readiness 标志均为 false；这不是多段版本的完整 GPU 集成验证，也不是正式数据训练或成功率验收。

三路 640×480 图像录制测试中，180 步按 30 Hz 运行用了 6.01 秒，图像复制与入队开销的中位数为 2.98 毫秒、95 分位为 4.66 毫秒，结束后的编码和文件关闭用了 3.96 秒。三路视频分别读回 181 帧，包含终止观测。测试使用已有 RGB 观测的平移副本，不包含真实相机与 CAN 延迟，也不代表长回合的编码时间。
