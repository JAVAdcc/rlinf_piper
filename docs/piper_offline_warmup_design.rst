Piper RLT 离线 Warmup 与 Actor 启用条件
=======================================

这个方案把第一轮离线训练固定为 20,000 个 learner 更新，并要求留出集验证和小规模真机检查都通过后，才在轨迹关键阶段启用带探索噪声的 actor。这里的数值是本任务的实验起点，不能保证成功率或真机安全。离线缓存、分阶段训练、指标报告及完整离线恢复已实现；online 数据池、真机验收和回合边界锁存仍待接入。实际入口与采集流程见 ``docs/piper_offline_warmup.rst``。

2026-10-13 实现：通用 ``examples/embodiment/train_offline_rl.py`` 仍只接受 ``offline_iql``；Piper 的离线入口采用 ``python -m toolkits.piper.warmup init|cache|train``，配置为 ``examples/offline_rl/config/piper_rlt_warmup.yaml``。采集入口支持逐 tick LeRobot 录制、``b`` 标记关键阶段和 ``s/f`` 评分。下文网络测速使用合成特征，不能代表真实数据的训练质量。

依据与现有实现的差异
--------------------

`RL Token 论文 <https://arxiv.org/html/2604.23073v1>`_ 冻结适配后的 VLA/RL token，以 reference 动作引导小型 actor-critic，在人工指定的任务关键阶段训练。论文 Algorithm 1 的 warmup 主要按环境步数选择 reference 行为，并未给出适用于 Piper 的离线就绪判定。因此不能把论文中的 warmup 步数直接当作 critic 已可靠的证明。

`TihayaKousaka/rlt 运行说明 <https://github.com/TihayaKousaka/rlt/blob/main/rlt_online_rl/README.md>`_ 是非官方复现，给出了收集后 20,000 次更新以及 learner/actor version 协调的实现例子。其 `离线训练入口 <https://github.com/TihayaKousaka/rlt/blob/main/rlt_online_rl/scripts/offline/README.md>`_ 支持 replay 预训练和验证。这些代码可借鉴流程，但动作语义、控制频率和权重格式不能直接移植。检查其 ``offline_train_from_replay.py`` 还发现验证集按随机 transition 行拆分，重叠 chunk 可能泄漏；Piper 应按完整 episode 拆分。

`Cal-QL 官方实现 <https://github.com/nakamotoo/Cal-QL>`_ 提供了离线预训练后在线微调的 value 校准参考。当前 RLT 的 twin-Q 加 BC 不能称作 Cal-QL；如果出现明显离线 Q 高估，可另做算法对照，先保持本次主实验可解释。

关键阶段应有一个可复现的定义
----------------------------

针对 USB 任务，先固定“已抓起目标线缆、连接器进入预先约定的插入邻域”作为人工进入判据候选，结合录像确定统一的退出判据。采集入口的 ``b`` 成对切换进入/退出，段内动作进入离线 replay；所有边界和最终成功/失败都写入数据。判据一旦选定，应在 baseline、数据标注和 online 中一致使用，不能根据本次动作是否看起来容易成功临时切换。

双 USB 可分别标记两个插入窗口：``b₁–b₂`` 为第一段，``b₃–b₄`` 为第二段。Piper SFT 采集和离线缓存已支持显式阶段 id 与起止帧；若缺失最后一个结束键，采集入口在 ``s/f`` 评分时将最后一段自动延长到任务终止帧，前面已配对的区间保持原边界。通用 online 键盘 wrapper 仍是原来的一次进入锁存机制，后续 online 接入需要同步成对切换与段末动作截断；当前采集入口不经过这个 wrapper。

完整任务成功率、进入关键阶段的比例、进入后成功率、关键阶段耗时分别报告。只从关键阶段初始化的试验不能直接和 Stage 1 的完整任务约 45% 比较。

先建立能用于离线训练的数据
--------------------------

服务器训练集是 97 条成功轨迹、93,155 帧、30 Hz。只使用成功演示不足以验证 critic 是否能识别失败。另在本机 ``/home/descfly/code/rlinf-piper-check/datasets/usb_iter2_rlt`` 检查到 101 条 episode，含 71 成功、30 失败；但其任务文字为单 USB，而且没有关键阶段标注。需要先确认录像确实符合双 USB 任务与动作定义，再标记关键阶段，不能直接把它当成已就绪的离线 replay。

验收工具的 ``trace.jsonl`` 用于诊断和评分，并不自动包含完整 replay 所需的全部特征、下一状态和标签。离线准备工具应读取原始图像/状态/实际下发动作和回合记录，生成以下数据与清单：

.. list-table:: 每个关键阶段 transition 的数据约定
   :header-rows: 1
   :widths: 30 70

   * - 内容
     - 约定
   * - 固定模型
     - 绑定本次 50000-step checkpoint、归一化统计、任务提示词和相机映射的哈希；冻结 VLA 与 RLT，推理时关闭训练增强。
   * - 状态与 reference
     - ``z_rl[2048]``、归一化 ``proprio[7]``、``ref_chunk[50,7]``，以及真实下一状态对应的同组特征；缓存 reference 采样种子。
   * - 实际行为
     - ``actions[10,7]`` 使用经过同一 codec 编码的实际执行动作，附 ``valid_steps``、执行时间和 human/reference/actor 来源；预测 reference 不能冒充发生过的行为。
   * - 回报与终止
     - 整个任务成功末步为 1，失败为 0；``terminated`` 仅指连接到任务终点的 transition。记录下一决策状态、紧随动作的 ``action_end`` 和含 reference 间隔的 ``discount_steps``；任务结束禁止 bootstrap。
   * - 分组与标记
     - ``episode_id``、session id、起始帧、phase id、所有关键阶段起止帧、human intervention 标记、成功/失败和数据来源哈希。

动作窗口不得跨越 episode、阶段终点或 reset；末尾不足 10 步的窗口需用有效长度屏蔽。若暂不实现长度屏蔽，可把最后一个完整窗口的起点前移，使其恰好结束于终止 tick；整个关键阶段不足 10 步时仍必须支持有效长度，不能丢掉这些短成功回合，也不能用重复 padding 制造额外真实动作。这个导入接口需要回归测试，尤其是最终成功 tick、下一观测和折扣的对齐。

按 episode 与采集 session 分组划分训练/验证，成功和失败都留出。以可用的 101 条为例，先固定 10 成功、10 失败作为验证，其余训练；若场景/session 数量少，应优先整组留出并补数据。训练集最低建议各有 20 条成功和失败、至少 600 个有效关键阶段 transition；这些是本轮数据要求，不是 600 条高度重叠窗口就能满足的独立经验保证。采样先选 episode 再选窗口，避免长失败轨迹垄断 batch。

预计算特征并管理数据池
----------------------

离线 warmup 建议先生成冻结 Stage 1 的特征缓存，再卸载 VLA，只训练约 207 万参数的 actor-critic。已有 ``Pi0Eval.extract_rlt_obs`` 一次调用会复用视觉/语言 prefix，输出 ``z_rl``、``proprio`` 和 ``ref_chunk``；缓存这三个输出即可，不需要保存整个 VLM prefix 或 KV cache。``z_rl`` 来自冻结编码器，reference 动作还包含 flow 采样，因此应记录采样种子。首轮为每个状态保存一个 reference 样本；将来若扩大 reference 采样覆盖，要另记版本和额外计算预算。

按 episode 存储状态表与 transition 索引。状态表每行包含 ``state_id``、原始帧索引和三个特征；transition 使用 ``state_id/next_state_id`` 引用，另存实际动作、奖励、有效长度、终止标记和数据来源。重叠窗口共享同一状态的缓存；末端观测也必须有对应特征。首轮 FP32 状态向量为 ``(2048 + 7 + 50*7)*4 = 9620`` 字节，十万个独立状态约 0.96 GB，另外计算实际动作和索引开销。直接在每个 transition 中复制当前/下一特征会接近双倍占用。

缓存清单绑定 Stage 1、norm、任务文本、相机映射、预处理、数据源与抽帧规则的哈希。episode shard 写完并校验后才进入可采样索引，支持逐 episode 断点续算；绑定不同的缓存拒绝混用。底层可使用连续数组和 memory map，内存足够时直接载入 RAM，再预取 batch 到 GPU。训练/验证 split 在抽窗口之前固定，增加新窗口也不能跨 split。

拟采用以下三个逻辑数据池；这里描述的是待实现的管理规则：

.. list-table:: 数据池与采样
   :header-rows: 1
   :widths: 24 40 36

   * - 数据池
     - 保留与用途
     - 采样方式
   * - 离线训练池
     - 固定的数据版本，保留合格的成功、失败和人工纠正轨迹。原始文件与特征长期落盘，不受在线窗口淘汰影响。
     - warmup 时 batch 64 全部来自此池；先选 episode 再选 transition，记录各来源与结果比例。失败轨迹用于 critic；失败动作不自动作为可信 BC 标签。
   * - 在线训练池
     - 只接收关键阶段数据，带 actor 版本与 reference/actor/human 来源。持续落盘，内存训练窗口首轮按 100,000 个 transition 限制，按完整 episode 淘汰；这是新容量设计，不是当前 ``cache_size`` 的含义。
     - 切到 online 后初期每 batch 32 个离线、32 个在线样本；在线池未达到预设样本/episode 门槛时继续使用离线池。人工纠正可提升采样权重，但通过唯一 transition id 避免重复记账。
   * - 固定验证池
     - 按 episode/session 留出，固定清单，永不写入训练 replay。评估期间采集的真实回合另存为评估数据。
     - 每 1,000 次更新按同一窗口与随机种子验证，episode 等权；失败和成功分别报告。

现有 ``ReplayBufferDataset`` 已支持 replay/demo 两池各抽一半，可以复用这一批次组合方式；当前 ``TrajectoryReplayBuffer`` 则在最近 N 条 trajectory 的所有 transition 中均匀采样，长 trajectory 的权重更大，还不是上述按 episode 分组的采样。一个 trajectory 是 rollout 存储单元，不保证等于一个完整 episode。

``piper_rlt_stage2.yaml`` 当前的 ``cache_size=200`` 和 ``sample_window_size=200`` 以 trajectory 为单位，``auto_save=false``；这是在线 replay 的内存缓存，不能替代离线 latent 预计算，也不能保证长期数据保留。实现时需要明确区分磁盘数据全集、可采样窗口和内存容量，并保存 sampler RNG、窗口索引、数据版本和 learner 状态以支持恢复。

从实际执行的控制时间定义折扣
----------------------------

当前 RLT 对 chunk 内奖励逐 tick 折扣，再用 ``gamma**chunk_length`` bootstrap。30 Hz 下 ``gamma=0.96`` 的回报半衰期约 0.57 秒，10 秒后的奖励权重仅约 4.8e-6。对需要数秒插入的阶段，这个尺度通常过短。

建议首轮采用每个 30 Hz tick 的 ``gamma=0.995``，10 步 chunk 的 bootstrap 系数约为 0.951，回报半衰期约 4.61 秒。该值应结合实测关键阶段时长调整；若希望半衰期为 T 秒，可用 ``gamma = 0.5**(1/(30*T))``，并把选择记入实验记录。

对实际执行长度为 L 的窗口，若没有离开当前关键段，下一决策就在 L 个 tick 后；若退出该段，下一决策是下一关键段的起点，中间使用实际记录的 reference 行为连接。最后一段则连接到实际任务终点。令 ``delta = next_frame - current_frame``，其中包括 reference 间隔；稀疏任务回报的 target 为 ``reward * gamma**(delta-1) + (1-terminated) * gamma**delta * Q_target(next_state, next_action)``，只有连接到成功任务终点时 reward 为 1。每个动作窗口最多包含本段内 L 条实际动作，下一动作窗口也按下一段的有效长度屏蔽，不对未参与训练的普通阶段状态调用 Q 作为 bootstrap 目标。

约定的任务时间预算耗尽仍由操作员评分；外部取消、掉线的未完成回合不进入当前离线清单。计算时间带来的停顿另行记录；若改成按墙钟时间优化，需要一致地修改数据和 target，不能只改 YAML 数值。

固定 20,000 次离线更新
----------------------

“一步”统一定义为一次 learner minibatch 更新，不是机器人控制 tick、回合数或 replay trajectory 数。记录三个独立计数：``offline_step``、``actor_optimizer_steps`` 和 ``critic_optimizer_steps``。首轮 batch size 64，学习率从现有 actor/critic 的 3e-4 开始，tau 0.005，种子 1234；配置、归一化和数据 split 随 checkpoint 保存。

.. list-table:: 首轮固定训练预算
   :header-rows: 1
   :widths: 20 22 58

   * - 离线更新区间
     - 更新对象
     - 目标与限制
   * - 1–5,000
     - Actor BC
     - 只蒸馏 reference，可信 human intervention 使用真实纠正动作。Q 权重为 0，BC 权重为 5，reference dropout 为 0；critic 暂不优化。结束后保存固定 BC actor。
   * - 5,001–15,000
     - Critic
     - 冻结已拟合的 BC actor，使用 twin-Q 的 TD 目标和对应 target 网络训练 critic。下一动作来自 BC actor，不从随机初始化 actor 生成。每步更新 critic target。
   * - 15,001–20,000
     - Critic 与 Actor
     - 每步更新 critic，每 4 步更新 actor。Q 权重线性从 0 升至 0.1，BC 权重从 5 降至 2，reference dropout 从 0 升至 0.5。只在离线数据上优化。

先拟合 actor、再固定它训练 critic，可避免随机 actor 的下一动作污染初始 TD target。最后阶段才允许 Q 梯度推动 actor 离开 reference。这个分段是针对当前工程状态提出的方案，不是论文规定的 20k 配方，也不是简单设置现有 ``train_actor_steps=5000`` 就能实现。

离线 20k 完成前禁止下发 actor 动作；若同时收集数据，仍使用 Stage 1 reference。离线 BC 5k 不应冒充 critic 已更新 5k，也不应直接写进当前 online ``update_step``。离线完成记录与 online learner 计数应分别存储，由 readiness 决定是否启用 actor。

训练速率与缓存时间
------------------

2026-10-13 在本机 RTX 5090 上，对原生 ``RLTMLPPolicy`` 做了分阶段微基准：batch 64，FP32，允许 TF32，CPU 缓存 4096 条合成特征，每阶段预热 100 次、计时 2000 次。计时包含抽 batch、CPU 到 GPU 复制、前向/反向、Adam、梯度裁剪、critic target 更新和 loss 标量同步；不包含真实 replay sampler、视频、VLA、读盘、Ray/FSDP、验证或 checkpoint。结果如下：

.. list-table:: 缓存输入下的网络计算速度
   :header-rows: 1
   :widths: 38 31 31

   * - 阶段
     - 更新/秒
     - 毫秒/更新
   * - Actor BC
     - 1114
     - 0.90
   * - Critic
     - 606
     - 1.65
   * - 联合阶段，每 4 次 critic 更新一次 actor
     - 510
     - 1.96

按上述速率外推 5k/10k/5k 预算约为 31 秒核心计算；这不是完整训练作业的 ETA，实际数据采样和验证可能成为主要开销。测速脚本及完整报告位于 ``outputs/review/benchmark_cached_rlt.py`` 和 ``outputs/review/warmup_compute_benchmark_20261013.json``，没有保存训练权重。

冻结 VLA 的特征生成成本另算。上一轮单观测完整 ``z_rl/ref_chunk`` 推理约 0.18–0.20 秒，以 0.19 秒粗估，每万个独立状态约 32 分钟。服务器 93,155 帧若逐帧计算约 4.9 小时；若按 C10 的间隔缓存所需状态、去重当前/下一状态，则约 9,300 个状态、30 分钟，再按关键阶段实际占比减少。这里未计视频解码和写盘，也未测批量提取提速，属于单样本速率外推。更密的重叠窗口会增加缓存成本，不能把 transition 数和独立状态数混为一谈。

验证决定能否进入真机试运行
--------------------------

每 1,000 步在固定的留出 episode 上验证，按 episode 等权聚合，关闭探索噪声与 reference dropout。先固定指标定义和阈值，再训练；不通过时应保留失败报告，不能训练结束后修改阈值来宣布通过。以下数值是首轮待校准阈值，要先用原始 VLA 的重复推理误差和实际动作量级检查是否合理。

.. list-table:: 首轮离线放行条件
   :header-rows: 1
   :widths: 24 52 24

   * - 检查对象
     - 条件
     - 目的
   * - 数据与绑定
     - 训练/验证 episode 不重叠；关键阶段标签与终止语义完整；模型、norm、任务、相机和 codec 哈希一致；成功/失败覆盖满足上述要求。
     - 阻止数据泄漏和配置误用。
   * - Actor 模仿
     - 无 human 标记的验证帧，actor mean 对对应 reference 的归一化动作 RMSE ≤ 0.05；human 帧另对纠正动作统计，不强迫回到错误 reference。
     - 检查初始化是否仍围绕可信行为。
   * - 解码后动作
     - 非 human 帧相对 reference 的逐关节绝对偏差 p95 ≤ 2 度、夹爪 p95 ≤ 3 mm；全部有限，既有物理动作限制违规数为 0。完整逐维分布也要保留。
     - 防止归一化误差掩盖实际动作偏差。
   * - Critic 终止回报
     - 对实际行为的终止窗口，以窗口内已观测的折扣回报为目标；成功/失败两组等权计算 RMSE ≤ 0.25，并比“训练集终止回报均值”这个固定预测基线至少降低 20%。两组误差分别报告。
     - 终止窗口不含未来 policy，目标可直接核实；恒为 0 的 critic 无法靠大量失败样本通过。
   * - Critic TD 一致性
     - 使用相同非终止验证窗口/采样种子计算 twin-Q 的 TD RMSE ≤ 0.25；验证/训练 TD RMSE 比值 ≤ 2（分母下限 0.01），最近三次没有连续恶化。
     - 检查学习稳定性与过拟合；TD 自洽本身不能证明 Q 正确。
   * - Critic 稳定性
     - 在实际行为与候选 actor 动作上，twin-Q 绝对差的均值 ≤ 0.10；99% 的 Q 在 [-0.05, 1.05] 内，损失/梯度/动作均有限。按成功/失败、阶段位置分别检查分布。
     - 检查不稳定和明显超出当前回报范围的估值。
   * - 连续通过
     - 至少完成 20,000 步，最近 3 次固定验证全部满足条件；不能凭一次低 loss 自动放行。
     - 避免瞬时波动触发切换。

上述 Q 范围依赖“一回合仅一次终端正奖励”的约定，不能套到持续奖励或惩罚后的版本。折扣 Q 也不是成功概率，不应直接算概率分类的校准分数。留出轨迹的 Monte Carlo return 对应原行为 policy；它可以作为行为评估 critic 的诊断，但不能直接要求改进后 actor 的 Q 等于旧行为 return。小 TD error、两个 Q 相近仍可能共同高估，所以离线验证只准许进入小规模确定性试运行。

如果 20k 结束后只是尚未收敛，可按预先约定追加两个 5k 区间，最多 30k；追加阶段保持 Q/BC/dropout 终值，重新要求连续 3 次验证通过。数据缺失败、绑定错误、动作越界或验证恶化时，停止延期并补数据/修正实现，不能靠无限 warmup 解决。

通过后的在线切换
----------------

先在保持相同关键阶段定义的 5 个回合中执行确定性 actor mean，操作员在场，记录相对 reference 的偏差、动作检查失败和任务结果；发生异常即回到 reference 并保留数据。这只是发现明显问题的试运行，5 回合不能证明成功率。通过后，才考虑启用原模板的小 Gaussian 噪声 ``fixed_std=0.002``，其单位是归一化动作，需同时检查解码后的实际量级。

每个新回合开始时锁存 ``actor_ready`` 和 actor 版本。建议的最终执行条件是 ``critical_phase AND episode_actor_ready``；其中 ``episode_actor_ready`` 要求离线训练和指标通过、确定性试运行完成、权重与验证报告哈希匹配。回合中收到的新权重或 readiness 留到下一回合使用，避免在接触动作中突然换 policy。``b`` 仅切换阶段状态，不能越过 readiness；退出该段后应恢复 reference。

使用同一批初始条件分别统计 Stage 1、确定性 actor、探索 actor 的表现；探索数据用于训练，评估回合关闭噪声且不纳入本轮训练。保存完整任务和关键阶段两套分母，保留失败、超时、中断和接管的数量。

实现与恢复训练的接口
--------------------

离线到在线需要导出 actor、twin critic、target 网络、优化器、RNG、数据 split 与 replay 指纹、三个更新计数，以及不可混用的训练/验证报告。``readiness.json`` 至少绑定 Stage 1 SHA256、Stage 2 权重 SHA256、norm/codec/data/split 哈希、phase 定义、预算、阈值、最近三次验证和确定性试运行结果。模型或数据变化就重新验证；恢复时没有报告默认继续 reference。

H50 在线配置应显式设置以下字段；这是配置核对表，不是当前可直接启动的完整配置：

.. code-block:: yaml

   rollout:
     rlt_feature_model:
       precision: fp32
       num_steps: 10
       num_action_chunks: 50
       openpi:
         config_name: pi05_piper_h50_state
         action_horizon: 50
         action_chunk: 50
         num_steps: 10
         discrete_state_input: true
         state_indices: [0, 1, 2, 3, 4, 5, 6]
   actor:
     model:
       precision: fp32
       z_dim: 2048
       proprio_dim: 7
       action_dim: 7
       num_action_chunks: 10
       ref_num_action_chunks: 50
   algorithm:
     gamma: 0.995
     replay_buffer:
       critical_phase_only: true

模型路径和 norm 绑定本次已校验的 50000-step checkpoint，其余 RLT profile 字段与 checkpoint 元数据一致。reference 预测 50 步，online 执行前 10 步再重新观测；因此在比较在线改善之前，还需做一组同为 C10 的纯 reference 对照，避免把重规划频率变化误认为 RL 收益。

实施顺序是：完成 Stage 1 人工验收并同步收集带关键阶段标签的成功/失败数据；用已实现的 episode 分组缓存和离线 runner 训练并读取验证报告；补齐 online 数据池、checkpoint 接续与回合锁存；再进行确定性试运行和在线探索。离线 runner 的 checkpoint 状态恢复已经实现，但不等于旧 FSDP online worker 已支持这一交接。
