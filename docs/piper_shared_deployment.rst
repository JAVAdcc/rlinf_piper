共享 RLT actor 的手动关键段评测
================================

本入口让操作员用 ``b`` 选择共享 actor，用 ``s/f`` 结束并评分当前关键段后恢复冻结 VLA。普通阶段返回 H50 reference，关键段返回 C10 actor；``--replan-steps 50`` 分别最多执行 50 和 10 步。每次切换都会清空旧队列并拒绝旧异步预测，重新获取当前观测。人工接管始终优先；交还后的等待保持人工归属，直到新 policy 命令实际下发。

权重与测试边界
--------------

首测使用 2026-10-19 的 ``local_segments_shared_20261019/actor``：actor 更新 5,000 次，critic 更新 10,000 次，没有联合更新。小网络权重 SHA256 为 ``51f1b082969e94720d18ddaefbaf83c1d6a4976d5e0bafbcb077949eb4b4b21e``，与原始导出逐字节一致。它需要 Stage1 checkpoint ID ``b1f2800f7b294b4c95834ad60e0ff23d``，norm SHA256 ``75a333d3278d227333ccd3c81ec366973af27c8082755e6aebf8f4516aa89d70``。保留 Stage1 完整 actor 目录；小网络包不能独立替代 VLA。

原离线 readiness 仍为 false。``--allow-offline-actor`` 是显式实验入口，不会把该报告改成通过，也不会启用在线学习。默认加载器继续拒绝 offline-local 格式；实验入口仅接受显式 ``phase_filter=null`` 的共享网络，专用网络仍拒绝。当前没有关键段身份分类或专家自动路由。

已完成真实小网络／冻结缓存特征的 CPU 数值对照，以及同步、异步、人工交还、来源记录和恢复绑定回归。Stage1 完整 GPU 推理和现场动作尚未验收。示例保留已有硬件配置的动作检查；不要把 critic 输入投影理解为推理提案已裁剪。若现场沿用先前的 ``--no-action-limits``，它仅关闭额外检查，底层驱动裁剪仍存在；选择应沿用现场确认的机型、零位和控制设置。

运行顺序
--------

先解压共享 actor 包到 ``checkpoints/piper_shared_prejoint/actor``，使用现场现有的 Stage1 和环境。以下两个 ``infer`` 命令读取录像，不连接机器人；先比较 H50 reference 与 C10 actor 的结果及耗时。

.. code-block:: bash

   cd /home/descfly/code/rlinf_piper
   export PIPER_PYTHON="$PWD/.venv-piper/bin/python"
   export PIPER_STAGE1_PATH="$PWD/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"
   export PIPER_ACTOR_PATH="$PWD/checkpoints/piper_shared_prejoint/actor"
   export PIPER_DATA_PATH="/path/to/one/recorded/episode/dataset"
   bash toolkits/piper/eval_shared.sh infer-reference --frame 300
   bash toolkits/piper/eval_shared.sh infer-actor --frame 300

操作员确认离线预测和现场硬件配置后，自行在推理机启动评测：

.. code-block:: bash

   bash toolkits/piper/eval_shared.sh rollout --output-dir outputs/piper/shared_prejoint_trial

``Right`` 开始；``b`` 进入关键段并切到 actor；``s/f`` 结束并评分该段，同时切回 reference。再次 ``b`` 仍使用同一个共享 actor。``Right`` 停止整条 episode，然后另行 ``s/f`` 评分；``Left`` 丢弃，``q`` 退出。段内重复 ``b`` 不关闭该段。要在结束关键段后立即停止整条任务，可先 ``Right`` 停止动作，再补评段和整条任务，不能把局部 ``s/f`` 当作停止键。

本脚本默认录制、手动复位、一个 episode；可显式增加 ``--num-episodes`` 或启用 ``--teleop piper_leader``。恢复使用公共入口 ``run_policy rollout --resume <目录>``；它核对 Stage1 和实际 actor 字节 SHA、模式及 horizon，不能在原目录替换权重。

记录 ``trace.jsonl`` 中的 action_source、route_generation、预测耗时与 prediction_age。30 Hz 是命令循环的名义频率；尤其 C10 下，每次队列耗尽后才重新推理，并不等于每秒 30 次新 VLA 预测。HIL 等待期间的 hold/human 比例、``b`` 到首个 actor 命令及 ``s/f`` 到首个 reference 命令的延迟需在现场量测。

二分类监视器候选
----------------

``rlinf.robotics.piper_phase_monitor.PiperPhaseMonitor`` 可在 CPU 上读取实验 ``state.npz``、``rgb.npz`` 或 ``state_rgb.npz``，对当前观测输出概率。``update`` 使用固定的 0.8／0.3 阈值和连续两次确认；调用者须按约 5 Hz 采样并在 episode 开始时 ``reset``。它仅输出建议状态，未接入上述机器人入口的自动切换。

候选使用完整 50 条采集数据，按父 episode 保持原 40／10 划分，40 条内部再分 32 条拟合、8 条选择正则系数，未做概率校准。输入仅当前七维状态和／或三路 RGB 的 16×16 缩略图，没有使用动作来源、人工接管、critical 标签、阶段序号或未来帧作为输入。线性模型只是固定场景下的低成本基线，不是论文中微调 VLA 的交接预测模块。数据来自一个 collection 目录，没有可核实的 session 标识；当前验证不足以证明跨 session、跨场景或 actor 改变行为后的泛化。

RL Token 原文第 V 节提出最后短暂微调 VLA，让它利用人工切换标签额外预测何时交给 RL；未给多专家分类器的具体结构、阈值或此二分类基线的配方。原文： https://arxiv.org/html/2604.23073v1#S5 。后续可比较冻结 VLA 特征或轻量视觉编码器，先以旁路提示核对误触发、漏触发和边界延迟，再讨论自动接管。
