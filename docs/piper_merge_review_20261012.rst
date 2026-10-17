Piper 代码合并与 Stage 1 验收记录
=================================

这份记录说明合并版的来源、已修复的问题，以及如何验收指定的 50000-step 权重。代码位于 ``/home/descfly/code/rlinf_piper``；已完成文件校验、GPU 推理对照和相关单元测试，真机成功率等待实际回合统计。后续在线训练方案见 `piper_offline_warmup_design.rst <piper_offline_warmup_design.rst>`_。

运行指定权重
------------

先在本地已有的录制数据上检查推理：

.. code-block:: bash

   /home/descfly/code/rlinf_piper/toolkits/piper/eval_stage1.sh infer --frame 0

这条命令已成功执行，使用 CUDA 加载 FP32 权重并输出 ``[50, 7]`` 动作，不连接机器人。默认数据集是 ``/home/descfly/code/rlinf-piper-check/datasets/usb_iter2_rlt``；它的任务文字是单根 USB，和训练集的双 USB 提示词不同，所以这里只用于检查加载、输入与数值链路。

操作员准备好机器人、相机和场景后，在交互终端执行：

.. code-block:: bash

   /home/descfly/code/rlinf_piper/toolkits/piper/eval_stage1.sh rollout

命令连接三路相机并使能 follower，随后等待右方向键开始每回合。按 ``s`` 标记成功，按 ``f`` 标记失败；两者都会停止当前回合，操作员恢复场景后再次按右方向键。``Ctrl-C`` 中断并退出。默认不执行自动复位轨迹，动作限幅和连接检查沿用本地版本。

默认运行 20 回合，每回合最多执行 3600 个控制目标，以 30 Hz 为目标控制频率，每次执行预测的 50 步后重新规划。推理、相机和等待标注会增加实际用时；达到步数上限后停止下发动作、等待人工评分。对于超时按失败处理，避免选择性忽略困难回合。中断或错误回合单独报告，现有统计只用已评分回合计算成功率。

真机命令使用从服务器训练集 ``meta/tasks.parquet`` 核实的提示词：

.. code-block:: text

   pick up the two USB cables and insert them into the USB port on the outlet.

输出写入 ``outputs/piper/<时间戳>_<随机标识>/``，包括 ``run.json``、``summary.json``、``hardware.yaml`` 和各回合的 ``trace.jsonl``。可追加 ``--num-episodes 50`` 或 ``--replan-steps 10``，但重规划步数改变后应作为另一组实验统计。历史约 45% 的回合数、重规划频率和初始场景分布尚未取得，当前默认 H50 全执行是一个明确记录的起点，不能宣称已复现历史协议。

如果 20 回合中成功 9 次，点估计是 45%，Wilson 95% 区间约为 25.8%–65.8%。它适合初步检查效果，不足以证明与 Evo/LeRobot 等效。两套实现应使用同一提示词、相机身份与顺序、初始场景分布、重规划步数、超时和成功定义；完整双 USB 任务与只从插入阶段开始的成功率分别统计。

定位和校验权重
--------------

指定服务器权重对应如下目录，连接使用 ``ssh -p 27 yiming-aliserver``：

.. code-block:: text

   /home/yiming/workspace/code/piper_stage1_h50_fp32_20261002/runs/20261003_005027_gth59yud/piper_usb_rlt_stage1_h50_fp32_b64_60k/checkpoints/global_step_50000/actor

在本地挂载盘找到了同一权重：

.. code-block:: text

   /media/descfly/新加卷/Evo-RL/outputs/train/piper_usb_rlt_stage1_h50_fp32_b64_60k/20261003_005027_gth59yud/checkpoints/global_step_50000/actor

已把推理需要的权重、归一化统计和模型元数据复制到内部磁盘，运行入口默认使用：

.. code-block:: text

   /home/descfly/code/rlinf_piper/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor

``model_state_dict/full_weights.pt`` 共 16,401,034,093 字节，本地原件、复制件与服务器 SHA256 一致：

.. code-block:: text

   4a646cad16ca969b845a730cf0477b04fb0f18434925ed6963eae053d1617309

``checkpoint_id`` 是 ``b1f2800f7b294b4c95834ad60e0ff23d``。``piper_inference.json`` 和 ``assets/piper/norm_stats.json`` 也已逐一校验。出处保存在权重目录的 ``origin.json``。本地副本只含推理资产；分布式优化器与 SFT 数据游标状态仍保留在原始完整 checkpoint 中。

之前容易混淆的 ``/home/descfly/code/RLinf/ckpt/actor`` 对应另一个 checkpoint id，``/home/descfly/actor`` 则是不完整副本；本次入口均不使用它们。

代码来源和合并原则
------------------

本地来源为 ``/home/descfly/code/RLinf``，基准 commit 是 ``807e5fdd836f0c3671fd351f65faf3bce5ddaf35``，包含原有未提交改动。远端来源为 ``/home/yiming/workspace/code/piper_stage1_h50_fp32_20261002/repo``；复制的远端目录没有 Git 元数据，不能给它推定 commit id。

新目录是独立 Git checkout，分支为 ``integration/piper-local-server``。合并结果保留为可审阅的工作区改动。源代码快照位于 ``.merge_sources/local`` 与 ``.merge_sources/server``，逐文件取舍和哈希见 `piper_merge_manifest.json <piper_merge_manifest.json>`_。两个原始代码目录保持原样。快照、权重和运行输出通过本仓库的 ``.git/info/exclude`` 排除在 Git 提交之外。

本机 ``.venv-piper`` 链接到既有 ``/home/descfly/code/RLinf/.venv-piper-rollout``，复用了安装好的环境，未改动其依赖。运行脚本显式设置新仓库的 ``PYTHONPATH``；移动目录或删除原环境时需重新配置 ``PIPER_PYTHON``。

对两份源文件清单发现 36 条差异，其中包含服务器的 pytest 缓存。实际合并按以下职责处理：

.. list-table:: 合并取舍
   :header-rows: 1
   :widths: 24 76

   * - 范围
     - 处理结果
   * - SFT 精度与权重加载
     - 采用服务器的先设参数 dtype、后加载权重流程，避免 FP32 checkpoint 先写入 BF16 参数再转回 FP32 而丢失数值。
   * - SFT 参数冻结与视觉编码器
     - 保留 ``train_expert_only`` 生效逻辑，以及 SigLIP FP32 stem 的模块调用方式，保证 FSDP 包装与梯度 hook 正常参与。
   * - 数据和恢复训练
     - 保留 LeRobot 按所需行读取的优化、有限 epoch sampler、逐 rank 数据游标/RNG 恢复和错误同步；不把缓存文件带入代码。
   * - 学习率与历史配方
     - 保留 ``evo_cosine`` 的绝对步数调度和衰减终点后的最小学习率，保留服务器三份 Piper 实验配置。
   * - 图像增强
     - Piper 默认关闭 crop/rotation，保留颜色增强；新增显式 ``spatial_augmentation`` 开关，其他 OpenPI profile 仍默认开启空间增强，避免服务器局部适配改变其他任务。
   * - 本地部署与采集
     - 保留 USB 序列号选相机、follower ``high_follow``、连接顺序、动作检查、交互评分、复位工具及对应 mock 测试。
   * - 本地其他改动
     - 保留 RECAP 等原有本地工作；通用 ``piper_rlt_stage1`` 的 ``train_expert_only: false`` 保持原设置。
   * - 测试
     - 合并双方数据与模型测试；服务器调度器测试归入已有 ``test_fsdp.py``，避免独立的修复专用测试文件。

服务器历史配置的 ``video_backend: pyav`` 原样保留。本机 torchvision 版本已没有 LeRobot 旧 PyAV 路径使用的 ``VideoReader``；录制数据推理验证使用 ``torchcodec``，本机若运行这些 SFT 配置也应显式覆盖 ``data.video_backend=torchcodec``。这不是模型权重或精度差异。

关键阶段与在线训练审查
----------------------

Piper 原生在线环境已经支持轨迹关键阶段。``rlinf/envs/real/piper/policy.py`` 中，``a`` 开始回合，``b`` 标记进入关键阶段，``c`` 成功，``q`` 失败。关键阶段标记在回合内锁存，下一动作 chunk 才能使用新标记，reset 后清除。这个键位协议属于在线训练，与上面 Stage 1 验收入口的右方向键、``s/f`` 不同。

审查发现并修复了两个会影响实验含义的问题：

* 真实机器人 route 原来只根据 ``b`` 切换 actor，忽略 learner warmup 更新数。现在开启 ``algorithm.rlt_schedule.enable`` 后，更新数不足时执行 reference；关键阶段的 reference 动作仍标记为可收集。
* 真实机器人 replay 原来没有按 ``record_transition`` 去掉任务前缀。现在配置 ``algorithm.replay_buffer.critical_phase_only: true`` 时，只加入关键阶段 chunk，保留原始 ``next_obs`` 和每个 rollout epoch 对齐后的终止标记。Piper Stage 2 模板已显式开启这个过滤。

``record_transition`` 表示“该数据属于训练阶段”，``actor_switch`` 表示“当前执行 actor”。两者已经解耦；在 warmup 中按 ``b`` 会收集关键阶段 reference 数据，不会提前执行 actor。新增边界与 replay 测试覆盖这一点。

现有在线实现还存在以下限制，下一次在线实验必须据此配置：

* ``piper_rlt_stage2.yaml`` 是本地原有 H10/BF16、4 次去噪、无离散 state 的模板。指定权重要求 H50/FP32、10 次去噪、离散 7 维 state；旧模板不能直接作为这次 H50 的在线入口。具体字段见 warmup 设计。
* ``train_actor_steps`` 实际表示 replay 达到多少条 trajectory 才训练 actor，不是只训练 critic 的 optimizer 步数。``rlt_schedule`` 目前也只按样本数和更新数调度，warmup 中仍可能更新 actor。
* 更新数门槛已经接通真实机器人 route，但没有验证指标门槛，也没有回合开始时锁存的 readiness。线上权重同步若在回合中跨过门槛，下一 chunk 就可能切换。
* SAC checkpoint 路径没有完整保存和恢复 ``update_step``、RLT 调度计数及 readiness；不能把一个 actor 权重文件当作完整的离线到在线恢复包。
* 当前标准 critic target 按 ``terminations`` 决定是否 bootstrap。人工失败会终止；任务超时和真正外部中断应分别定义，不能混成一个无限续接状态。
* 当前 Stage 2 默认 ``gamma=0.96`` 按控制 tick 使用。30 Hz 下 10 秒后的奖励权重约为 ``0.96**300 = 4.8e-6``，会弱化较长插入过程的反馈。设计中给出了按时间尺度选择折扣的方法。

验证证据与边界
--------------

在 RTX 5090、PyTorch 2.11.0+cu128 上，从录制数据读取 frame 0、1500、50000，固定采样随机种子，分别通过远端原版快照和合并版加载同一个 checkpoint。frame 0 另重复一次，共 4 次输出逐值相同，最大绝对差为 0。所有输出为有限 ``[50, 7]``，加载后的 729 个 state tensor 均为 FP32，视觉 stem、位置编码、语言模型、动作输出和 RLT 模块的抽查权重与磁盘值完全一致。

合并版冷调用约 0.526 秒，后续三次为 0.183–0.199 秒，峰值 PyTorch GPU 分配约 15.62 GiB。这些是录制观测的推理耗时，不含实时相机、机器人通信和在线训练。FP32 路径保留原实现的 TF32 设置；此处验证的是两份代码的一致性，不代表与 Evo/LeRobot 已做数值等价验证。

测试结果为 175 项通过：模型/数据/调度器 91 项，机器人 mock 和验收/复位相关 84 项。所有改动 Python 文件 Ruff 检查、格式检查及 ``git diff --check`` 通过。四份 SFT 配置已完成 Hydra 解析。未执行真实八卡 SFT 恢复、在线 RL 训练或真机动作。

复现命令和证据保存在 ``outputs/review/``：

.. code-block:: bash

   cd /home/descfly/code/rlinf_piper
   USE_TF=0 JAX_PLATFORMS=cpu PYTHONPATH="$PWD" .venv-piper/bin/python -m pytest -q \
     tests/unit_tests/test_models.py tests/unit_tests/test_data.py \
     tests/unit_tests/test_fsdp.py -k 'piper or openpi or evo_scheduler'
   USE_TF=0 JAX_PLATFORMS=cpu PYTHONPATH="$PWD" .venv-piper/bin/python -m pytest -q \
     tests/unit_tests/test_real_env.py tests/unit_tests/test_robotics.py \
     tests/unit_tests/test_piper_evaluation.py tests/unit_tests/test_piper_reset.py -k piper

两组分别在独立 Python 进程中运行，避免现有测试套件的依赖 stub 在收集阶段相互影响。主要日志为 ``final_core_tests.log``、``hardware_mock_tests.log``、``inference_parity.json``、``merged_inference.json``、``server_inference.json`` 和 ``infer_cli.log``。这些证据支持开始人工验收，不构成约 45% 成功率的实测结果。
