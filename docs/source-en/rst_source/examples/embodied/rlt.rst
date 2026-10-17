RL Token: Bootstrapping Online RL with Vision-Language-Action Models
====================================================================

**RL Token: Bootstrapping Online RL with Vision-Language-Action Models** trains
a compact reinforcement-learning policy on top of a frozen VLA feature model.
In RLinf configs and code, this workflow is abbreviated as **RLT**. It has two
stages:

1. Train a VLA checkpoint together with an RLT token transformer on
   demonstration data.
2. Freeze that feature model and train a lightweight off-policy actor-critic
   policy using the extracted RLT state.

The checked-in examples target Franka peg insertion and the ManiSkill
``PegInsertionSideWideClearance-v1`` joint-control simulation. The pipeline is
not tied to either task. Reuse the same two-stage structure when the
demonstrations, environment config, action shape, state semantics, and OpenPI
dataconfig stay aligned.

Official project page: `Precise Manipulation with Efficient Online RL <https://www.pi.website/research/rlt>`_.

Overview
--------

RLT separates representation learning from online RL control.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Stage 1
      :text-align: center

      VLA SFT + RLT token transformer

   .. grid-item-card:: Stage 2
      :text-align: center

      Compact off-policy actor-critic

   .. grid-item-card:: State
      :text-align: center

      ``z_rl`` + proprio + reference chunk

   .. grid-item-card:: Deployment
      :text-align: center

      Franka real robot / ManiSkill simulation

| **You'll do:** prepare demonstrations -> train Stage 1 -> point Stage 2 at
  the Stage 1 checkpoint -> launch actor-critic training -> monitor replay-buffer and task
  success metrics.
| **Prerequisites:** `OpenPI π₀.₅ base weights <https://huggingface.co/lerobot/pi05_base>`__, plus the environment for your example—either :doc:`Franka real-world <../embodied/franka>` or :doc:`ManiSkill simulation <../embodied/maniskill>`.

Provided Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - Stage
     - Config
     - Purpose
   * - Franka Stage 1
     - ``examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml``
     - SFT π₀.₅ together with the RLT token transformer on Franka demonstrations.
   * - Franka Stage 2
     - ``examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml``
     - Run real-world RLT actor-critic training with the frozen Stage 1 feature model.
   * - ManiSkill Stage 1
     - ``examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml``
     - Jointly train the ManiSkill OpenPI base policy and RLT token transformer.
   * - ManiSkill Stage 2
     - ``examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml``
     - Run simulated RLT actor-critic training with automatic ``rlt_policy_switch`` and transition replay.
   * - ManiSkill Stage 2 (TD3)
     - ``examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml``
     - Run the simulated TD3-MLP variant with the same frozen Stage 1 feature model and replay route.

Installation
------------

1. Clone RLinf Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   # For mainland China users, you can use the following for better download speed:
   # git clone https://gh-proxy.com/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Docker Image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
      # For mainland China users, you can use the following for better download speed:
      # infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

Please switch to the OpenPI virtual environment via the built-in ``switch_env`` utility:

.. code:: bash

   source switch_env openpi

**Option 2: Custom Environment**

.. code:: bash

   # For mainland China users, you can add the `--use-mirror` flag to the install.sh command for better download speed.

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

RLT uses ``model_type: openpi`` in the configs below. The install target
is still ``openpi`` because the vendored PyTorch model shares the OpenPI runtime
and the RLT Stage 1 dataloader keeps using the OpenPI data pipeline for
ManiSkill and real-world compatibility.

How RLT Works
-------------

Stage 1: Learn the RLT Feature Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 starts from a VLA checkpoint and optimizes two objectives from the
same demonstration batch:

1. The normal VLA action objective, reported as ``vla_loss``.
2. The RLT token objective, reported as ``rlt_loss``.

The total Stage 1 loss is:

.. code:: text

   total_loss = rlt_loss + rlt_alpha * vla_loss

The OpenPI model exposes the VLA prefix hidden states. The RLT token
transformer reads those prefix states and produces a compact vector ``z_rl``.
Stage 2 uses ``z_rl`` as the learned RL representation rather than training the
actor-critic directly on image observations.

Important Stage 1 fields:

.. code:: yaml

   # examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml
   data:
     train_data_paths: "/path/to/data"

   actor:
     model:
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"
       model_type: "openpi"
       is_lora: False
       model_path: "/path/to/model"
       num_action_chunks: 20
       action_dim: 7
       num_steps: 4
       add_value_head: False
       openpi:
         task: sft
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         # Network prediction horizon (Pi0Config.action_horizon). Matches official
         # TrainConfig for config_name pi05_franka_state. Distinct from action_chunk
         # / num_action_chunks, which is the env-executed (RLT) window.
         action_horizon: 20
         action_chunk: ${actor.model.num_action_chunks}
         action_env_dim: ${actor.model.action_dim}
         num_steps: ${actor.model.num_steps}
         model_action_dim: 32
         paligemma_variant: "gemma_2b"
         action_expert_variant: "gemma_300m"
         max_token_len: 200
         discrete_state_input: True
         use_rlt: True
         rlt_alpha: 1.0
         rlt_prefix_seq_len: 1024
         rlt_image_only: False
         rlt_use_mask: True

Keep ``repo_id``, ``config_name``, and normalization stats consistent with the
Stage 2 feature-model config. ``openpi_data`` belongs under ``actor.model``.

.. note::

   The ManiSkill joint example does not rely on manual ``state_indices``
   slicing. The ``pi05_rlt_maniskill_joint`` dataconfig maps the LeRobot
   ``state`` field into OpenPI ``observation.state``. Stage 2 rollout also uses
   the processed OpenPI ``observation.state`` as ``proprio``. Stage 1, Stage 2,
   and OpenPI normalization therefore see the same state semantics.

Stage 2: Train the Actor-Critic Policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 2 freezes the Stage 1 feature model and trains only the compact RLT MLP
actor and critic.

.. note::

   The current Stage 2 implementation is not standard maximum-entropy SAC.

During rollout:

1. The environment returns raw observations and task metadata.
2. ``rollout.rlt_feature_model`` runs the frozen Stage 1 model and converts the
   raw observation into:

   - ``z_rl``: the compact RLT representation.
   - ``proprio``: the selected robot or simulator state.
   - ``ref_chunk``: the VLA reference action chunk.

3. The Stage 2 actor consumes ``ref_chunk``, ``z_rl``, and ``proprio``.
4. The replay buffer stores RLT transitions:

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = action actually sent to the environment
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

For the provided real-robot config, ``keyboard_reward_wrapper:
rlt_policy_switch`` adds an ``rlt_switch_flags`` flag. Before the operator presses
``b``, the executed action is the VLA ``ref_chunk``; after ``b`` is pressed,
the executed action switches to the Stage 2 actor.

For the ManiSkill joint config, ``env.*.rlt_policy_switch`` automatically
produces ``rlt_switch_flags`` (actor/ref phase) and ``intervene_flag`` (expert
takeover request) from task information. The HF rollout route uses those flags
to choose the actor action, VLA ``ref_chunk``, or expert action at whole-chunk
granularity, then writes ``record_transition``, ``actor_switch``, and
``intervention_requested`` into ``forward_inputs`` for replay and monitoring.

The critic is trained with a TD target over chunked rewards. Rewards inside
the action chunk are discounted and then bootstrapped with the next-state Q
value:

.. code:: text

   target_q = discounted_chunk_reward + gamma ** chunk_horizon * Q_target(next_obs, next_action)

If the episode terminates and ``bootstrap_type`` is ``standard``, the bootstrap
term is removed.

Important Stage 2 fields:

.. code:: yaml

   # examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml
   algorithm:
     loss_type: rlt_ac
     q_weight: 0.1
     bc_weight: 5
     reference_dropout_prob: 0.5
     gamma: 0.96
     entropy_tuning:
       alpha_type: fixed_alpha
       initial_alpha: 0.0

   rollout:
     collect_transitions: True
     model:
       model_path: null
       precision: ${actor.model.precision}
       action_dim: ${actor.model.action_dim}
       num_action_chunks: ${actor.model.num_action_chunks}
       ref_num_action_chunks: ${actor.model.ref_num_action_chunks}
     rlt_feature_model:
       model_type: "openpi"
       precision: bf16
       is_lora: False
       num_action_chunks: 20
       action_dim: 7
       num_steps: 4
       add_value_head: False
       model_path: "/path/to/stage1/checkpoint/actor"
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"
         norm_stats_path: /path/to/lerobot_dataset/norm_stats.json
       openpi:
         task: eval
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         # Network prediction horizon (Pi0Config.action_horizon). Matches official
         # TrainConfig for config_name pi05_franka_state. Distinct from action_chunk
         # / num_action_chunks, which is the env-executed (RLT) window.
         action_horizon: 20
         action_chunk: ${actor.model.ref_num_action_chunks}
         action_env_dim: ${rollout.rlt_feature_model.action_dim}
         num_steps: ${rollout.rlt_feature_model.num_steps}
         model_action_dim: 32
         paligemma_variant: "gemma_2b"
         action_expert_variant: "gemma_300m"
         max_token_len: 200
         discrete_state_input: True
         state_indices: []      # keep the full raw state, e.g. 19D
         use_rlt: True
         rlt_prefix_seq_len: 1024
         rlt_image_only: False
         rlt_use_mask: True

   actor:
     model:
       model_type: "rlt_mlp_policy"
       precision: fp32
       add_value_head: False
       add_q_head: True
       q_head_type: "default"
       fixed_std: 0.002
       is_lora: False
       z_dim: 2048
       proprio_dim: 19
       action_dim: 7
       num_action_chunks: 10
       ref_num_action_chunks: 20

   env:
     train:
       keyboard_reward_wrapper: rlt_policy_switch

The Stage 2 actor loss is:

.. code:: text

   actor_loss = -q_weight * Q(obs, pi(obs)) + bc_weight * BC(pi(obs), target_action)

The BC target is the VLA reference action for normal policy steps. If a human
intervention action is stored for a step, the BC target for that step becomes
the human action. ManiSkill disables ``expert_takeover`` by default, so the
default simulated route uses only the VLA reference and actor actions.

Run the Provided Franka Example
-------------------------------

Data: Collect Franka Demonstrations and Compute Normalization Stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 expects a Franka demonstration dataset in LeRobot format; the dataset
directory should directly contain ``data/`` and ``meta/``. On the controller
node, follow the data-collection flow in the :doc:`Franka real-world guide
<../embodied/franka>` to prepare the robot and target pose, and export LeRobot
data from the collection config:

.. code:: yaml

   env:
     data_collection:
       enabled: True
       export_format: "lerobot"

Then launch collection:

.. code:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data

After collection, place the LeRobot dataset on the training node and compute
normalization statistics for the RLT OpenPI dataconfig. ``repo_id`` should
match ``actor.openpi_data.repo_id`` and
``rollout.rlt_feature_model.openpi_data.repo_id`` in the Stage 1 / Stage 2
configs:

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_franka_state \
       --repo-id realworld_peg_insertion_rlt_stage1

Then point ``data.train_data_paths`` in the Stage 1 config at that LeRobot dataset directory. If the checkpoint does not embed norm stats, set ``norm_stats_path`` in ``openpi_data`` to point at ``norm_stats.json``.

Stage 1: Train the RLT Feature Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the Stage 1 config paths before launch:

.. code:: yaml

   data:
     train_data_paths: /path/to/lerobot_dataset

   actor:
     openpi_data:
       repo_id: "realworld_peg_insertion_rlt_stage1"
     model:
       model_path: /path/to/model
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         rlt_prefix_seq_len: 1024

Launch SFT:

.. code:: bash

   bash examples/sft/run_vla_sft.sh realworld_rlt_stage1_sft_openpi_pi05

The saved checkpoint directory should look like:

.. code:: text

   logs/<run-name>/checkpoints/global_step_<step>/actor

Use this ``actor`` directory as ``rollout.rlt_feature_model.model_path`` in Stage 2.
Do not put the Stage 1 checkpoint under ``rollout.model.model_path`` or
``actor.model.model_path``; those fields do not load the Stage 1 feature model.

Stage 2: Run RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the Stage 2 config:

.. code:: yaml

   rollout:
     model:
       model_path: null
     rlt_feature_model:
       model_path: /path/to/stage1/checkpoint/actor
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         state_indices: []
         rlt_prefix_seq_len: 1024

   cluster:
     node_groups:
       - label: <gpu_node_group>
         node_ranks: <gpu_node_rank>
       - label: <env_node_group>
         node_ranks: <env_node_rank>
         hardware:
           type: <robot_type>
           configs:
             - robot_ip: <robot_ip>

   env:
     train:
       keyboard_reward_wrapper: rlt_policy_switch
       override_cfg:
         target_ee_pose: [<target_x>, <target_y>, <target_z>, <target_roll>, <target_pitch>, <target_yaw>]

Launch the async run from the master node:

.. code:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_rlt_stage2_ac_mlp

The default keyboard module implements the key phase switch used by RLT: press
``b`` to enter the Stage 2 actor-controlled phase. Other behavior can be
customized for the task in
``rlinf/envs/real/wrappers/episode/policy_switch.py``.

Run the ManiSkill Joint Example
-------------------------------

The ManiSkill joint example uses ``PegInsertionSideWideClearance-v1``,
two-view RGB observations, the first 9 Panda qpos dimensions, and 8D
``pd_joint_delta_pos`` actions. Each Stage 2 action is a 10-step chunk. The
language instruction is fixed to ``insert the peg in the hole``.

Data: Prepare the Joint-Control LeRobot Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Stage 1 and Stage 2 OpenPI feature model uses the
``pi05_rlt_maniskill_joint`` dataconfig. The LeRobot dataset must contain at
least:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``image``
     - Main-view RGB image.
   * - ``wrist_image``
     - Wrist RGB image.
   * - ``state``
     - Panda joint proprioception. The current example uses a 9D state.
   * - ``actions``
     - 8D ``pd_joint_delta_pos`` actions.
   * - ``task``
     - Language instruction. You can also set ``default_prompt`` to ``insert the peg in the hole``.

You can start from the reference dataset
`RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ
<https://huggingface.co/datasets/RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ>`__.
It contains successful ManiSkill ``PegInsertionSideWideClearance-v1``
demonstrations, uses the ``pd_joint_delta_pos`` action space, and follows the
joint-control LeRobot fields above.

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   huggingface-cli download RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ \
       --repo-type dataset \
       --local-dir ${HF_LEROBOT_HOME}/maniskill_peginsertionside_joint

You can also regenerate the dataset with the collection script in this branch.
The script first runs ManiSkill's Panda motion-planning solver, converts
``pd_joint_pos`` solver actions into ``pd_joint_delta_pos`` actions, and saves
only episodes that replay successfully:

.. code:: bash

   python toolkits/lerobot/collect_maniskill_peg_lerobot_joint.py \
       --repo-id maniskill_peginsertionside_joint \
       --num-episodes 400 \
       --seed 0 \
       --max-attempts 4000 \
       --overwrite

.. note::

   The collection script lives under ``toolkits/lerobot`` and is intended for
   RLT joint data preparation. It depends on ManiSkill's
   PegInsertionSide Panda motion-planning solver. If importing the solver fails,
   first check that your ManiSkill installation includes the motion-planning
   examples.

When you compute normalization statistics, keep ``--config-name`` and
``--repo-id`` aligned with the training config:

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_rlt_maniskill_joint \
       --repo-id maniskill_peginsertionside_joint

.. warning::

   The Stage 1 checkpoint, Stage 2 ``rollout.rlt_feature_model``, and OpenPI
   assets must use the same ``norm_stats.json``. If the SFT base policy and
   Stage 2 load stats from different ``repo_id`` directories, the scale of VLA
   reference actions shifts. Set ``openpi_data.norm_stats_path`` to that ``norm_stats.json`` when you need an explicit path.

Stage 1: Jointly Train the ManiSkill OpenPI + RLT Feature Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the data path in
``examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml``:

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/maniskill_peginsertionside_joint
         weight: 1.0

   actor:
     model:
       model_path: /path/to/pi05_base
       openpi:
         config_name: pi05_rlt_maniskill_joint
       openpi_data:
         repo_id: maniskill_peginsertionside_joint
         default_prompt: insert the peg in the hole
         norm_stats_path: /path/to/maniskill_peginsertionside_joint/norm_stats.json

Launch training:

.. code:: bash

   bash examples/sft/run_vla_sft.sh maniskill_rlt_stage1_sft_openpi_pi05

The saved FSDP checkpoint usually looks like:

.. code:: text

   logs/<run-name>/checkpoints/global_step_<step>/actor

Point this ``actor`` directory directly at
``rollout.rlt_feature_model.model_path`` in Stage 2, for example
``/path/to/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_<step>/actor``.

Stage 2: Run ManiSkill RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the Stage 1 checkpoint in
``examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml``:

.. code:: yaml

   env:
     train:
       wrap_obs_mode: rlt_openpi_joint
       rlt_policy_switch:
         enable: True
         task_mode: full_task
         trigger_mode: auto
     eval:
       wrap_obs_mode: rlt_openpi_joint
       rlt_policy_switch:
         enable: True
         task_mode: full_task
         trigger_mode: auto

   rollout:
     rlt_feature_model:
       model_path: /path/to/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_<step>/actor
       openpi_data:
         repo_id: maniskill_peginsertionside_joint
         norm_stats_path: /path/to/maniskill_peginsertionside_joint/norm_stats.json
       openpi:
         config_name: pi05_rlt_maniskill_joint
     expert_model:
       model_path: /path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_<step>/actor
       precision: null
       openpi:
         use_rlt: False

Launch training:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_rlt_stage2_ac_mlp

To run the TD3-MLP variant, apply the same Stage 1 checkpoint settings to
``maniskill_rlt_stage2_td3_mlp.yaml`` and launch:

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_rlt_stage2_td3_mlp

This variant is currently configured only for ManiSkill simulation. It keeps
the frozen Stage 1 features and transition replay path, while replacing the
Stage 2 policy and update objective with a direct TD3 actor and twin-Q critic.

This config starts actor, rollout, and ManiSkill env workers. The rollout side
freezes ``rollout.rlt_feature_model`` and only synchronizes the Stage 2 MLP
actor. Before ``ready_for_online``, the ManiSkill route executes the VLA
``ref_chunk``. After ``algorithm.rlt_schedule.warmup_post_collect_updates``,
the actor can take over in the automatic critical phase.

Tune ManiSkill Critical Phase and Intervention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ManiSkill switch is simulator-specific. It uses peg-insertion task info to
turn on ``rlt_switch_flags`` / ``in_critical_phase`` and, if enabled, to request
expert takeover after repeated no-progress actor chunks. The comments below are
documentation-only annotations; keep the checked-in YAML clean.

.. code:: yaml

   algorithm:
     rlt_schedule:
       enable: True                    # Use the ManiSkill rollout/update budget.
       warmup_post_collect_updates: 30000  # Updates before actor rollout starts.
       train_every_transitions: 5       # Add update budget per N recorded transitions.
     actor_weight_schedule:
       enable: True                    # Ramp BC/Q weights instead of fixed weights.
       warmup_updates: 20000           # Keep warmup BC/Q weights for this many updates.
       ramp_updates: 50000             # Smoothly move to online BC/Q weights.

   env:
     train:
       rlt_policy_switch:
         task_mode: full_task          # VLA before critical phase, actor inside it.
         trigger_mode: auto            # Compute critical phase from task info.
         latch_until_done: True        # Keep actor control after entering critical phase.
         auto_gate:
           require_grasp: True         # Enter only after the peg is grasped.
           require_not_success: True   # Do not switch after task success.
           near_hole_x_min: -0.16      # Larger value narrows the x entry gate.
           near_hole_yz_margin: 1.5    # Larger value widens the y/z entry gate.
         expert_takeover:
           enable: True                # Train-only simulated expert intervention.
           trigger_mode: stalled_progress  # Wait for repeated no-progress chunks.
           gate:
             near_hole_x_min: -0.10    # Start stall checks only near the hole.
             near_hole_yz_margin: 2.0  # Start stall checks only when y/z is close.
             stuck_chunks_before_takeover: 3  # No-progress chunks before takeover.
             min_x_progress: 0.003     # Forward progress that resets stall count.
             min_yz_progress: 0.0015   # Alignment progress that resets stall count.
             min_score_progress: 0.002 # Combined-score progress that resets stalls.
             progress_yz_weight: 1.0   # y/z penalty in the combined progress score.

     eval:
       rlt_policy_switch:
         expert_takeover:
           enable: False               # Eval uses base policy + actor, never expert.

Optional: Enable ManiSkill Expert Takeover
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ManiSkill expert takeover path is disabled by default. Enable it when you
want the rollout route to replace actor actions with a stronger SFT expert in
the critical phase and store those expert actions as intervention targets:

.. code:: yaml

   rollout:
     expert_model:
       model_path: /path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_<step>/actor
       precision: null
       openpi:
         use_rlt: False

   env:
     train:
       rlt_policy_switch:
         expert_takeover:
           enable: True
           trigger_mode: stalled_progress
           gate:
             stuck_chunks_before_takeover: 3
             min_x_progress: 0.003
             min_yz_progress: 0.0015
             min_score_progress: 0.002
             progress_yz_weight: 1.0

You can replace ``rollout.expert_model.model_path`` with a more fully trained
joint-control SFT checkpoint. Keep the expert's OpenPI dataconfig and norm
stats aligned with the Stage 2 dataset. The expert is only used for train
rollout; eval rollout runs without expert takeover and measures the learned actor.

Replay Buffer Behavior
----------------------

For ``loss_type: rlt_ac``, the replay buffer does not store raw image
observations as the RL state. The rollout worker returns RLT features, and the
learner side stores those features as transitions:

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = action chunk actually sent to the environment
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

This means:

- Steps before a real-robot policy switch are still useful. Their executed
  action is the VLA reference action, and their transition is stored in the
  same replay buffer.
- Steps after the switch use the actor action and are also stored with the
  same RLT observation format.
- The ManiSkill route chooses actor actions or the VLA ``ref_chunk`` at whole
  chunk granularity. The replay ``action`` is always the executed action.
- The ManiSkill learner splits rollout chunks into 1-sample transition
  trajectories before adding them to RLinf ``TrajectoryReplayBuffer``.
- ``sample_window_size`` controls the recent transition window sampled from the
  replay buffer. It does not need to match ``max_steps_per_rollout_epoch``.
- ``max_steps_per_rollout_epoch`` controls how many environment steps are
  collected before the rollout worker flushes a batch to training.

Monitoring
----------

For metric definitions, see :doc:`Training metrics <../../reference/metrics>`.
Useful RLT signals:

- Stage 1 SFT:

  - ``vla_loss``: the OpenPI action-prediction loss.
  - ``rlt_loss``: the RLT token reconstruction/compression loss.

- Stage 2 actor-critic:

  - ``critic/critic_loss``: Q-function TD loss.
  - ``actor/actor_loss``: combined ``-Q + BC`` actor objective.
  - ``q_pi`` and ``q_value_*``: learned Q-values for actor and critic heads.
  - ``actor/bc_loss``, ``actor/bc_ref_loss``, ``actor/bc_human_loss``: BC regularization terms.
  - ``actor/human_mask_ratio``: fraction of the current training batch marked as expert intervention (from ``trajectory.intervene_flags``).
  - ``train/replay_buffer/size``: number of stored replay transitions.
  - ``env/success_once`` and ``env/episode_len``: task outcome metrics.
  - ``eval/success_once``: success rate on fixed eval reset ids.

  ManiSkill rollout / replay diagnostics (logged from trajectories received by the actor):

  - ``train/replay/record_transition_rate``: fraction of collected steps saved as RLT transitions (from ``forward_inputs.record_transition``).
  - ``train/replay/actor_switch_rate``: fraction of collected steps where the actor/student action actually controlled the env (from ``forward_inputs.actor_switch``).
  - ``train/replay/intervention_requested_rate``: fraction of steps where the env requested expert takeover (from ``forward_inputs.intervention_requested``).
  - ``train/replay/intervention_rate``: fraction of steps where the route actually applied expert actions (from ``trajectory.intervene_flags``).
  - ``train/replay/transition_count``, ``train/replay/reward_mean``, ``train/replay/reward_positive_rate``, ``train/replay/done_rate``: ManiSkill transition-replay ingest stats for the current collect step.

  RLT schedule / learner backlog (ManiSkill ``algorithm.rlt_schedule.enable``):

  - ``train/rlt/ready_for_online``: whether learner ``update_step`` has passed ``warmup_post_collect_updates``.
  - ``train/rlt/actor_updates_run``, ``train/rlt/critic_updates_run``, and ``train/rlt/pending_update_budget``: actor/critic updates executed in the step and remaining learner backlog.
  - ``train/rlt/updates_to_run``, ``train/rlt/should_train``, and ``train/rlt/skip_reason``: scheduled training for the step and why training may be skipped.
  - ``train/rlt/global_transitions_since_train`` and ``train/rlt/global_total_transitions_added``: global replay ingest counters since the last training burst.

Experimental Results
--------------------

The RL Token training result on the peg_insertion task in RLinf is shown below.

.. raw:: html

   <div style="display: flex; justify-content: center; margin: 20px 0;">
     <div style="flex: 0.5; text-align: center;">
       <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/RLT_real.png" style="width: 100%;"/>
       <p><em>RL Token training result on the RLinf peg_insertion task</em></p>
     </div>
   </div>

Practical Notes
--------------------------------------------------------------------------------

- Keep Stage 1 and Stage 2 data settings consistent: ``repo_id``,
  ``config_name``, ``action_dim``, ``proprio_dim``, ``ref_num_action_chunks``,
  and ``z_dim`` must agree. To keep the full raw state, use
  ``state_indices: []``.
- ``rollout.rlt_feature_model`` should point to the Stage 1 checkpoint, while
  ``actor.model`` is the Stage 2 MLP policy updated by the actor-critic
  worker.
- ``rollout.model`` is the synced Stage 2 MLP copy on rollout workers. Keep
  ``rollout.model.model_path: null`` for scratch Stage 2 training; use
  ``runner.resume_dir`` to resume a Stage 2 run or ``runner.ckpt_path`` to load
  a single Stage 2 weight file.
- Do not configure ``actor.model.model_path`` for Stage 1. ``actor.model`` only
  describes the Stage 2 MLP input/output shape and Q-head settings.
- Stage 2 MLP settings are defined inline under ``actor.model`` in each Stage 2
  YAML, not in a separate model defaults file.
- ``keyboard_reward_wrapper: rlt_policy_switch`` is only needed for
  operator-controlled critical-phase switching.
- The ManiSkill joint example uses ``env.*.rlt_policy_switch``. Do not use the
  real-robot keyboard wrapper there.
- ManiSkill ``proprio`` comes from the processed OpenPI ``observation.state``.
  When you add a new simulator dataconfig, check the dataset ``state``, OpenPI
  transform, and Stage 2 ``proprio_dim`` together.
- Stage 1, Stage 2, and checkpoint assets must load ``norm_stats.json`` from the same data semantics and ``repo_id``. Prefer setting ``openpi_data.norm_stats_path`` explicitly so Stage 1 runs still work when the checkpoint does not embed norm stats.
- ``rollout.rlt_feature_model.model_path`` should point to the Stage 1 FSDP ``actor`` directory, for example ``.../checkpoints/global_step_<step>/actor``. Do not convert the RLT Stage 1 checkpoint to a bare ``model.safetensors`` for Stage 2, because the RLT token module is stored in the full wrapper checkpoint.
- To add a simulator example, create a simulator environment config, keep
  ``loss_type: rlt_ac`` and ``rollout.rlt_feature_model``, and replace the
  real-robot phase-switching logic with simulator-appropriate behavior.

Use Evo-RL Piper Data and Local Hardware
--------------------------------------------------------------------------------

Single-arm Piper follows the native RLinf RLT pipeline: train Stage 1 on LeRobot v3 recordings, then freeze it and update the Stage 2 actor/critic from online experience. The ``piper_rlt_stage1`` and ``piper_rlt_stage2`` recipes reuse the existing RLT models, learner, and replay path. RECAP value training and advantage labeling are separate workflows.

Prepare Stage 1 Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use LeRobot 0.4.4 in the RLinf OpenPI environment. The preparer expects seven absolute targets (six joint angles in degrees and gripper width in millimetres), the corresponding state layout, and three RGB views named ``front / front_2 / wrist`` or ``base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb``. It copies metadata, links trajectories and videos, and computes normalization in model units using ten-step action windows.

.. code-block:: bash

   export REPO_PATH="$PWD"
   export PIPER_DATA_PATH=/path/to/prepared/piper_rlt
   python -m toolkits.lerobot.prepare_piper \
     --source /path/to/Evo-RL/data/usb_new_merged \
     --output "$PIPER_DATA_PATH"
   export PI05_MODEL_PATH=/path/to/pi05_base_pytorch
   bash examples/sft/run_vla_sft.sh piper_rlt_stage1

Choose a new output directory. When transferring it to a server, copy the actual trajectory/video files or recreate valid links there. Curate suitable demonstrations for a full run: format compatibility does not make failed recorded actions good SFT targets. The initial recipe runs 100 steps to exercise training; adjust the step count and optimizer schedule for experiments.

Add ``--interactive --num-episodes 20`` to the local rollout command to evaluate
20 trials. Keep the terminal focused. Tap Right to start each trial (including
the first), then press ``s`` for success or ``f`` for failure without Enter.
Scoring stops new targets; the arm holds its last target while the operator
restores the scene. By default, no reset trajectory or motor release is performed. Keys
entered during synchronous inference are handled before the next target is sent.

``--steps`` caps actions per trial. At the limit, wait for a manual score rather
than counting a failure. Model and hardware connections are reused. Each
``episode_001/`` directory contains its trace, result, and optionally observations.
``summary.json`` is updated after each trial; success rate uses only scored trials.
Ctrl-C or a fault aborts evaluation and preserves partial results. Unscored,
interrupted, and faulted trials are excluded. ``--no-action-limits`` applies to
all trials when supplied.

Add ``--reset-after-episode initial`` to capture the pose immediately before the
first trial and return after every score, including the last. The sequence opens
the gripper, returns the joints, then restores its initial opening. Select
``zero`` for zero joints and a closed gripper, or ``none`` (default) for no reset.
Connections remain open. The implementation follows Evo-RL's
``_move_robot_to_zero_pose`` and ``prepare_for_reset``: reassert joint control,
confirm motor enable, then linearly interpolate from measured position.
``--reset-duration-s`` defaults to 3 seconds and
``--reset-gripper-duration-s`` to 0.5 seconds for each gripper stage.
Each tick includes its I/O time; slow ticks do not cause catch-up bursts.
The former 5-degree start-recovery gate is removed. Sent targets still use the
driver's original travel clipping; controller status, fresh feedback and finite
value checks remain. Controller faults are not automatically cleared. Leaders,
teaching mode and camera/CAN connection ownership are unchanged.

``trajectory_sent`` means transmission completed, not verified physical arrival.
Final measured pose and error are recorded in ``result.json`` and ``reset.jsonl``.
Check the pose and restore the scene before the next Right press. The fixed
return target is saved in ``reset_pose.json``. Reset-time keys are discarded.
If reset faults, the score is saved and connections remain open: resolve the
fault, then press ``r`` to retry reset, or ``q`` / Ctrl-C to exit. Right cannot
skip a failed reset. Retries do not duplicate scores and have separate logs
such as ``reset_retry_02.jsonl``. Other rollout faults still end evaluation.


Each Stage 1 full checkpoint includes ``piper_inference.json`` and ``assets/piper/norm_stats.json`` under ``actor/``. Download the complete actor directory, preserving the VLA and ``rlt_module.*`` tensors in ``model_state_dict/full_weights.pt``. The metadata carries model settings, dataset frame rate, and normalization validation information, so local inference does not depend on server-side model/data paths. OpenPI tokenizer resources are still required: allow the initial download or prepare their cache on an offline machine.

The local loader supports ``pi05_piper`` (10-step horizon, no discrete state
prompt) and ``pi05_piper_h50_state`` (50-step horizon, discrete state input).
It selects the observation transforms and normalized RLT action branch using
the checkpoint configuration. Keep the exported ``piper_inference.json``
configuration name unchanged.

Check Recorded Inputs and Run Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with the frozen Stage 1 reference policy on one recorded observation. This command reads the dataset and saves predictions without opening cameras or CAN:

.. code-block:: bash

   export PIPER_STAGE1_PATH=/path/to/downloaded/stage1/actor
   python -m toolkits.piper.run_policy infer \
     --checkpoint "$PIPER_STAGE1_PATH" \
     --dataset "$PIPER_DATA_PATH" --frame 0

Each invocation creates a fresh result directory containing arguments, the observation, and ``prediction.json``. For physical rollout, set the follower USB CAN serial, camera USB identities, and gripper stroke in ``examples/embodiment/config/piper_usb_hardware.yaml``. The example selects each RGB capture node by USB serial and interface: Gemini 335 uses interface ``04``, while its depth interface is ``00``. This avoids ambiguous device aliases after USB re-enumeration. Explicit ``index_or_path`` camera paths remain supported as an alternative to USB identity. Cameras must retain their training placement. The example enables ``reset_on_connect`` for the two Gemini cameras to recover color streaming between runs. Before capture, it checks that the selected USB/video nodes are idle, resets only those cameras, and resolves their RGB nodes again; it requires ``fuser``, ``usbreset``, ``udevadm``, and USB write permission. Wrist and CAN devices are not reset. All cameras must provide fresh frames before the follower connects. ``front_2`` is a second external view mapped into the model's left slot.

.. code-block:: bash

   python -m toolkits.piper.run_policy rollout \
     --checkpoint "$PIPER_STAGE1_PATH" \
     --hardware examples/embodiment/config/piper_usb_hardware.yaml \
     --task "pick up the two USB cables and insert them into the USB port on the outlet." \
     --steps 900 --save-observations

The USB hardware example uses Evo-RL's USB inference settings: ``high_follow: true``, ``speed_percent: 100``, and ``gripper_force: 1.0``. pyAgxArm's ``move_js`` corresponds to ``MotionCtrl_2(1, 1, 100, 0xAD)`` followed by absolute joint targets. This is joint follower mode, not direct torque commands through ``move_mit``. The gripper's wire effort remains 1000. Generic ``PiperArm`` instances and hardware configurations without ``high_follow`` retain planned ``move_j`` commands. Local rollout and Stage 2 share this hardware construction path.

This command enables the configured follower and starts sending policy targets without a homing trajectory. The loop follows Evo-RL's synchronous ``select_action`` queue: observe on each tick, predict when the queue is empty, send one target, then sleep only for the remaining tick budget. Observation, inference, logging, and submission all count toward that budget. Inference still introduces gaps; continuous 30 Hz is not guaranteed. By default, consume the full policy horizon before predicting again; override with ``--replan-steps``. The horizon is 10 for ``pi05_piper`` and 50 for ``pi05_piper_h50_state``; use the checkpoint configuration.

The USB example sets ``execution.max_prediction_age_s: 3.0``. This bounds the total age since observation, including inference, recording and chunk playback, rather than inference alone. At 30 Hz, an H50 chunk needs about 1.67 seconds of playback in addition to inference; a 0.58-second first prediction can exceed the former 2-second limit before the chunk ends. Excessively old predictions still stop rollout, with elapsed time and chunk progress in the error.

``--steps`` counts submitted targets; reaching it ends the run without judging task success. Match the task text to the dataset; the example text describes ``usb_light`` with two USB cables. Ctrl-C or a failure closes connections while motors retain the last target. The executor checks feedback freshness, joint travel, and per-command changes from measured positions. ``trace.jsonl`` contains diagnostics rather than an online replay dataset. ``--save-observations`` saves each prediction's three RGB views, raw state, and task to ``observations/step_*.npz`` and records the file path in the prediction trace for offline replay. Disk writes also count toward the tick budget.

Follower mode is reasserted after the final speed setting. Target submission waits for feedback confirming CAN control, joint motion mode, and normal status. The SDK's speed setter sends a frame with motion mode ``0xFF``, so enabled motors alone do not confirm that joint control is ready.

Add ``--no-action-limits`` to disable the rollout layer's extra target-range and per-command step checks. Piper's underlying driver still clips commands to mechanical travel; finite-value and stale-input checks remain active. The effective setting is saved in each run's ``hardware.yaml``.

Run Native RLT Online Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PiperPolicyEnv-v1`` supplies three RGB views and seven raw state values to frozen Stage 1. The feature model returns ``z_rl``, seven normalized ``proprio`` values, and normalized ``ref_chunk`` actions. Actor, critic, and replay share those normalized action coordinates. At execution, the environment uses Stage 1 statistics to recover absolute degrees/mm, then converts to driver radians and gripper opening fraction.

.. code-block:: bash

   bash examples/embodiment/run_realworld_async.sh piper_rlt_stage2

The recipe runs existing RLT online workers on a single workstation connected to Piper. Set ``PIPER_STAGE1_PATH``, check the hardware configuration, and select ``RLINF_KEYBOARD_DEVICE`` as described in the keyboard documentation. Each episode waits for ``a`` to start with Stage 1 reference actions; ``b`` selects the Stage 2 actor, ``c`` reports success, and ``q`` reports failure. Successful termination yields reward 1; other rewards are 0. Timeout is a truncation. Reset waits for manual scene preparation and sends no reset trajectory; leader takeover is not connected. Actor/reference routing and transition recording retain native real-world RLT semantics.

Stage 2 checkpoints include ``piper_rlt_actor.json`` bound to the Stage 1 identifier used in training. Add ``--actor /path/to/stage2/actor`` to either local command to evaluate the learned actor. Retain the corresponding Stage 1 directory; the loader rejects mismatched Stage 1/Stage 2 pairs.

Shared Interfaces and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``PiperInputs / PiperOutputs`` share camera and unit conversion. ``PiperSession.observe()`` and ``execute()`` provide algorithm-independent local I/O. Other algorithms can reuse these interfaces but still need their own policy and learner integration. ``--backend cfg`` loads a RECAP/CFG checkpoint; that checkpoint cannot substitute for RLT Stage 1.

Validation covers real dataset reading, training/online preprocessing agreement, checkpoint metadata relocation, native RLT actor routing, mock SDK execution and cleanup, and dummy environment episodes. Full GPU training, inference with real trained weights, and physical online-learning performance still require validation in the deployment environment.
