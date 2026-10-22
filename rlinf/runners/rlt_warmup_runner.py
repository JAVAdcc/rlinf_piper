# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-device offline RLT training over frozen, annotated Piper features."""

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.segment import (
    action_mask,
    actor_training_observation,
    discounted_target,
    masked_behavior_loss,
)
from rlinf.data.datasets.rlt_warmup import (
    CachedRLTReplay,
    directory_lock,
    file_sha256,
    implementation_fingerprints,
    json_fingerprint,
    runtime_source_fingerprints,
    runtime_versions,
    write_json,
)
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.utils.logging import get_logger

logger = get_logger()


@dataclass
class WarmupTraining:
    """Optimizer update budget; BC and critic phases have separate owners."""

    bc_steps: int = 5000
    critic_steps: int = 10000
    joint_steps: int = 5000
    actor_every: int = 4
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.995
    tau: float = 0.005
    grad_clip: float = 10.0
    bc_start: float = 5.0
    bc_end: float = 2.0
    q_end: float = 0.1
    dropout_end: float = 0.5
    eval_every: int = 1000
    save_every: int = 1000
    keep_checkpoints: int = 3


@dataclass
class WarmupValidation:
    """Fixed first-run thresholds; an offline pass permits deterministic evaluation."""

    max_rows_per_episode: int = 256
    consecutive_passes: int = 3
    min_train_per_outcome: int = 20
    min_validation_per_outcome: int = 10
    min_train_transitions: int = 600
    actor_rmse: float = 0.05
    joint_p95_deg: float = 2.0
    gripper_p95_mm: float = 3.0
    terminal_rmse: float = 0.25
    baseline_improvement: float = 0.2
    td_rmse: float = 0.25
    td_ratio: float = 2.0
    q_gap: float = 0.1
    q_low: float = -0.05
    q_high: float = 1.05
    q_in_range_fraction: float = 0.99


@dataclass
class WarmupConfig:
    """Paths and independent cache/training settings for the Piper entrypoint."""

    stage1: str = "checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"
    annotations: str = "outputs/piper/episodes.json"
    cache: str = "outputs/piper/rlt_cache"
    output: str = "outputs/piper/rlt_warmup"
    hardware: str = "examples/embodiment/config/piper_usb_hardware.yaml"
    device: str = "cuda:0"
    seed: int = 1234
    threads: int = 4
    chunk_length: int = 10
    stride: int = 10
    tail_policy: str = "pad"
    video_backend: str = "torchcodec"
    phase_filter: int | None = None
    critic_action_projection: str = "none"
    training: WarmupTraining = field(default_factory=WarmupTraining)
    validation: WarmupValidation = field(default_factory=WarmupValidation)


def load_warmup_config(path: str | Path, overrides: list[str] = ()) -> WarmupConfig:
    """Load one typed YAML plus Hydra-style dotted overrides; reject unknown keys."""
    cfg = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(WarmupConfig),
            OmegaConf.load(path),
            OmegaConf.from_dotlist(overrides),
        )
    )
    for key in (
        "bc_steps",
        "critic_steps",
        "actor_every",
        "batch_size",
        "eval_every",
        "save_every",
        "keep_checkpoints",
    ):
        if getattr(cfg.training, key) <= 0:
            raise ValueError(f"training.{key} must be positive.")
    if cfg.training.joint_steps < 0:
        raise ValueError(
            "training.joint_steps must be nonnegative; zero stops before joint."
        )
    if cfg.threads <= 0 or cfg.seed < 0 or not 0 < cfg.stride <= cfg.chunk_length:
        raise ValueError("Invalid threads, seed or cache stride/chunk length.")
    if cfg.tail_policy not in ("pad", "align"):
        raise ValueError("tail_policy must be pad or align.")
    if cfg.video_backend not in ("torchcodec", "pyav"):
        raise ValueError("video_backend must be torchcodec or pyav.")
    if cfg.phase_filter is not None and (
        type(cfg.phase_filter) is not int or cfg.phase_filter < 0
    ):
        raise ValueError("phase_filter must be a nonnegative ordinal or null.")
    if cfg.critic_action_projection not in ("none", "driver_limits"):
        raise ValueError("critic_action_projection must be none or driver_limits.")
    for key in ("learning_rate", "grad_clip", "bc_start", "bc_end"):
        if (
            not np.isfinite(getattr(cfg.training, key))
            or getattr(cfg.training, key) <= 0
        ):
            raise ValueError(f"training.{key} must be finite and positive.")
    if not 0 < cfg.training.gamma <= 1 or not 0 < cfg.training.tau <= 1:
        raise ValueError("gamma and tau must be in (0,1].")
    if not 0 <= cfg.training.dropout_end <= 1 or not 0 <= cfg.training.q_end < float(
        "inf"
    ):
        raise ValueError("Invalid reference dropout or Q weight.")
    for key, value in asdict(cfg.validation).items():
        if not np.isfinite(value) or (key not in ("q_low", "q_high") and value <= 0):
            raise ValueError(f"Invalid validation.{key}.")
    if (
        cfg.validation.baseline_improvement >= 1
        or cfg.validation.q_in_range_fraction > 1
    ):
        raise ValueError("Invalid validation fraction.")
    if cfg.validation.q_low >= cfg.validation.q_high:
        raise ValueError("Invalid Q interval.")
    return cfg


def offline_warmup_passed(
    history: list[dict], *, step: int, budget: int, interval: int, required_passes: int
) -> bool:
    """Require completed budget, consecutive checks and validation of current weights."""
    if (
        not history
        or step != budget
        or history[-1]["step"] != step
        or not history[-1]["passed"]
    ):
        return False
    reports = [row for row in history if row["scheduled"]][-required_passes:]
    return (
        len(reports) == required_passes
        and all(row["passed"] for row in reports)
        and all(b["step"] - a["step"] == interval for a, b in zip(reports, reports[1:]))
    )


class RLTWarmupRunner:
    """Train native RLT MLP heads without loading the frozen VLA or robot."""

    def __init__(self, cfg: WarmupConfig):
        self.cfg = cfg
        torch.set_num_threads(cfg.threads)
        torch.set_float32_matmul_precision("high")
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.replay = CachedRLTReplay(
            cfg.cache, seed=cfg.seed, phase_filter=cfg.phase_filter
        )
        if self.replay.manifest["chunk_length"] != cfg.chunk_length:
            raise ValueError("Config chunk_length disagrees with cache.")
        self.hardware = OmegaConf.to_container(
            OmegaConf.load(cfg.hardware), resolve=True
        )
        from rlinf.robotics.piper_runtime import PiperExecutionConfig

        self.execution = PiperExecutionConfig(**self.hardware["execution"])
        if not self.execution.check_action_limits:
            raise ValueError("Offline readiness requires configured action checks.")
        if self.execution.fps != self.replay.manifest["stage1"]["fps"]:
            raise ValueError("Hardware FPS disagrees with cache.")
        from rlinf.models.embodiment.openpi.policies.piper_policy import (
            PiperActionCodec,
        )

        self.codec = PiperActionCodec(self.replay.path / "norm_stats.json")
        from rlinf.robotics.parts.arms.piper import PiperArm

        driver_bounds = np.stack(
            (
                np.r_[np.rad2deg(PiperArm.JOINT_LIMITS_LOWER), 0.0],
                np.r_[
                    np.rad2deg(PiperArm.JOINT_LIMITS_UPPER),
                    self.execution.gripper_max_width_m * 1000,
                ],
            )
        )
        self.driver_action_bounds = torch.as_tensor(
            self.codec.encode(driver_bounds), dtype=torch.float32, device=self.device
        )
        model_cfg = {
            k: v
            for k, v in self.replay.manifest["model"].items()
            if k not in ("model_type", "precision")
        }
        self.model = RLTMLPPolicy(**model_cfg).to(self.device).train()
        self.target = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.actor_params = list(self.model.backbone.parameters()) + list(
            self.model.actor_mean.parameters()
        )
        self.critic_params = list(self.model.q_head.parameters())
        self.actor_optimizer = torch.optim.Adam(
            self.actor_params, lr=cfg.training.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_params, lr=cfg.training.learning_rate
        )
        self.step = self.actor_steps = self.critic_steps = 0
        self.history: list[dict] = []
        self.output = Path(cfg.output).resolve()
        self.binding = json_fingerprint(
            {
                "training": asdict(cfg.training),
                "validation": asdict(cfg.validation),
                "seed": cfg.seed,
                "device": str(self.device),
                "threads": cfg.threads,
                "cache": self.replay.manifest["content_fingerprint"],
                "hardware": self.hardware,
                "phase_filter": cfg.phase_filter,
                "critic_action_projection": cfg.critic_action_projection,
                "driver_action_bounds": self.driver_action_bounds.cpu().tolist(),
                "implementation": implementation_fingerprints(
                    [
                        "rlinf/runners/rlt_warmup_runner.py",
                        "rlinf/data/datasets/rlt_warmup.py",
                        "rlinf/data/schema/critical_phase.py",
                        "rlinf/models/embodiment/mlp_policy",
                        "rlinf/models/embodiment/base_policy.py",
                        "rlinf/models/embodiment/modules/q_head.py",
                        "rlinf/models/embodiment/modules/utils.py",
                        "rlinf/models/embodiment/openpi/policies/piper_policy.py",
                        "rlinf/robotics/piper_runtime.py",
                        "rlinf/robotics/parts/arms/piper.py",
                    ]
                ),
                "runtime_versions": runtime_versions(),
                "runtime_source_sha256": runtime_source_fingerprints(),
            }
        )

    @property
    def budget(self) -> int:
        """Number of minibatches in the complete offline schedule."""
        t = self.cfg.training
        return t.bc_steps + t.critic_steps + t.joint_steps

    def phase(self, step: int) -> str:
        """Return the phase for a one-based minibatch index."""
        if step <= self.cfg.training.bc_steps:
            return "actor_bc"
        if step <= self.cfg.training.bc_steps + self.cfg.training.critic_steps:
            return "critic"
        return "joint"

    def _next_q(self, batch: dict, *, deterministic: bool) -> torch.Tensor:
        actions, _, _ = self.model.sac_forward(
            batch["next_obs"], deterministic=deterministic
        )
        chunk = batch["actions"].shape[1]
        actions = (
            self.critic_actions(actions.reshape_as(batch["actions"]))
            * action_mask(batch["next_valid_steps"], chunk)[..., None]
        )
        return self.target.sac_q_forward(batch["next_obs"], actions).min(-1).values

    def critic_actions(self, proposed: torch.Tensor) -> torch.Tensor:
        """Project Q inputs to sent targets, retaining raw actor outputs for BC.

        Affine quantile normalization makes this equivalent to decode, driver
        joint/gripper clipping, then encode. Apply padding masks afterwards.
        Clamp gradients are zero outside limits; no straight-through substitute.
        """
        if self.cfg.critic_action_projection == "none":
            return proposed
        return proposed.clamp(
            min=self.driver_action_bounds[0], max=self.driver_action_bounds[1]
        )

    def update(self) -> dict:
        """Consume one training batch, respecting actor-only and critic-only phases."""
        if self.step >= self.budget:
            raise ValueError("Offline update budget is exhausted.")
        t = self.cfg.training
        next_step = self.step + 1
        phase = self.phase(next_step)
        batch = self.replay.sample(t.batch_size, self.device)
        metrics = {"step": next_step, "phase": phase}
        if phase != "actor_bc":
            with torch.no_grad():
                target = discounted_target(
                    batch, self._next_q(batch, deterministic=False), t.gamma
                )
            self.critic_optimizer.zero_grad(set_to_none=True)
            q = self.model.sac_q_forward(batch["curr_obs"], batch["actions"])
            loss = (q - target[:, None]).square().mean()
            loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_params, t.grad_clip, error_if_nonfinite=True
            )
            self.critic_optimizer.step()
            with torch.no_grad():
                for target_param, param in zip(
                    self.target.q_head.parameters(), self.critic_params, strict=True
                ):
                    target_param.lerp_(param, t.tau)
            self.critic_steps += 1
            metrics["critic_loss"] = float(loss.detach())
            metrics["critic_grad_norm"] = float(critic_grad_norm)
        joint_step = next_step - t.bc_steps - t.critic_steps
        if phase == "actor_bc" or (
            phase == "joint" and joint_step % t.actor_every == 0
        ):
            progress = max(0.0, (joint_step - 1) / max(1, t.joint_steps - 1))
            dropout = t.dropout_end * progress
            bc_weight = t.bc_start + (t.bc_end - t.bc_start) * progress
            q_weight = t.q_end * progress
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.model.q_head.requires_grad_(False)
            try:
                predicted, _, _ = self.model.sac_forward(
                    actor_training_observation(batch),
                    deterministic=phase == "actor_bc",
                    apply_reference_dropout=phase == "joint",
                    reference_dropout_prob=dropout,
                )
                bc = masked_behavior_loss(predicted, batch)
                actor_loss = bc_weight * bc
                if phase == "joint":
                    masked = (
                        self.critic_actions(predicted.reshape_as(batch["actions"]))
                        * action_mask(batch["valid_steps"], self.cfg.chunk_length)[
                            ..., None
                        ]
                    )
                    actor_loss = (
                        actor_loss
                        - q_weight
                        * self.model.sac_q_forward(batch["curr_obs"], masked)[
                            ..., 0
                        ].mean()
                    )
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_params, t.grad_clip, error_if_nonfinite=True
                )
                self.actor_optimizer.step()
            finally:
                self.model.q_head.requires_grad_(True)
            self.actor_steps += 1
            metrics.update(
                actor_loss=float(actor_loss.detach()),
                bc_loss=float(bc.detach()),
                reference_dropout=dropout,
                bc_weight=bc_weight,
                q_weight=q_weight,
                actor_grad_norm=float(actor_grad_norm),
            )
        self.step = next_step
        return metrics

    def _checkpoint(self) -> Path:
        directory = self.output / "checkpoints"
        directory.mkdir(exist_ok=True)
        path = directory / f"step_{self.step:08d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "format": "piper-rlt-warmup-checkpoint-v1",
                "binding": self.binding,
                "model": self.model.state_dict(),
                "target": self.target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "step": self.step,
                "actor_steps": self.actor_steps,
                "critic_steps": self.critic_steps,
                "history": self.history,
                "sampler_rng": self.replay.rng_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None,
            },
            temporary,
        )
        temporary.replace(path)
        write_json(
            self.output / "latest.json",
            {"file": str(path.relative_to(self.output)), "sha256": file_sha256(path)},
        )
        for stale in sorted(directory.glob("step_*.pt"))[
            : -self.cfg.training.keep_checkpoints
        ]:
            stale.unlink()
        return path

    def _restore(self) -> None:
        latest = json.loads((self.output / "latest.json").read_text())
        path = (self.output / latest["file"]).resolve()
        if (
            not path.is_relative_to(self.output)
            or file_sha256(path) != latest["sha256"]
        ):
            raise ValueError("Warmup checkpoint fingerprint mismatch.")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            state["format"] != "piper-rlt-warmup-checkpoint-v1"
            or state["binding"] != self.binding
        ):
            raise ValueError(
                "Warmup config, implementation or cache changed; start a new output directory."
            )
        self.model.load_state_dict(state["model"])
        self.target.load_state_dict(state["target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.step, self.actor_steps, self.critic_steps = (
            state["step"],
            state["actor_steps"],
            state["critic_steps"],
        )
        self.history = state["history"]
        self.replay.restore_rng(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], self.device)
        # Any metrics after the durable checkpoint belong to the interrupted attempt.
        log = self.output / "metrics.jsonl"
        if log.exists():
            rows = []
            lines = log.read_text().splitlines(keepends=True)
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if index != len(lines) - 1 or line.endswith("\n"):
                        raise
                    logger.warning("Discarding an interrupted final metrics record.")
            log.write_text(
                "".join(
                    json.dumps(row) + "\n" for row in rows if row["step"] <= self.step
                )
            )

    @torch.inference_mode()
    def _episode_metrics(self, index: int, phase_id: int | None = None) -> dict:
        data = self.replay.episodes[index]
        eligible = self.replay.selected_rows[index]
        if phase_id is not None:
            eligible = eligible[data["phase_id"][eligible].numpy() == phase_id]
        count = len(eligible)
        rows = np.unique(
            np.r_[
                eligible[
                    np.linspace(
                        0,
                        count - 1,
                        min(count, self.cfg.validation.max_rows_per_episode),
                        dtype=int,
                    )
                ],
                eligible[data["terminated"][eligible].numpy()],
            ]
        ).tolist()
        batch = self.replay.batch([index] * len(rows), rows, self.device)
        predicted, _, _ = self.model.sac_forward(batch["curr_obs"], deterministic=True)
        predicted = predicted.reshape_as(batch["actions"])
        mask = action_mask(batch["valid_steps"], self.cfg.chunk_length)
        q = self.model.sac_q_forward(batch["curr_obs"], batch["actions"])
        actor_q = self.model.sac_q_forward(
            batch["curr_obs"], self.critic_actions(predicted) * mask[..., None]
        )
        next_q = self._next_q(batch, deterministic=True)
        td_target = discounted_target(batch, next_q, self.cfg.training.gamma)
        if not all(
            torch.isfinite(value).all() for value in (predicted, q, actor_q, td_target)
        ):
            raise FloatingPointError("Non-finite offline validation output.")
        terminal = batch["terminated"]
        reference = batch["curr_obs"]["ref_chunk"][:, : self.cfg.chunk_length]
        policy_mask = mask & ~batch["human"]
        errors = (predicted - reference).square().mean(-1)[policy_mask]
        human_mask = mask & batch["human"]
        human_errors = (predicted - batch["actions"]).square().mean(-1)[human_mask]
        all_q = torch.cat((q, actor_q))
        v = self.cfg.validation
        values = {
            "episode": index,
            "phase_id": phase_id,
            "outcome": self.replay.manifest["episodes"][index]["outcome"]
            if phase_id is None
            else self.replay.manifest["episodes"][index]["critical_phases"][phase_id][
                "outcome"
            ],
            "sampled_rows": len(rows),
            "actor_projection_fraction": float(
                (
                    (self.critic_actions(predicted) - predicted).abs().amax(-1)[mask]
                    > 1e-6
                )
                .float()
                .mean()
            ),
            "actor_mse": float(errors.mean()) if errors.numel() else None,
            "human_mse": float(human_errors.mean()) if human_errors.numel() else None,
            "human_ticks": int(human_mask.sum()),
            "td_mse": float(
                (q.min(-1).values[~terminal] - td_target[~terminal]).square().mean()
            )
            if (~terminal).any()
            else None,
            "terminal_mse": float(
                (q.min(-1).values[terminal] - td_target[terminal]).square().mean()
            ),
            "terminal_targets": td_target[terminal].cpu().tolist(),
            "q_gap": float((all_q[:, 0] - all_q[:, 1]).abs().mean()),
            "q_in_range_fraction": float(
                ((all_q >= v.q_low) & (all_q <= v.q_high)).float().mean()
            ),
        }
        decoded = self.codec.decode(predicted.cpu().numpy())
        decoded_ref = self.codec.decode(reference.cpu().numpy())
        selected = policy_mask.cpu().numpy()
        differences = np.abs(decoded - decoded_ref)[selected]
        values["joint_p95_deg"] = (
            float(np.percentile(differences[:, :6], 95)) if len(differences) else None
        )
        values["gripper_p95_mm"] = (
            float(np.percentile(differences[:, 6], 95)) if len(differences) else None
        )
        from rlinf.robotics.parts.arms.piper import PiperArm

        valid = mask.cpu().numpy()
        low, high = (
            np.rad2deg(PiperArm.JOINT_LIMITS_LOWER),
            np.rad2deg(PiperArm.JOINT_LIMITS_UPPER),
        )
        targets = decoded[valid]
        previous = np.concatenate(
            (batch["raw_state"].cpu().numpy()[:, None], decoded[:, :-1]), axis=1
        )
        delta = np.abs(decoded - previous)[valid]
        values["nominal_action_bounds"] = bool(
            np.isfinite(targets).all()
            and np.all(targets[:, :6] >= low - 0.001)
            and np.all(targets[:, :6] <= high + 0.001)
            and np.all(targets[:, 6] >= -0.01)
            and np.all(
                targets[:, 6] <= self.execution.gripper_max_width_m * 1000 + 0.01
            )
            and np.all(delta[:, :6] <= self.execution.max_joint_step_deg)
            and np.all(delta[:, 6] <= self.execution.max_gripper_step_mm)
        )
        return values

    def validate(self) -> dict:
        """Evaluate fixed episode rows without consuming training RNG or samples."""
        train = [
            self._episode_metrics(i, phase)
            for i, phase in self.replay.metric_groups("train")
        ]
        validation = [
            self._episode_metrics(i, phase)
            for i, phase in self.replay.metric_groups("validation")
        ]

        def mean(rows, key):
            values = [row[key] for row in rows if row[key] is not None]
            return float(np.mean(values)) if values else None

        def rmse(rows, key):
            value = mean(rows, key)
            return value**0.5 if value is not None else None

        # A constant predictor fit only to the training terminals is the baseline.
        baseline = float(np.mean([np.mean(row["terminal_targets"]) for row in train]))
        by_outcome = {}
        for outcome in ("success", "failure"):
            selected = [row for row in validation if row["outcome"] == outcome]
            by_outcome[outcome] = {
                "terminal_mse": mean(selected, "terminal_mse"),
                "baseline_mse": float(
                    np.mean(
                        [
                            np.mean(
                                (np.asarray(row["terminal_targets"]) - baseline) ** 2
                            )
                            for row in selected
                        ]
                    )
                )
                if selected
                else None,
            }
        covered = all(row["terminal_mse"] is not None for row in by_outcome.values())
        terminal_rmse = (
            float(np.mean([row["terminal_mse"] for row in by_outcome.values()]) ** 0.5)
            if covered
            else None
        )
        baseline_rmse = (
            float(np.mean([row["baseline_mse"] for row in by_outcome.values()]) ** 0.5)
            if covered
            else None
        )
        train_td, val_td = rmse(train, "td_mse"), rmse(validation, "td_mse")
        actor_rmse = rmse(validation, "actor_mse")
        joint = [
            row["joint_p95_deg"]
            for row in validation
            if row["joint_p95_deg"] is not None
        ]
        gripper = [
            row["gripper_p95_mm"]
            for row in validation
            if row["gripper_p95_mm"] is not None
        ]
        joint_p95, gripper_p95 = (
            max(joint) if joint else None,
            max(gripper) if gripper else None,
        )
        q_gap, q_range = (
            mean(validation, "q_gap"),
            mean(validation, "q_in_range_fraction"),
        )
        tc, vc = self.replay.counts("train"), self.replay.counts("validation")
        v = self.cfg.validation
        by_phase = {}
        for phase in sorted(
            {
                row["phase_id"]
                for row in train + validation
                if row["phase_id"] is not None
            }
        ):
            tr = [r for r in train if r["phase_id"] == phase]
            va = [r for r in validation if r["phase_id"] == phase]
            by_phase[str(phase)] = {
                "train_counts": {
                    o: sum(r["outcome"] == o for r in tr)
                    for o in ("success", "failure")
                },
                "validation_counts": {
                    o: sum(r["outcome"] == o for r in va)
                    for o in ("success", "failure")
                },
                "actor_rmse": rmse(va, "actor_mse"),
                "td_rmse": rmse(va, "td_mse"),
                "train_td_rmse": rmse(tr, "td_mse"),
                "terminal_rmse_observed": rmse(va, "terminal_mse"),
                "terminal_rmse_by_outcome": {
                    o: rmse([r for r in va if r["outcome"] == o], "terminal_mse")
                    for o in ("success", "failure")
                },
                "q_gap": mean(va, "q_gap"),
                "q_in_range_fraction": mean(va, "q_in_range_fraction"),
            }
            summary = by_phase[str(phase)]
            covered_phase = all(
                summary["validation_counts"][o] > 0 for o in ("success", "failure")
            )
            baseline_phase = (
                float(np.mean([np.mean(r["terminal_targets"]) for r in tr]))
                if tr
                else None
            )
            summary["both_validation_outcomes"] = covered_phase
            summary["terminal_rmse_balanced"] = (
                float(
                    np.sqrt(
                        np.mean(
                            [
                                summary["terminal_rmse_by_outcome"][o] ** 2
                                for o in ("success", "failure")
                            ]
                        )
                    )
                )
                if covered_phase
                else None
            )
            summary["constant_baseline_rmse_balanced"] = (
                float(
                    np.sqrt(
                        np.mean(
                            [
                                np.mean(
                                    [
                                        np.mean(
                                            (
                                                np.asarray(r["terminal_targets"])
                                                - baseline_phase
                                            )
                                            ** 2
                                        )
                                        for r in va
                                        if r["outcome"] == o
                                    ]
                                )
                                for o in ("success", "failure")
                            ]
                        )
                    )
                )
                if covered_phase and baseline_phase is not None
                else None
            )
        prior = [
            row["td_rmse"]
            for row in self.history
            if row["scheduled"] and row["td_rmse"] is not None
        ][-3:]
        trend = prior + [val_td] if val_td is not None else []
        worsening = len(trend) == 4 and all(b > a for a, b in zip(trend, trend[1:]))
        checks = {
            "real_data": not self.replay.manifest["annotations"].get(
                "test_only", False
            ),
            "episode_coverage": all(
                tc[x] >= v.min_train_per_outcome
                and vc[x] >= v.min_validation_per_outcome
                for x in ("success", "failure")
            ),
            "transition_coverage": tc["transitions"] >= v.min_train_transitions,
            "actor_reference": actor_rmse is not None and actor_rmse <= v.actor_rmse,
            "decoded_reference": joint_p95 is not None
            and joint_p95 <= v.joint_p95_deg
            and gripper_p95 <= v.gripper_p95_mm,
            "nominal_action_bounds": all(
                row["nominal_action_bounds"] for row in validation
            ),
            "terminal_value": terminal_rmse is not None
            and terminal_rmse <= v.terminal_rmse,
            "terminal_baseline": terminal_rmse is not None
            and terminal_rmse <= (1 - v.baseline_improvement) * baseline_rmse,
            "td_error": val_td is not None and val_td <= v.td_rmse,
            "td_generalization": train_td is not None
            and val_td is not None
            and val_td / max(0.01, train_td) <= v.td_ratio,
            "td_trend": not worsening,
            "q_gap": q_gap <= v.q_gap,
            "q_range": q_range >= v.q_in_range_fraction,
        }
        if self.replay.reward_scope == "critical_segment":
            checks["local_phase_coverage"] = all(
                row["train_counts"][o] >= v.min_train_per_outcome
                and row["validation_counts"][o] >= v.min_validation_per_outcome
                for row in by_phase.values()
                for o in ("success", "failure")
            )
        return {
            "step": self.step,
            "scheduled": self.step % self.cfg.training.eval_every == 0,
            "passed": all(checks.values()),
            "checks": checks,
            "actor_rmse": actor_rmse,
            "human_rmse": rmse(validation, "human_mse"),
            "human_validation_ticks": sum(row["human_ticks"] for row in validation),
            "joint_p95_deg": joint_p95,
            "gripper_p95_mm": gripper_p95,
            "terminal_rmse": terminal_rmse,
            "constant_baseline_rmse": baseline_rmse,
            "td_rmse": val_td,
            "train_td_rmse": train_td,
            "q_gap": q_gap,
            "q_in_range_fraction": q_range,
            "train_counts": tc,
            "validation_counts": vc,
            "by_outcome": by_outcome,
            "by_phase": by_phase,
            "reward_scope": self.replay.reward_scope,
            "phase_filter": self.cfg.phase_filter,
            "actor_projection_fraction": mean(validation, "actor_projection_fraction"),
            "episodes": validation,
        }

    def export(self) -> dict:
        """Export inference-compatible heads and a report bound to those weights."""
        directory = self.output / "actor"
        weights = directory / "model_state_dict/full_weights.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        temporary = weights.with_suffix(".tmp")
        torch.save(
            {k: v.detach().cpu() for k, v in self.model.state_dict().items()}, temporary
        )
        temporary.replace(weights)
        write_json(
            directory / "piper_rlt_actor.json",
            {
                "format": "rlinf-piper-rlt-offline-local-actor-v1"
                if self.replay.reward_scope == "critical_segment"
                else "rlinf-piper-rlt-actor-v1",
                "stage1_checkpoint_id": self.replay.manifest["stage1"]["checkpoint_id"],
                "model": self.replay.manifest["model"],
                "reward_scope": self.replay.reward_scope,
                "phase_filter": self.cfg.phase_filter,
                "critic_action_projection": self.cfg.critic_action_projection,
                "critic_warmup_steps": self.critic_steps,
            },
        )
        passed = offline_warmup_passed(
            self.history,
            step=self.step,
            budget=self.budget,
            interval=self.cfg.training.eval_every,
            required_passes=self.cfg.validation.consecutive_passes,
        )
        report = {
            "offline_step": self.step,
            "budget": self.budget,
            "actor_steps": self.actor_steps,
            "critic_steps": self.critic_steps,
            "offline_validation_passed": passed,
            "ready_for_deterministic_eval": passed
            and self.replay.reward_scope == "global_episode",
            "ready_for_online_exploration": False,
            "next_requirement": "Local heads are offline-only; implement and review critical-phase routing before hardware evaluation."
            if self.replay.reward_scope == "critical_segment"
            else "Operator-scored deterministic trials and episode-boundary online handoff.",
            "actor": str(directory),
            "weights_sha256": file_sha256(weights),
            "cache_fingerprint": self.replay.manifest["content_fingerprint"],
            "binding": self.binding,
            "validation": self.history[-1],
        }
        write_json(self.output / "readiness.json", report)
        return report

    def run(self, *, resume: bool = False, until_step: int | None = None) -> dict:
        """Train to the fixed budget or a requested checkpoint, with exact resume."""
        limit = self.budget if until_step is None else until_step
        if not 0 < limit <= self.budget:
            raise ValueError("until_step must be inside the fixed update budget.")
        with directory_lock(self.output):
            if resume:
                self._restore()
            elif any(path.name != ".lock" for path in self.output.iterdir()):
                raise FileExistsError(
                    "Warmup output is nonempty; use --resume or a new output."
                )
            if limit < self.step:
                raise ValueError("until_step precedes the restored checkpoint.")
            write_json(self.output / "config.json", asdict(self.cfg))
            start, first = time.monotonic(), self.step
            with (self.output / "metrics.jsonl").open("a") as log:
                while self.step < limit:
                    metrics = self.update()
                    if self.step % self.cfg.training.eval_every == 0:
                        report = self.validate()
                        self.history.append(report)
                        metrics["validation"] = report
                        log.write(json.dumps(metrics, allow_nan=False) + "\n")
                        log.flush()
                        logger.info(
                            "Offline step %s/%s (%s): validation=%s",
                            self.step,
                            self.budget,
                            metrics["phase"],
                            report["passed"],
                        )
                    if self.step % self.cfg.training.save_every == 0:
                        self._checkpoint()
                if not self.history or self.history[-1]["step"] != self.step:
                    # An extra final report is informative but does not count as a scheduled pass.
                    report = self.validate()
                    self.history.append(report)
                    log.write(
                        json.dumps(
                            {"step": self.step, "validation": report}, allow_nan=False
                        )
                        + "\n"
                    )
                checkpoint = self._checkpoint()
            result = self.export()
            result.update(
                checkpoint=str(checkpoint),
                elapsed_seconds=time.monotonic() - start,
                completed_this_run=self.step - first,
            )
            write_json(self.output / "result.json", result)
            return result
