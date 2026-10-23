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

"""Model registration, embeddings, inference adapters, and reward helpers."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models import get_model, register_model
from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenTransformer,
)
from rlinf.models.embodiment.openpi.apxinf_adapter import (
    OpenPIApxInfAdapter,
    _active_token_ids,
)
from rlinf.scheduler import Worker
from rlinf.utils.env_helpers import HistoryManager
from rlinf.utils.env_helpers.delay_sampler import (
    ConstantDelaySampler,
    DelaySampler,
    ExponentialDelaySampler,
    GaussianDelaySampler,
    UniformDelaySampler,
)


@pytest.mark.parametrize("reference_dropout", [0.0, 0.5, 1.0])
def test_piper_online_native_losses_match_masked_segment_warmup(reference_dropout):
    import copy

    from rlinf.algorithms.rlt.segment import (
        action_mask,
        actor_training_observation,
        discounted_target,
        masked_behavior_loss,
    )
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from rlinf.workers.actor.piper_rlt_worker import PiperRLTActor, segment_loss_batch

    # Process setup is exercised by the GPU smoke test; these are real model
    # forwards with the same native replay tensors passed to the worker.
    worker = object.__new__(PiperRLTActor)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"num_action_chunks": 10, "action_dim": 7}},
            "algorithm": {
                "gamma": 0.995,
                "bc_weight": 2.0,
                "q_weight": 0.1,
                "reference_dropout_prob": reference_dropout,
            },
            "piper": {"driver_action_bounds": [[-0.2] * 7, [0.2] * 7]},
        }
    )
    worker.model = RLTMLPPolicy(
        z_dim=8,
        proprio_dim=7,
        action_dim=7,
        num_action_chunks=10,
        ref_num_action_chunks=50,
    )
    worker.target_model = copy.deepcopy(worker.model).requires_grad_(False)
    obs = {
        "z_rl": torch.randn(3, 8),
        "proprio": torch.randn(3, 7),
        "ref_chunk": torch.randn(3, 50, 7),
    }
    batch = {
        "curr_obs": obs,
        "next_obs": obs,
        "actions": torch.zeros(3, 70),
        "intervene_flags": torch.zeros(3, 70, dtype=torch.bool),
        "rewards": torch.zeros(3, 10),
        "terminations": torch.zeros(3, 10, dtype=torch.bool),
        "forward_inputs": {
            "valid_steps": torch.tensor([10, 3, 4]),
            "discount_steps": torch.tensor([10, 3, 4]),
            "next_valid_steps": torch.tensor([3, 0, 0]),
        },
    }
    batch["rewards"][1, 2] = 1
    batch["terminations"][1, 2] = True
    batch["terminations"][2, 3] = True
    batch["intervene_flags"][1, :14] = True
    batch["actions"][1, :14] = 0.15
    data = segment_loss_batch(batch, 10, 7)
    original_ref = obs["ref_chunk"].clone()
    conditioned = actor_training_observation(data)
    torch.testing.assert_close(
        conditioned["ref_chunk"][1, :2], torch.full((2, 7), 0.15)
    )
    torch.testing.assert_close(obs["ref_chunk"], original_ref)
    torch.manual_seed(123)
    with torch.no_grad():
        predicted, _, _ = worker.model.sac_forward(obs)
        masked = (
            predicted.reshape(3, 10, 7).clamp(-0.2, 0.2)
            * action_mask(data["next_valid_steps"], 10)[..., None]
        )
        next_q = worker.target_model.sac_q_forward(obs, masked).min(-1).values
        target = discounted_target(data, next_q, 0.995)
        assert target[1].item() == pytest.approx(0.995**2)
        assert target[2].item() == 0
        expected = (
            (worker.model.sac_q_forward(obs, data["actions"]) - target[:, None])
            .square()
            .mean()
        )
    torch.manual_seed(123)
    actual, _ = worker.forward_critic(batch)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(p.grad is None for p in worker.model.backbone.parameters())
    assert any(p.grad is not None for p in worker.model.q_head.parameters())
    worker.model.zero_grad()
    torch.manual_seed(456)
    predicted, _, _ = worker.model.sac_forward(
        conditioned,
        apply_reference_dropout=True,
        reference_dropout_prob=reference_dropout,
    )
    masked = (
        predicted.reshape(3, 10, 7).clamp(-0.2, 0.2)
        * action_mask(data["valid_steps"], 10)[..., None]
    )
    expected_actor = (
        2 * masked_behavior_loss(predicted, data)
        - 0.1 * worker.model.sac_q_forward(obs, masked)[:, 0].mean()
    )
    torch.manual_seed(456)
    actual_actor, _, metrics = worker.forward_actor(batch)
    torch.testing.assert_close(actual_actor, expected_actor)
    assert metrics["reference_dropout_prob"] == reference_dropout
    torch.testing.assert_close(obs["ref_chunk"], original_ref)
    actual_actor.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in worker.model.backbone.parameters()
    )


@pytest.mark.parametrize("chunk", [30, 50])
def test_piper_reference_dropout_masks_whole_input_without_scaling_or_mutation(chunk):
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy

    model = RLTMLPPolicy(8, 7, 7, chunk, ref_num_action_chunks=50)
    obs = {
        "ref_chunk": torch.ones(128, 50, 7),
        "z_rl": torch.randn(128, 8),
        "proprio": torch.randn(128, 7),
    }
    original = model._actor_state(obs)
    torch.manual_seed(1234)
    dropped = model._actor_state(
        obs, apply_reference_dropout=True, reference_dropout_prob=0.5
    )
    reference = dropped[:, : chunk * 7]
    keep = (reference == 1).all(dim=1)
    zero = (reference == 0).all(dim=1)
    assert (keep | zero).all() and keep.any() and zero.any()
    torch.testing.assert_close(dropped[:, chunk * 7 :], original[:, chunk * 7 :])
    torch.testing.assert_close(obs["ref_chunk"], torch.ones(128, 50, 7))
    # Normal inference still uses the complete reference after training forwards.
    torch.testing.assert_close(model._actor_state(obs), original)


@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_openpi_stem_preserves_float32_values_gradients_and_checkpoint_keys(
    parameter_dtype, with_bias
):
    from rlinf.models.embodiment.openpi.modules.siglip import Float32Conv2d

    stem = Float32Conv2d(3, 8, kernel_size=2, stride=2, bias=with_bias).to(
        parameter_dtype
    )
    reference = torch.nn.Conv2d(3, 8, kernel_size=2, stride=2, bias=with_bias).to(
        parameter_dtype
    )
    reference.load_state_dict(stem.state_dict(), strict=True)
    actual_input = torch.randn(2, 3, 8, 8, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)
    actual = stem(actual_input)
    expected = torch.nn.functional.conv2d(
        reference_input.float(),
        reference.weight.float(),
        None if reference.bias is None else reference.bias.float(),
        stride=reference.stride,
        padding=reference.padding,
    )
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    assert torch.equal(actual_input.grad, reference_input.grad)
    for actual_parameter, reference_parameter in zip(
        stem.parameters(), reference.parameters(), strict=True
    ):
        assert torch.equal(actual_parameter.grad, reference_parameter.grad)


def test_openpi_siglip_forward_enters_stem_module_hooks(monkeypatch):
    from rlinf.models.embodiment.openpi.modules import siglip

    monkeypatch.setattr(
        siglip,
        "_decode_variant",
        lambda _: {
            "width": 8,
            "depth": 1,
            "num_heads": 2,
            "mlp_dim": 16,
            "patch_size": (14, 14),
        },
    )
    vision = siglip.SigLIPViT()
    entered = []
    hook = vision.stem.register_forward_hook(lambda *_: entered.append(True))
    try:
        tokens, _ = vision(torch.randn(1, 224, 224, 3))
        tokens.square().mean().backward()
    finally:
        hook.remove()
    assert entered == [True]
    assert tokens.shape == (1, 256, 8)
    assert vision.stem.weight.grad is not None
    assert torch.count_nonzero(vision.stem.weight.grad) > 0


@pytest.mark.parametrize("train_expert_only", [False, True])
def test_openpi_sft_factory_preserves_action_and_rlt_training(
    tmp_path, train_expert_only
):
    """Exercise checkpoint loading and freezing without allocating model weights."""
    from rlinf.models.embodiment.openpi import get_model as build_openpi
    from rlinf.models.embodiment.openpi.pi0 import Pi0
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi.rlt_config import OpenPiPytorchRLTConfig

    with torch.device("meta"):
        original = Pi0(
            Pi0Config(pi05=True, action_horizon=10),
            rlt_cfg=OpenPiPytorchRLTConfig(use_rlt=True),
        )
    torch.save(original.state_dict(), tmp_path / "full_weights.pt")
    cfg = OmegaConf.create(
        {
            "model_path": str(tmp_path),
            "precision": "fp32",
            "action_dim": 7,
            "num_action_chunks": 10,
            "num_steps": 4,
            "openpi": {
                "task": "sft",
                "train_expert_only": train_expert_only,
                "use_rlt": True,
                "torch_compile": False,
                "action_horizon": 10,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
            },
        }
    )
    with torch.device("meta"):
        loaded = build_openpi(cfg)
    loaded.train()
    backbone = list(loaded.img.parameters()) + list(loaded.llm.embedder.parameters())
    expert = []
    for block in loaded.llm.layers:
        for modules in (
            block.pre_attention_norms,
            block.pre_ffw_norms,
            block.mlps,
            block.attn.q_proj,
            block.attn.k_proj,
            block.attn.v_proj,
            block.attn.o_proj,
        ):
            for index, destination in ((0, backbone), (1, expert)):
                if modules[index] is not None:
                    destination.extend(modules[index].parameters())
    for index, destination in ((0, backbone), (1, expert)):
        if loaded.llm.final_norms[index] is not None:
            destination.extend(loaded.llm.final_norms[index].parameters())
    for module in (
        loaded.action_in_proj,
        loaded.action_out_proj,
        loaded.time_mlp_in,
        loaded.time_mlp_out,
        loaded.rlt_module,
    ):
        expert.extend(module.parameters())
    assert backbone and expert
    assert all(p.requires_grad == (not train_expert_only) for p in backbone)
    assert all(p.requires_grad for p in expert)
    assert {id(p) for p in backbone + expert} == {id(p) for p in loaded.parameters()}


@pytest.mark.parametrize("precision", [None, "bf16", "fp32"])
@pytest.mark.parametrize("checkpoint_format", ["pt", "safetensors"])
def test_openpi_factory_loads_saved_values_after_final_storage_dtype_is_set(
    tmp_path, monkeypatch, precision, checkpoint_format
):
    from safetensors.torch import save_file

    from rlinf.models.embodiment.openpi import get_model as build_openpi
    from rlinf.models.embodiment.openpi.pi0 import Pi0

    def small_model_init(self, config, **kwargs):
        # Keep the real Pi0 dtype conversion and real factory/checkpoint loaders.
        torch.nn.Module.__init__(self)
        self.llm = torch.nn.Linear(3, 3)
        self.img = torch.nn.Module()
        self.img.stem = torch.nn.Linear(3, 3)
        self.img.pos_embedding = torch.nn.Parameter(torch.empty(3))
        self.img.projection = torch.nn.Linear(3, 3)
        self.action_out_proj = torch.nn.Linear(3, 3)

    monkeypatch.setattr(Pi0, "__init__", small_model_init)
    source = Pi0(None)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(
                torch.linspace(0.1234567, 0.2345678, parameter.numel()).reshape_as(
                    parameter
                )
            )
    saved = source.state_dict()
    if checkpoint_format == "pt":
        torch.save(saved, tmp_path / "full_weights.pt")
    else:
        save_file(saved, tmp_path / "model.safetensors")
    cfg = OmegaConf.create(
        {
            "model_path": str(tmp_path),
            "precision": precision,
            "action_dim": 7,
            "num_action_chunks": 50,
            "num_steps": 10,
            "openpi": {
                "task": "sft",
                "use_rlt": False,
                "torch_compile": False,
                "train_expert_only": False,
                "action_horizon": 50,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
            },
        }
    )
    loaded = build_openpi(cfg)
    for name, value in loaded.state_dict().items():
        fp32 = precision == "fp32" or name.startswith(
            ("img.stem.", "img.pos_embedding", "action_out_proj.")
        )
        expected_dtype = torch.float32 if fp32 else torch.bfloat16
        assert value.dtype == expected_dtype, name
        assert torch.equal(value, saved[name].to(expected_dtype)), name


class _DummyModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class _DummyFSDPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _DummyBlock()
        self.head = torch.nn.Linear(4, 2)
        self.head._fsdp_wrap_name = "custom_head"


def test_custom_model_registration_smoke():
    model_type = f"custom_model_smoke_{int(time.time() * 1000)}"
    received = {"torch_dtype": None}

    def _builder(cfg, torch_dtype):
        received["torch_dtype"] = torch_dtype
        return _DummyModel()

    register_model(model_type, _builder, category="embodied")

    supported_model = SupportedModel(model_type)
    assert supported_model.value == model_type

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    model = get_model(cfg)

    assert isinstance(model, _DummyModel)
    assert received["torch_dtype"] == torch.float32


def test_custom_model_registration_with_fsdp_wrap_policy():
    model_type = f"custom_model_fsdp_{int(time.time() * 1000)}"

    def _builder(cfg, torch_dtype):
        return _DummyFSDPModel()

    register_model(
        model_type,
        _builder,
        category="embodied",
    )

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    fsdp_cfg = OmegaConf.create(
        {
            "wrap_policy": {
                "transformer_layer_cls_to_wrap": ["_DummyBlock"],
                "module_classes_to_wrap": ["_DummyBlock"],
                "no_split_names": ["custom_head"],
            },
            "use_orig_params": True,
        }
    )
    model = get_model(cfg)
    wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config=fsdp_cfg,
        is_lora=False,
        model_type=model_type,
    )

    assert wrap_policy is not None
    assert wrap_policy(module=model.block, recurse=False, nonwrapped_numel=0)
    assert wrap_policy(module=model.head, recurse=False, nonwrapped_numel=0)


def _make_model(*, prefix_seq_len: int = 5) -> RLTTokenTransformer:
    torch.manual_seed(0)
    return RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=prefix_seq_len,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )


def test_decoder_causal_mask_blocks_future_teacher_targets():
    model = _make_model()
    model.eval()
    rl_tokens = torch.randn(1, 1, model.embed_dim)
    targets = torch.randn(1, model.prefix_seq_len, model.input_dim)

    changed_targets = targets.clone()
    changed_targets[:, 2:] += 100.0

    original_output = model.decode(rl_tokens, targets)
    changed_output = model.decode(rl_tokens, changed_targets)

    # target[2:] enters decoder positions 3+, so positions 0..2 must not
    # change when causal attention prevents access to future positions.
    torch.testing.assert_close(
        original_output[:, :3],
        changed_output[:, :3],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(original_output[:, 3:], changed_output[:, 3:])


def test_loss_masks_trailing_padding():
    model = _make_model(prefix_seq_len=4)
    model.eval()
    prefix_embs = torch.randn(2, 4, model.input_dim)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    loss, _ = model.loss(prefix_embs, mask)
    reconstructed, _ = model.reconstruct(prefix_embs, mask)
    valid = mask.unsqueeze(-1).to(dtype=torch.float32)
    expected_loss = (
        torch.square(reconstructed.float() - prefix_embs.float()) * valid
    ).sum() / (valid.sum() * model.input_dim)
    torch.testing.assert_close(loss, expected_loss)

    changed_padding = prefix_embs.clone()
    changed_padding[~mask] += 1000.0
    changed_loss, _ = model.loss(changed_padding, mask)
    torch.testing.assert_close(loss, changed_loss, rtol=1e-5, atol=1e-5)


def test_reconstruct_output_shape_matches_prefix_embeddings():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(3, 4, model.input_dim)

    reconstructed, _ = model.reconstruct(prefix_embs)

    assert reconstructed.shape == prefix_embs.shape


def test_reconstruct_detaches_targets_but_trains_encoder_and_decoder():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(2, 4, model.input_dim, requires_grad=True)

    loss, _ = model.loss(prefix_embs)
    loss.backward()

    assert prefix_embs.grad is None
    encoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    decoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.decoder.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad_norm > 0
    assert decoder_grad_norm > 0


class _FakeValueExpert:
    def __init__(self, image_emb, lang_emb):
        self.image_emb = image_emb
        self.lang_emb = lang_emb

    def embed_image(self, image):
        return self.image_emb.to(device=image.device)

    def embed_language_tokens(self, tokens):
        return self.lang_emb.to(device=tokens.device)


def _load_value_critic_model(monkeypatch):
    value_model_dir = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/value_model/recap"
    )
    package_name = "value_model_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(value_model_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    module_name = f"{package_name}.modeling_critic"
    spec = importlib.util.spec_from_file_location(
        module_name,
        value_model_dir / "modeling_critic.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.ValueCriticModel


def test_value_model_does_not_rescale_gemma3_language_embeddings(monkeypatch):
    """Gemma3 embed_tokens already applies sqrt(hidden_size) internally."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.Gemma3ForCausalLM")

    ValueCriticModel = _load_value_critic_model(monkeypatch)

    hidden_size = 4
    image_emb = torch.zeros(1, 2, hidden_size)
    lang_emb = torch.arange(12, dtype=torch.float32).reshape(1, 3, hidden_size)

    model = SimpleNamespace(
        gradient_checkpointing_enabled=False,
        training=False,
        value_expert=_FakeValueExpert(image_emb=image_emb, lang_emb=lang_emb),
        _apply_checkpoint=lambda func, *args: func(*args),
    )

    prefix_embs, prefix_pad_masks = ValueCriticModel.embed_prefix(
        model,
        images=[torch.empty(1, 3, 8, 8)],
        img_masks=[torch.tensor([True])],
        lang_tokens=torch.tensor([[1, 2, 3]]),
        lang_masks=torch.tensor([[True, True, False]]),
    )

    torch.testing.assert_close(prefix_embs[:, 2:], lang_emb)
    torch.testing.assert_close(
        prefix_pad_masks,
        torch.tensor([[True, True, True, True, False]]),
    )


_STARVLA_UTILS_DIR = (
    Path(__file__).resolve().parents[2] / "rlinf/models/embodiment/starvla/utils"
)
_FRANKA_ACTION_STATS = {
    "q01": [-0.5] * 7,
    "q99": [0.5] * 7,
    "min": [-1.0] * 7,
    "max": [1.0] * 7,
    "mask": [True] * 6 + [False],
}


def _load_starvla_util(name: str) -> ModuleType:
    # The starvla package __init__ imports starVLA, which only its venv has.
    spec = importlib.util.spec_from_file_location(
        f"starvla_{name}_under_test", _STARVLA_UTILS_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("source", "bound"), [("q01q99", 0.5), ("minmax", 1.0)])
def test_starvla_action_stats_follow_the_configured_source(source, bound):
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    stats = action_space.resolve_action_norm_stats(
        model, "franka", action_dim=7, action_stats_source=source
    )

    np.testing.assert_array_equal(stats["q99"], [bound] * 7)
    np.testing.assert_array_equal(stats["q01"], [-bound] * 7)
    np.testing.assert_array_equal(stats["mask"], [True] * 6 + [False])


def test_starvla_action_stats_name_the_available_keys_for_an_unknown_key():
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    with pytest.raises(RuntimeError, match=r"available keys: \['franka'\]"):
        action_space.resolve_action_norm_stats(model, "libero_spatial", action_dim=7)


def test_starvla_action_stats_require_a_norm_stats_mapping():
    action_space = _load_starvla_util("action_space")

    with pytest.raises(RuntimeError, match="no usable 'norm_stats' mapping"):
        action_space.resolve_action_norm_stats(
            SimpleNamespace(norm_stats=None), "franka", action_dim=7
        )


def test_starvla_env_actions_keep_their_shape_and_map_the_libero_gripper(monkeypatch):
    action_space = _load_starvla_util("action_space")
    received_shapes = []

    def unnormalize_actions(actions, action_norm_stats):
        received_shapes.append(actions.shape)
        return actions

    tools = ModuleType("starVLA.model.tools")
    tools.FrameworkTools = SimpleNamespace(unnormalize_actions=unnormalize_actions)
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    normalized = np.zeros((2, 3, 7), dtype=np.float32)
    normalized[..., 0] = 0.25
    normalized[0, :, 6] = 1.0
    stats = {"q99": np.ones(7), "q01": -np.ones(7), "mask": np.ones(7, dtype=bool)}

    env_actions = action_space.unnormalize_actions_for_env(
        normalized, stats, policy_setup="libero"
    )

    # starVLA unnormalizes [T, action_dim]; the chunk layout comes back intact.
    assert received_shapes == [(6, 7)]
    assert env_actions.shape == (2, 3, 7)
    np.testing.assert_array_equal(env_actions[..., 0], 0.25)
    # LIBERO wants the 0/1 gripper as -1 (open) / +1 (closed).
    np.testing.assert_array_equal(env_actions[0, :, 6], -1.0)
    np.testing.assert_array_equal(env_actions[1, :, 6], 1.0)


def test_starvla_autocast_targets_the_worker_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    # CPU stands in for a non-CUDA accelerator such as an Ascend NPU.
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16


def test_starvla_autocast_is_a_noop_without_an_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    monkeypatch.setattr(Worker, "torch_device_type", None)

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cuda")


def test_starvla_gaussian_is_float32_and_keeps_the_gradient_path():
    accelerator = _load_starvla_util("accelerator")
    mean = torch.zeros(2, 3, dtype=torch.bfloat16, requires_grad=True)
    log_std = torch.nn.Parameter(torch.zeros(3))

    dist = accelerator.build_gaussian(mean, log_std.exp())
    sample = dist.rsample()

    assert dist.loc.dtype == dist.scale.dtype == sample.dtype == torch.float32
    dist.log_prob(sample.detach()).sum().backward()
    assert mean.grad is not None and mean.grad.dtype == torch.bfloat16
    assert log_std.grad is not None


class _QwenVisionPatchEmbed(torch.nn.Module):
    """Shape contract of Qwen2.5-VL PatchEmbed: Conv3d kernel == stride."""

    def __init__(self, in_channels=3, temporal=2, patch=4, embed_dim=8):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal
        self.patch_size = patch
        kernel = (temporal, patch, patch)
        self.proj = torch.nn.Conv3d(
            in_channels, embed_dim, kernel_size=kernel, stride=kernel, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(self.proj.weight.dtype))
        return hidden_states.view(-1, self.proj.out_channels)


def test_qwen_vl_linear_patch_embed_matches_conv3d_and_backprops():
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        _linear_patch_embed_forward,
    )

    torch.manual_seed(0)
    module = _QwenVisionPatchEmbed()
    patches = torch.randn(5, 3 * 2 * 4 * 4, requires_grad=True)

    conv_out = module(patches)
    linear_out = _linear_patch_embed_forward(module, patches)
    torch.testing.assert_close(linear_out, conv_out, rtol=1e-5, atol=1e-5)

    linear_out.sum().backward()
    assert module.proj.weight.grad is not None
    assert patches.grad is not None


def test_qwen_vl_linear_patch_embed_is_rebound_on_npu(monkeypatch):
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        _linear_patch_embed_forward,
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())
    original_forward = model[0].forward

    assert patch_vision_patch_embed(model) == 1
    assert model[0].forward.__func__ is _linear_patch_embed_forward
    assert original_forward.__func__ is not _linear_patch_embed_forward


def test_qwen_vl_linear_patch_embed_is_left_alone_on_nvidia(monkeypatch):
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())

    assert patch_vision_patch_embed(model) == 0
    assert model[0].forward.__func__ is _QwenVisionPatchEmbed.forward


_WAN_NPU_PATCHES = "rlinf.envs.sim.world_model.backend.npu_patches"


@pytest.fixture
def wan_dit(monkeypatch):
    """diffsynth's Wan DiT module, whose operators the NPU patches rebind."""
    from rlinf.utils.patcher import Patcher

    def flash_attention(q, k, v, num_heads, compatibility_mode=False):
        return q

    def rope_apply(x, freqs, num_heads):
        return x

    class RMSNorm(torch.nn.Module):
        def forward(self, x):
            return x

    dit = ModuleType("diffsynth.models.wan_video_dit")
    dit.flash_attention, dit.rope_apply, dit.RMSNorm = (
        flash_attention,
        rope_apply,
        RMSNorm,
    )
    for name in ("diffsynth", "diffsynth.models"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, dit.__name__, dit)
    yield dit
    Patcher.clear()


def _import_wan_npu_patches(monkeypatch, *, mindiesd: bool) -> ModuleType:
    """Import the Wan NPU patches on an Ascend stack with or without MindIE-SD."""
    vendor = {"torch_npu": ModuleType("torch_npu"), "mindiesd": None}
    if mindiesd:
        names = (
            "mindiesd",
            "mindiesd.layers",
            "mindiesd.layers.flash_attn",
            "mindiesd.layers.flash_attn.attention_forward",
        )
        vendor.update({name: ModuleType(name) for name in names})
        vendor["mindiesd"].rotary_position_embedding = lambda *args, **kwargs: None
        vendor[names[-1]].attention_forward = lambda *args, **kwargs: None
    for name, module in vendor.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.find_spec(_WAN_NPU_PATCHES)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _WAN_NPU_PATCHES, module)
    spec.loader.exec_module(module)
    return module


def _patch_like_wan_backend(npu_patches: ModuleType) -> None:
    """Run the patch sequence ``WanBackend._build_pipeline`` runs."""
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    npu_patches.apply_npu_patches(Patcher)
    Patcher.apply()


def _wan_operators(dit: ModuleType) -> tuple:
    return dit.flash_attention, dit.rope_apply, dit.RMSNorm.forward


def test_wan_npu_patches_leave_diffsynth_alone_on_nvidia(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators


def test_wan_npu_patches_log_why_mindiesd_is_unavailable(wan_dit, monkeypatch, caplog):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=False)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators
    assert "import of mindiesd halted" in caplog.text


def test_wan_npu_patches_rebind_the_dit_operators_on_every_build(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    kernels = (
        npu_patches.npu_flash_attention,
        npu_patches.npu_rope_apply,
        npu_patches.npu_rmsnorm_forward,
    )

    # Every WanBackend in the process repeats the sequence on the patched module.
    for _ in range(2):
        _patch_like_wan_backend(npu_patches)
        assert _wan_operators(wan_dit) == kernels


def _history_cfg():
    return OmegaConf.create(
        {
            "model": {
                "history_buffers": {
                    "main": {
                        "history_size": 2,
                        "min_history_size": 1,
                        "input_interval": 3,
                        "history_keys": ["main_images"],
                        "input_on_done": True,
                    }
                }
            }
        }
    )


def _append_step(manager: HistoryManager, value: int) -> None:
    manager.append_to_history_entries(
        {"main_images": torch.tensor([[value], [value + 10]])}
    )


def test_build_history_input_skips_between_interval_ticks():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_input == {}
    assert history_length == {}
    assert manager.history_counts == [2, 2]


def test_build_history_input_emits_on_interval_tick():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)
    _append_step(manager, 3)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_length == {"main": [2, 2]}
    assert history_input["main"]["main_images"][0] == [
        torch.tensor([2]),
        torch.tensor([3]),
    ]
    assert history_input["main"]["main_images"][1] == [
        torch.tensor([12]),
        torch.tensor([13]),
    ]


def _success_potential_state_machine():
    from rlinf.models.embodiment.reward.vlm_reward_model import (
        ShapedVLMRewardModel,
    )

    model = ShapedVLMRewardModel.__new__(ShapedVLMRewardModel)
    model.potential_gamma = 1.0
    model.potential_scale = 1.0
    model.potential_ema_alpha = 0.5
    model.potential_clip = 0.0
    model.success_threshold = 0.5
    model.success_bonus = 1.0
    model.success_confirmation_windows = 1
    model.gt_success_bonus = 0.0
    model.infer_micro_batch_size = 0
    model._previous_potentials = None
    model._success_fired = None
    model._success_streak = None
    return model


def test_empty_history_input_still_resets_shaping_state_on_done():
    model = _success_potential_state_machine()
    model._previous_potentials = torch.tensor([0.4, 0.8])
    model._success_fired = torch.tensor([True, True])
    model._success_streak = torch.tensor([3, 1], dtype=torch.int32)

    rewards = model.compute_reward(
        {
            "history_input": {},
            "dones": torch.tensor([True, False]),
        }
    )

    assert rewards.tolist() == pytest.approx([0.0, 0.0])
    assert torch.isnan(model._previous_potentials[0])
    assert float(model._previous_potentials[1]) == pytest.approx(0.8)
    assert model._success_fired.tolist() == [False, True]
    assert model._success_streak.tolist() == [0, 1]


VALUE_CLIP = 0.2
HUBER_DELTA = 10.0


def _critic_metrics(values, prev_values, returns, loss_mask=None):
    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=loss_mask,
    )
    return metrics


def test_value_clip_ratio_is_zero_when_no_update_is_clipped():
    prev_values = torch.zeros(4, 8)
    values = torch.full((4, 8), VALUE_CLIP / 2)
    returns = torch.zeros(4, 8)

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_clip_ratio_reports_the_fraction_of_clipped_updates():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    # Half of the entries move outside the trust region, half stay inside.
    values = torch.full((4, 8), VALUE_CLIP / 2)
    values[:, :4] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_grows_with_the_size_of_the_value_update():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)

    ratios = [
        float(
            _critic_metrics(torch.full((4, 8), scale), prev_values, returns)[
                "critic/value_clip_ratio"
            ]
        )
        for scale in (0.5 * VALUE_CLIP, 2 * VALUE_CLIP)
    ]

    assert ratios == [pytest.approx(0.0), pytest.approx(1.0)]


def test_value_clip_ratio_ignores_masked_out_entries():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    loss_mask[:, :2] = True

    # Every valid entry is clipped; every padded entry is not.
    values = torch.zeros(4, 8)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(1.0)


def test_value_clip_ratio_broadcasts_a_narrower_loss_mask():
    prev_values = torch.zeros(4, 8, 3)
    returns = torch.zeros(4, 8, 3)
    loss_mask = torch.zeros(4, 8, 1, dtype=torch.bool)
    loss_mask[:, :4] = True

    values = torch.zeros(4, 8, 3)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    # 2 of the 4 unmasked steps are clipped.
    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_is_zero_when_every_entry_is_masked_out():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    values = torch.full((4, 8), 10 * VALUE_CLIP)

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_loss_is_unchanged_by_the_metric_computation():
    torch.manual_seed(0)
    prev_values = torch.randn(4, 8)
    values = torch.randn(4, 8, requires_grad=True)
    returns = torch.randn(4, 8)

    loss, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=None,
    )

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -VALUE_CLIP, VALUE_CLIP
    )
    expected = torch.max(
        torch.nn.functional.huber_loss(
            values, returns, delta=HUBER_DELTA, reduction="none"
        ),
        torch.nn.functional.huber_loss(
            value_pred_clipped, returns, delta=HUBER_DELTA, reduction="none"
        ),
    ).mean()

    assert float(loss.detach()) == pytest.approx(float(expected.detach()), abs=1e-6)
    assert loss.requires_grad
    assert not metrics["critic/value_clip_ratio"].requires_grad


def test_create_builds_expected_sampler_types():
    constant = DelaySampler.create(
        OmegaConf.create({"type": "constant", "delay": 0.12})
    )
    uniform = DelaySampler.create(
        OmegaConf.create({"type": "uniform", "min_delay": 0.03, "max_delay": 0.08})
    )
    exponential = DelaySampler.create(
        OmegaConf.create({"type": "exponential", "rate": 0.5})
    )
    gaussian = DelaySampler.create(
        OmegaConf.create({"type": "gaussian", "mean": 0.20, "stddev": 0.03})
    )

    assert isinstance(constant, ConstantDelaySampler)
    assert isinstance(uniform, UniformDelaySampler)
    assert isinstance(exponential, ExponentialDelaySampler)
    assert isinstance(gaussian, GaussianDelaySampler)


def test_create_accepts_none():
    assert DelaySampler.create(None) is None


def test_same_seed_produces_same_sequence_per_sampler():
    first = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)
    second = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)

    assert first.sample(8) == second.sample(8)


def test_constant_sampler_uses_seconds_helpers():
    sampler = ConstantDelaySampler(delay=0.25)

    assert sampler.sample(3) == [0.25, 0.25, 0.25]
    assert sampler.sample_one() == 0.25


def test_gaussian_sampler_never_returns_negative_seconds():
    sampler = GaussianDelaySampler(mean=0, stddev=0.1, seed=0)

    assert all(delay >= 0 for delay in sampler.sample(100))


def test_invalid_ranges_raise_clear_errors():
    with pytest.raises(ValueError, match="min_delay must be <="):
        UniformDelaySampler(min_delay=0.2, max_delay=0.1)

    with pytest.raises(ValueError, match="rate must be > 0"):
        ExponentialDelaySampler(rate=0)


def test_num_samples_must_be_non_negative_int():
    sampler = ConstantDelaySampler(delay=1)

    with pytest.raises(TypeError, match="num_samples must be an int"):
        sampler.sample(1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_samples must be >= 0"):
        sampler.sample(-1)


class _FakeEnv:
    """Minimal non-gym env exposing the chunk_step/reset surface."""

    def chunk_step(self, *args, **kwargs):
        return "stepped"

    def reset(self, *args, **kwargs):
        return "obs", {}


# Mock gymnasium and its transitive imports for unit-test environments that
# do not install the embodied extras. A minimal gym.Wrapper shim is enough
# because InsertDelay only delegates to self.env.


class _FakeGymEnv:
    pass


class _FakeGymWrapper:
    def __init__(self, env):
        self.env = env


_fake_gym = MagicMock()
_fake_gym.Env = _FakeGymEnv
_fake_gym.Wrapper = _FakeGymWrapper

if "gymnasium" not in sys.modules:
    sys.modules["gymnasium"] = _fake_gym
if "imageio" not in sys.modules:
    sys.modules["imageio"] = MagicMock()


def _delayed_env(delay: float):
    from rlinf.envs.wrappers import InsertDelay

    return InsertDelay(
        _FakeEnv(), OmegaConf.create({"type": "constant", "delay": delay})
    )


def test_chunk_step_does_not_block_the_caller():
    env = _delayed_env(0.5)

    start = time.monotonic()
    assert env.chunk_step() == "stepped"
    elapsed = time.monotonic() - start

    # The delay is sampled, not slept: blocking here would stall the event loop.
    assert elapsed < 0.05


def test_wait_delay_waits_out_the_accumulated_delay():
    env = _delayed_env(0.05)
    env.chunk_step()
    env.chunk_step()

    start = time.monotonic()
    asyncio.run(env.wait_delay())
    elapsed = time.monotonic() - start

    # Both sampled delays are paid, never dropped.
    assert elapsed == pytest.approx(0.1, abs=0.03)


def test_wait_delay_yields_to_other_coroutines():
    env = _delayed_env(0.2)
    env.chunk_step()
    progressed = []

    async def main():
        async def ticker():
            for _ in range(4):
                await asyncio.sleep(0.01)
                progressed.append(1)

        await asyncio.gather(env.wait_delay(), ticker())

    asyncio.run(main())
    # A blocking sleep would have starved the ticker entirely.
    assert len(progressed) == 4


def test_wait_delay_is_a_noop_when_nothing_is_pending():
    env = _delayed_env(0.5)

    start = time.monotonic()
    asyncio.run(env.wait_delay())

    assert time.monotonic() - start < 0.05


def test_delay_metrics_report_every_sample():
    env = _delayed_env(0.03)
    env.chunk_step()
    env.reset()

    metrics = env.insert_delay_metrics()

    assert metrics.tolist() == pytest.approx([0.03, 0.03])
    assert env.insert_delay_metrics().numel() == 0


class _FakeApxInfModel:
    action_horizon = 10
    action_dim = 32
    num_views = 2
    image_size = 224

    def __init__(self, output_shape=(10, 32)):
        self.output_shape = output_shape
        self.calls = []
        self.closed = False

    def infer_rgb(self, rgb_u8, layout, token_ids, *, noise=None):
        self.calls.append((rgb_u8, layout, token_ids, noise))
        offset = len(self.calls) * 1000
        return (
            np.arange(np.prod(self.output_shape), dtype=np.float32).reshape(
                self.output_shape
            )
            + offset
        )

    def close(self):
        self.closed = True


class _FakeApxInfProcessor:
    def __init__(self):
        self.preprocess_calls = []
        self.postprocess_calls = []

    def preprocess_batch(self, env_obs, *, num_views, image_size):
        self.preprocess_calls.append((env_obs, num_views, image_size))
        prepared = []
        for index in range(len(env_obs["task_descriptions"])):
            prepared.append(
                {
                    "rgb_u8": np.full(
                        (num_views, image_size, image_size, 3),
                        index,
                        dtype=np.uint8,
                    ),
                    "token_ids": np.array([index, index + 1], dtype=np.uint32),
                    "state": np.full(32, index, dtype=np.float32),
                }
            )
        return prepared

    def postprocess_batch(self, normalized_actions, prepared):
        self.postprocess_calls.append((normalized_actions.copy(), prepared))
        return torch.from_numpy(normalized_actions[:, :5, :7].copy())


def _apxinf_model_cfg(**apxinf_overrides):
    apxinf = {
        "action_horizon": 10,
        "num_flow_steps": 5,
        "noise_source": "apxinf",
        "seed": 0,
        **apxinf_overrides,
    }
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": "/not/loaded/in/unit/test",
            "num_action_chunks": 5,
            "action_dim": 7,
            "openpi": {
                "config_name": "pi05_libero",
                "num_steps": 5,
                "noise_method": "flow_sde",
                "noise_level": 0.3,
            },
            "apxinf": apxinf,
        }
    )


def _apxinf_env_obs(batch_size=2):
    return {
        "main_images": torch.zeros(batch_size, 8, 8, 3, dtype=torch.uint8),
        "wrist_images": torch.ones(batch_size, 8, 8, 3, dtype=torch.uint8),
        "extra_view_images": None,
        "states": torch.zeros(batch_size, 8),
        "task_descriptions": [f"task {index}" for index in range(batch_size)],
    }


def _apxinf_adapter(*, model=None, processor=None, **apxinf_overrides):
    return OpenPIApxInfAdapter(
        _apxinf_model_cfg(**apxinf_overrides),
        "cpu",
        model=model or _FakeApxInfModel(),
        processor=processor or _FakeApxInfProcessor(),
    )


def test_apxinf_strips_openpi_prompt_padding_before_l1_inference():
    transformed = {
        "tokenized_prompt": np.array([2, 42, 108, 0, 0], dtype=np.int32),
        "tokenized_prompt_mask": np.array([True, True, True, False, False]),
    }

    tokens = _active_token_ids(transformed)

    np.testing.assert_array_equal(tokens, np.array([2, 42, 108], dtype=np.uint32))
    assert tokens.flags.c_contiguous


def test_apxinf_calls_l1_infer_rgb_and_delegates_pre_and_postprocessing():
    model = _FakeApxInfModel()
    processor = _FakeApxInfProcessor()
    adapter = _apxinf_adapter(model=model, processor=processor)
    env_obs = _apxinf_env_obs()

    actions, result = adapter.predict_action_batch(env_obs, mode="eval")

    assert actions.shape == (2, 5, 7)
    assert actions.dtype == torch.float32
    assert processor.preprocess_calls == [(env_obs, 2, 224)]
    assert len(model.calls) == 2
    assert model.calls[0][0].shape == (2, 224, 224, 3)
    assert model.calls[0][0].dtype == np.uint8
    assert model.calls[0][1] == "nhwc"
    assert model.calls[0][2].dtype == np.uint32
    assert model.calls[0][3] is None
    normalized = processor.postprocess_calls[0][0]
    assert normalized.shape == (2, 10, 32)
    assert len(result["apxinf_timing"]) == 2


def test_apxinf_explicit_noise_is_split_and_forwarded_exactly():
    model = _FakeApxInfModel()
    adapter = _apxinf_adapter(model=model, noise_source="observation")
    env_obs = _apxinf_env_obs()
    env_obs["noise"] = torch.arange(2 * 10 * 32, dtype=torch.float32).reshape(2, 10, 32)

    adapter.predict_action_batch(env_obs)

    np.testing.assert_array_equal(model.calls[0][3], env_obs["noise"][0].numpy())
    np.testing.assert_array_equal(model.calls[1][3], env_obs["noise"][1].numpy())


def test_apxinf_observation_noise_is_required():
    adapter = _apxinf_adapter(noise_source="observation")
    with pytest.raises(ValueError, match="requires env_obs"):
        adapter.predict_action_batch(_apxinf_env_obs())


def test_apxinf_observation_noise_does_not_override_other_noise_sources():
    env_obs = _apxinf_env_obs()
    explicit_noise = torch.full((2, 10, 32), 123.0)
    env_obs["noise"] = explicit_noise

    apxinf_model = _FakeApxInfModel()
    _apxinf_adapter(model=apxinf_model, noise_source="apxinf").predict_action_batch(
        env_obs
    )
    assert all(call[3] is None for call in apxinf_model.calls)

    torch_model = _FakeApxInfModel()
    torch_adapter = _apxinf_adapter(model=torch_model, noise_source="torch")
    torch_adapter.predict_action_batch(env_obs)
    assert all(call[3] is not None for call in torch_model.calls)
    for index, call in enumerate(torch_model.calls):
        assert not np.array_equal(call[3], explicit_noise[index].numpy())


def test_apxinf_torch_noise_is_reproducible_and_has_model_shape():
    model_a = _FakeApxInfModel()
    model_b = _FakeApxInfModel()
    adapter_a = _apxinf_adapter(model=model_a, noise_source="torch", seed=7)
    adapter_b = _apxinf_adapter(model=model_b, noise_source="torch", seed=7)

    adapter_a.predict_action_batch(_apxinf_env_obs())
    adapter_b.predict_action_batch(_apxinf_env_obs())

    assert model_a.calls[0][3].shape == (10, 32)
    np.testing.assert_array_equal(model_a.calls[0][3], model_b.calls[0][3])
    np.testing.assert_array_equal(model_a.calls[1][3], model_b.calls[1][3])


def test_apxinf_rejects_bad_normalized_action_shape():
    adapter = _apxinf_adapter(model=_FakeApxInfModel(output_shape=(10, 7)))
    with pytest.raises(ValueError, match="normalized actions have shape"):
        adapter.predict_action_batch(_apxinf_env_obs(batch_size=1))


def test_apxinf_rejects_mismatched_openpi_and_apxinf_flow_steps():
    with pytest.raises(ValueError, match="must match OpenPI num_steps"):
        _apxinf_adapter(num_flow_steps=10)


def test_apxinf_rejects_training_mode():
    adapter = _apxinf_adapter()
    with pytest.raises(ValueError, match="eval-only"):
        adapter.predict_action_batch(_apxinf_env_obs(), mode="train")


def test_apxinf_close_delegates_to_model():
    model = _FakeApxInfModel()
    adapter = _apxinf_adapter(model=model)
    adapter.close()
    assert model.closed


def _stub_apxinf_robo(monkeypatch, resolved_tactics=None):
    """Install a fake ``apxinf_robo`` and record what ``_load_model`` asks it for."""
    seen = {}

    def load_bare_model(path, **kwargs):
        seen["path"] = path
        seen["kwargs"] = kwargs
        return _FakeApxInfModel()

    def resolve_tactics(device, precision, **kwargs):
        seen["resolve"] = {"device": device, "precision": precision, **kwargs}
        return resolved_tactics

    module = ModuleType("apxinf_robo")
    module.load_bare_model = load_bare_model
    engine = ModuleType("apxinf_robo.engine")
    engine.resolve_tactics = resolve_tactics
    module.engine = engine
    monkeypatch.setitem(sys.modules, "apxinf_robo", module)
    monkeypatch.setitem(sys.modules, "apxinf_robo.engine", engine)
    return seen


def test_apxinf_loads_through_the_apxinf_robo_l1_entry_point(monkeypatch):
    seen = _stub_apxinf_robo(monkeypatch)

    OpenPIApxInfAdapter(_apxinf_model_cfg(), "cpu", processor=_FakeApxInfProcessor())

    assert seen["path"] == Path("/not/loaded/in/unit/test")
    kwargs = seen["kwargs"]
    assert kwargs["model"] == "pi05"
    assert kwargs["device"] == "cpu"
    assert kwargs["precision"] == "bf16"
    assert kwargs["action_horizon"] == 10
    assert kwargs["num_flow_steps"] == 5
    assert kwargs["sampling_seed"] == 0
    # Left out so load_bare_model selects the tuned tactics.
    assert "tactics" not in kwargs
    assert "resolve" not in seen


def test_apxinf_a_configured_tactics_file_wins_over_the_default_selection(monkeypatch):
    seen = _stub_apxinf_robo(monkeypatch)

    OpenPIApxInfAdapter(
        _apxinf_model_cfg(tactics="/mine.json"), "cpu", processor=_FakeApxInfProcessor()
    )

    assert seen["kwargs"]["tactics"] == "/mine.json"
    assert "resolve" not in seen


def test_apxinf_an_explicit_weights_file_resolves_tactics_from_the_checkpoint_dir(
    monkeypatch,
):
    seen = _stub_apxinf_robo(monkeypatch, resolved_tactics="/ckpt/tactics.json")

    OpenPIApxInfAdapter(
        _apxinf_model_cfg(checkpoint="/ckpt/model-00001-of-00002.safetensors"),
        "cpu",
        processor=_FakeApxInfProcessor(),
    )

    # The weights file goes to the loader, the directory to the tactics lookup:
    # keying the lookup on the file would miss a checkpoint-local tactics.json.
    assert seen["path"] == Path("/ckpt/model-00001-of-00002.safetensors")
    assert seen["resolve"]["model_dir"] == Path("/not/loaded/in/unit/test")
    assert seen["resolve"]["precision"] == "bf16"
    assert seen["kwargs"]["tactics"] == "/ckpt/tactics.json"


def test_apxinf_a_missing_apxinf_robo_names_what_to_install(monkeypatch):
    # ``None`` in sys.modules is how CPython marks an import as unavailable.
    monkeypatch.setitem(sys.modules, "apxinf_robo", None)

    with pytest.raises(ImportError, match="apxinf_robo"):
        OpenPIApxInfAdapter(
            _apxinf_model_cfg(), "cpu", processor=_FakeApxInfProcessor()
        )


def _cosmos3_sglang_cfg(batching_max_size=None):
    server = (
        {} if batching_max_size is None else {"batching_max_size": batching_max_size}
    )
    return OmegaConf.create(
        {
            "rollout": {
                "model": {"num_action_chunks": 16},
                "sglang": {"server": server},
            },
            "env": {"eval": {"env_type": "libero"}},
        }
    )


def test_cosmos3_sglang_requests_group_envs_by_prompt_up_to_the_server_batch():
    # The Cosmos3 server rejects a batch whose prompts tokenize to different
    # lengths, and one larger than its --batching-max-size.
    from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter

    adapter = Cosmos3SGLangAdapter(_cosmos3_sglang_cfg(batching_max_size=2), rank=0)
    tasks = ["pick the bowl", "open the drawer", "pick the bowl", "pick the bowl"]

    groups = adapter.request_groups({"task_descriptions": tasks})

    assert groups == [[0, 2], [3], [1]]


def test_cosmos3_sglang_requests_default_to_one_env_like_sglang():
    from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter

    adapter = Cosmos3SGLangAdapter(_cosmos3_sglang_cfg(), rank=0)

    groups = adapter.request_groups({"task_descriptions": ["a", "a", "a"]})

    assert groups == [[0], [1], [2]]


def test_sglang_request_groups_round_trip_to_env_order():
    from rlinf.models.embodiment.sglang_adapter import (
        gather_env_rows,
        select_env_rows,
    )

    env_obs = {
        "main_images": torch.arange(5).view(5, 1),
        "task_descriptions": ["a", "b", "a", "a", "b"],
        "wrist_images": None,
        "_rlinf_stage_id": 1,
    }
    groups = [[0, 2], [3], [1, 4]]

    parts = [select_env_rows(env_obs, group) for group in groups]
    assert parts[2]["task_descriptions"] == ["b", "b"]
    assert parts[0]["_rlinf_stage_id"] == 1 and parts[0]["wrist_images"] is None

    # Each "request" answers with its envs' rows; gathering restores env order.
    order = [index for group in groups for index in group]
    actions = gather_env_rows([p["main_images"] * 10 for p in parts], order)
    info = gather_env_rows([{"x": {"y": p["main_images"]}} for p in parts], order)
    assert torch.equal(actions, torch.arange(5).view(5, 1) * 10)
    assert torch.equal(info["x"]["y"], torch.arange(5).view(5, 1))


def test_cosmos3_sglang_response_keeps_every_env_in_input_order():
    # The action envelope carries one ``data`` entry per env of the request.
    from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter

    adapter = Cosmos3SGLangAdapter(_cosmos3_sglang_cfg(), rank=0)

    def entry(input_index, gripper):
        values = np.zeros((16, 12), dtype=np.float32)
        values[:, 3] = 1.0  # rot6d columns (1, 0, 0) and (0, 1, 0): no rotation
        values[:, 7] = 1.0
        values[:, 9] = gripper
        return {"input_index": input_index, "action": {"values": values}}

    actions, _ = adapter.parse_response(
        {"data": [entry(1, 0.5), entry(0, -0.5)]}, state={}
    )

    assert actions.shape == (2, 16, 7)
    assert torch.all(actions[0, :, 6] == -0.5)
    assert torch.all(actions[1, :, 6] == 0.5)


def _piper_test_checkpoint(tmp_path, config_name="pi05_piper"):
    import json

    from rlinf.models.embodiment.piper_inference import save_piper_inference_assets

    root = tmp_path / "actor"
    (root / "model_state_dict").mkdir(parents=True)
    torch.save(
        {"rlt_module.weight": torch.ones(2)}, root / "model_state_dict/full_weights.pt"
    )
    stats = tmp_path / "norm_stats.json"
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    key: {
                        "mean": [0.0] * 7,
                        "std": [1.0] * 7,
                        "q01": [0.0] * 7,
                        "q99": [1.0] * 6 + [0.07],
                    }
                    for key in ("state", "actions")
                }
            }
        )
    )
    cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": "/server/base",
            "precision": "bf16",
            "openpi": {"config_name": config_name, "task": "sft", "use_rlt": True},
            "openpi_data": {"norm_stats_path": str(stats)},
        }
    )
    save_piper_inference_assets(root, cfg, fps=30)
    return root


@pytest.mark.parametrize("config_name", ["pi05_piper", "pi05_piper_h50_state"])
def test_piper_checkpoint_assets_survive_relocation_and_detect_mismatched_stats(
    tmp_path,
    config_name,
):
    import shutil

    from rlinf.models.embodiment.piper_inference import load_piper_inference_config

    root = _piper_test_checkpoint(tmp_path, config_name)
    download = tmp_path / "download/actor"
    shutil.copytree(root, download)
    cfg, manifest = load_piper_inference_config(download.parent)
    assert cfg.model_path == str(download)
    assert manifest["fps"] == 30
    assert cfg.openpi.config_name == config_name
    assert cfg.openpi_data.norm_stats_path == str(
        download / "assets/piper/norm_stats.json"
    )
    stats = download / "assets/piper/norm_stats.json"
    stats.write_text(stats.read_text() + " ")
    with pytest.raises(ValueError, match="Normalization does not match"):
        load_piper_inference_config(download)


def test_piper_rlt_actor_metadata_binds_the_exact_stage1(tmp_path):
    import json

    from rlinf.models.embodiment.piper_inference import (
        load_piper_inference_config,
        save_piper_rlt_actor_assets,
    )

    root = _piper_test_checkpoint(tmp_path)
    actor_config = OmegaConf.create(
        {
            "model_type": "rlt_mlp_policy",
            "z_dim": 8,
            "proprio_dim": 7,
            "action_dim": 7,
            "num_action_chunks": 2,
        }
    )
    save_piper_rlt_actor_assets(root, actor_config, root)
    metadata = json.loads((root / "piper_rlt_actor.json").read_text())
    _, manifest = load_piper_inference_config(root)
    assert metadata["stage1_checkpoint_id"] == manifest["checkpoint_id"]
    assert metadata["model"]["model_type"] == "rlt_mlp_policy"


@pytest.mark.parametrize("horizon", [10, 50])
def test_piper_shared_local_loader_requires_manual_opt_in_and_binds_weights(
    tmp_path, monkeypatch, horizon
):
    import json

    from rlinf.models.embodiment import openpi
    from rlinf.models.embodiment.mlp_policy import get_model
    from rlinf.models.embodiment.piper_inference import PiperRLTPolicy

    root = _piper_test_checkpoint(tmp_path)
    stage1 = json.loads((root / "piper_inference.json").read_text())
    actor_root = tmp_path / "stage2"
    (actor_root / "model_state_dict").mkdir(parents=True)
    config = {
        "model_type": "rlt_mlp_policy",
        "z_dim": 8,
        "proprio_dim": 7,
        "action_dim": 7,
        "num_action_chunks": horizon,
        "ref_num_action_chunks": 50,
        "add_q_head": True,
    }
    metadata = {
        "format": "rlinf-piper-rlt-offline-local-actor-v1",
        "stage1_checkpoint_id": stage1["checkpoint_id"],
        "reward_scope": "critical_segment",
        "phase_filter": None,
        "model": config,
    }
    path = actor_root / "piper_rlt_actor.json"
    path.write_text(json.dumps(metadata))
    model = get_model(OmegaConf.create(config), torch_dtype=torch.float32)
    weights = actor_root / "model_state_dict/full_weights.pt"
    torch.save(model.state_dict(), weights)

    class FeatureModel(torch.nn.Module):
        action_chunk = 50

        def extract_rlt_obs(self, observation):
            return {
                "z_rl": torch.zeros(1, 8),
                "proprio": torch.zeros(1, 7),
                "ref_chunk": torch.zeros(1, 50, 7),
            }

    monkeypatch.setattr(openpi, "get_model", lambda cfg: FeatureModel())
    with pytest.raises(ValueError, match="experimental opt-in"):
        PiperRLTPolicy(root, actor=actor_root, device="cpu")
    policy = PiperRLTPolicy(
        root,
        actor=actor_root,
        device="cpu",
        manual_critical=True,
        allow_offline_actor=True,
    )
    assert policy.action_horizon == 50 and policy.actor_horizon == horizon
    sample = {"observation.state": np.zeros(7)}
    for camera in ("front", "front_2", "wrist"):
        sample[f"observation.images.{camera}"] = np.zeros((8, 9, 3), np.uint8)
    assert policy.infer(sample, "USB", actor_enabled=False).shape == (50, 7)
    assert policy.infer(sample, "USB", actor_enabled=True).shape == (horizon, 7)
    paired, diagnostics = policy.infer_with_features(
        sample, "USB", actor_enabled=True, mode="eval"
    )
    np.testing.assert_array_equal(
        paired, policy.infer(sample, "USB", actor_enabled=True)
    )
    assert diagnostics["replay_features"]["ref_chunk"].shape == (50, 7)
    with pytest.raises(ValueError, match="explicit actor_enabled"):
        policy.infer(sample, "USB")
    original_manifest = policy.manifest
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    torch.save(model.state_dict(), weights)
    changed = PiperRLTPolicy(
        root,
        actor=actor_root,
        device="cpu",
        manual_critical=True,
        allow_offline_actor=True,
    )
    assert changed.manifest != original_manifest
    metadata["format"] = "rlinf-piper-rlt-online-segment-actor-v1"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="manual critical"):
        PiperRLTPolicy(root, actor=actor_root, device="cpu")
    online = PiperRLTPolicy(root, actor=actor_root, device="cpu", manual_critical=True)
    assert online.infer(sample, "USB", actor_enabled=True, mode="train").shape == (
        horizon,
        7,
    )
    with pytest.raises(ValueError, match="mode must"):
        online.infer(sample, "USB", actor_enabled=True, mode="unknown")
    metadata["phase_filter"] = 0
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="specialist routing"):
        PiperRLTPolicy(
            root,
            actor=actor_root,
            device="cpu",
            manual_critical=True,
            allow_offline_actor=True,
        )


def test_piper_online_requires_offline_data_and_trained_critic_for_c50(
    tmp_path, piper_local_warmup_cache
):
    import json

    from omegaconf import OmegaConf

    from rlinf.runners.piper_online_runner import online_config

    stage1 = _piper_test_checkpoint(tmp_path, "pi05_piper_h50_state")
    asset = stage1 / "piper_inference.json"
    manifest = json.loads(asset.read_text())
    manifest["model"]["precision"] = "fp32"
    manifest["model"]["openpi"]["action_chunk"] = 50
    asset.write_text(json.dumps(manifest))
    actor = tmp_path / "heads"
    (actor / "model_state_dict").mkdir(parents=True)
    torch.save({}, actor / "model_state_dict/full_weights.pt")
    metadata = {
        "format": "rlinf-piper-rlt-offline-local-actor-v1",
        "stage1_checkpoint_id": manifest["checkpoint_id"],
        "reward_scope": "critical_segment",
        "phase_filter": None,
        "critic_action_projection": "driver_limits",
        "critic_warmup_steps": 0,
        "model": {
            "model_type": "rlt_mlp_policy",
            "precision": "fp32",
            "num_action_chunks": 50,
            "ref_num_action_chunks": 50,
            "action_dim": 7,
            "proprio_dim": 7,
            "z_dim": 2048,
        },
    }
    path = actor / "piper_rlt_actor.json"
    path.write_text(json.dumps(metadata))
    kwargs = {
        "stage1": stage1,
        "actor": actor,
        "hardware": {"execution": {"gripper_max_width_m": 0.07}},
        "task": "TEST ONLY",
        "output": tmp_path / "online",
        "allow_offline_actor": True,
    }
    profile = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/piper_rlt_online.yaml"
    )
    with pytest.raises(ValueError, match="warmed-up critic"):
        online_config(profile, **kwargs)
    metadata["critic_warmup_steps"] = 10000
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="offline-cache"):
        online_config(profile, **kwargs)
    cache_manifest_path = piper_local_warmup_cache / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    cache_manifest["stage1"] = manifest
    cache_manifest_path.write_text(json.dumps(cache_manifest))
    with pytest.raises(ValueError, match="model mismatch"):
        online_config(profile, **kwargs, offline_cache=piper_local_warmup_cache)
    # The config layer checks geometry; the cache loader separately verifies shards.
    cache_manifest["model"]["num_action_chunks"] = 50
    cache_manifest_path.write_text(json.dumps(cache_manifest))
    cfg = online_config(profile, **kwargs, offline_cache=piper_local_warmup_cache)
    assert cfg.actor.model.num_action_chunks == 50
    assert cfg.algorithm.demo_buffer.sample_window_size == 0
    assert not cfg.algorithm.demo_buffer.add_interventions
    assert cfg.algorithm.demo_buffer.full_batch
    snapshot = tmp_path / "online.yaml"
    OmegaConf.save(cfg, snapshot)
    cache_manifest_path.unlink()
    restored = online_config(snapshot, **kwargs, resume=True)
    assert OmegaConf.to_container(restored, resolve=True) == OmegaConf.to_container(
        cfg, resolve=True
    )


def test_piper_rollout_installs_only_complete_valid_monotonic_snapshots(tmp_path):
    import json

    from rlinf.data.datasets.rlt_warmup import file_sha256, write_json
    from rlinf.workers.rollout.piper_rlt_worker import PiperRLTRollout

    actor = torch.nn.Linear(2, 2)
    worker = object.__new__(PiperRLTRollout)
    worker.version = 0
    worker.policy = SimpleNamespace(
        actor=actor,
        infer_with_reference=lambda *args, **kwargs: (
            actor.weight.detach().clone(),
            {},
        ),
    )
    worker.enable_snapshots(str(tmp_path), "test-binding")
    write_json(tmp_path / "latest.json", {"binding": "test-binding", "version": 0})
    original, diagnostics = worker.infer_with_reference({}, "USB", actor_enabled=True)
    assert diagnostics["policy_version"] == 0
    target = tmp_path / "checkpoint_001/actor/model_state_dict/full_weights.pt"
    target.parent.mkdir(parents=True)
    state = {key: torch.ones_like(value) for key, value in actor.state_dict().items()}
    torch.save(state, target)
    # Unpublished files cannot change inference.
    unchanged, _ = worker.infer_with_reference({}, "USB", actor_enabled=True)
    torch.testing.assert_close(unchanged, original)
    pointer = {
        "binding": "test-binding",
        "version": 2,
        "checkpoint": "checkpoint_001",
        "snapshot_sha256": file_sha256(target),
    }
    write_json(tmp_path / "latest.json", pointer)
    # Publishing a newer checkpoint must not alter the active episode.
    unchanged, diagnostics = worker.infer_with_reference({}, "USB", actor_enabled=True)
    torch.testing.assert_close(unchanged, original)
    assert diagnostics["policy_version"] == 0
    assert worker.begin_episode() == {"version": 2}
    updated, diagnostics = worker.infer_with_reference({}, "USB", actor_enabled=True)
    assert diagnostics["policy_version"] == 2 and (updated == 1).all()

    state["weight"][0, 0] = float("nan")
    torch.save(state, target)
    pointer.update(version=3, snapshot_sha256=file_sha256(target))
    write_json(tmp_path / "latest.json", pointer)
    with pytest.raises(ValueError, match="tensor contract"):
        worker.begin_episode()
    assert worker.version == 2 and (actor.weight == 1).all()
    pointer["snapshot_sha256"] = "wrong checksum"
    write_json(tmp_path / "latest.json", pointer)
    with pytest.raises(ValueError, match="checksum"):
        worker.begin_episode()
    pointer["version"] = 1
    write_json(tmp_path / "latest.json", pointer)
    unchanged, diagnostics = worker.infer_with_reference({}, "USB", actor_enabled=True)
    assert diagnostics["policy_version"] == 2 and (unchanged == 1).all()
    assert worker.begin_episode() == {"version": 2}
    pointer["binding"] = "another-run"
    (tmp_path / "latest.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="another online run"):
        worker.begin_episode()


def test_piper_rlt_inference_uses_native_actor_route_and_decodes_once(tmp_path):
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from rlinf.models.embodiment.openpi.policies.piper_policy import PiperActionCodec
    from rlinf.models.embodiment.piper_inference import PiperRLTPolicy

    root = _piper_test_checkpoint(tmp_path)
    codec = PiperActionCodec(root / "assets/piper/norm_stats.json")
    features = {
        "z_rl": torch.zeros(1, 8),
        "proprio": torch.zeros(1, 7),
        "ref_chunk": torch.arange(350).reshape(1, 50, 7).float() / 350,
    }

    class FeatureModel:
        calls = 0
        action_chunk = 50

        def extract_rlt_obs(self, obs):
            self.calls += 1
            assert obs["extra_view_images"].shape == (1, 2, 8, 9, 3)
            return features

    policy = PiperRLTPolicy.__new__(PiperRLTPolicy)
    policy.feature_model = FeatureModel()
    policy.actor = RLTMLPPolicy(
        z_dim=8, proprio_dim=7, action_dim=7, num_action_chunks=10
    )
    policy.actor.eval()
    policy.codec = codec
    policy.action_horizon = 50
    policy.actor_horizon = 10
    sample = {"observation.state": np.zeros(7)}
    for name in ("front", "front_2", "wrist"):
        sample[f"observation.images.{name}"] = np.zeros((8, 9, 3), np.uint8)
    expected, _ = policy.actor.predict_action_batch(features, mode="eval")
    np.testing.assert_allclose(
        policy.infer(sample, "insert USB"), codec.decode(expected[0].numpy()), atol=1e-5
    )
    assert policy.feature_model.calls == 1
    for mode in ("eval", "train"):
        torch.manual_seed(17)
        plain = policy.infer(sample, "insert USB", mode=mode)
        torch.manual_seed(17)
        calls = policy.feature_model.calls
        paired, diagnostics = policy.infer_with_reference(
            sample, "insert USB", mode=mode
        )
        assert policy.feature_model.calls == calls + 1
        np.testing.assert_array_equal(paired, plain)
        np.testing.assert_allclose(
            diagnostics["reference_actions"],
            codec.decode(features["ref_chunk"][0].numpy()),
        )
        assert diagnostics["reference_horizon"] == 50
        assert diagnostics["observation_state"] == [0] * 7
        assert diagnostics["action_units"] == ["deg"] * 6 + ["mm"]
        assert diagnostics["mode"] == mode
    reference, diagnostics = policy.infer_with_reference(
        sample, "insert USB", actor_enabled=False
    )
    np.testing.assert_allclose(reference, diagnostics["reference_actions"])
    # Exercise the real worker method through the controller's RPC boundary.
    from rlinf.runners.piper_online_runner import PiperOnlinePolicy
    from rlinf.workers.rollout.piper_rlt_worker import PiperRLTRollout

    worker = object.__new__(PiperRLTRollout)
    worker.policy = policy
    worker.snapshot_root = None
    worker.version = 64

    class RemoteWorker:
        def infer_with_reference(self, *args, **kwargs):
            result = worker.infer_with_reference(*args, **kwargs)
            return SimpleNamespace(wait=lambda: [result])

    remote = PiperOnlinePolicy(
        RemoteWorker(),
        {
            "manifest": {},
            "fps": 30,
            "action_horizon": 50,
            "actor_horizon": 10,
        },
    )
    remote.version = 64
    torch.manual_seed(29)
    plain = worker.infer(sample, "insert USB", actor_enabled=True)
    torch.manual_seed(29)
    calls = policy.feature_model.calls
    paired, diagnostics = remote.infer_with_reference(
        sample, "insert USB", actor_enabled=True
    )
    np.testing.assert_array_equal(paired, plain)
    assert policy.feature_model.calls == calls + 1
    assert diagnostics["policy_version"] == 64 and diagnostics["mode"] == "train"
    assert len(diagnostics["reference_actions"]) == 50
    policy.actor = None
    np.testing.assert_allclose(
        policy.infer(sample, "insert USB"),
        codec.decode(features["ref_chunk"][0].numpy()),
        atol=1e-5,
    )


@pytest.mark.parametrize("config_name", ["pi05_piper", "pi05_piper_h50_state"])
def test_piper_train_and_online_observation_transforms_agree(tmp_path, config_name):
    from openpi.transforms import compose

    from rlinf.models.embodiment.openpi.transforms.apply import apply_input_transform
    from rlinf.models.embodiment.openpi.transforms.env import repack_env_obs
    from rlinf.models.embodiment.openpi.transforms.pipeline import (
        build_openpi_transforms,
    )
    from rlinf.models.embodiment.piper_inference import piper_env_observation

    root = _piper_test_checkpoint(tmp_path, config_name)
    kwargs = {"norm_stats_path": str(root / "assets/piper/norm_stats.json")}
    inputs, _ = build_openpi_transforms(str(root), config_name, kwargs)
    sample = {
        "observation.state": np.array([0, 30, -30, 0, 0, 0, 35], np.float32),
        "prompt": "insert USB",
    }
    for i, name in enumerate(("front", "front_2", "wrist")):
        sample[f"observation.images.{name}"] = np.full((8, 9, 3), i * 50, np.uint8)
    expected = compose(inputs)(sample.copy())
    online = repack_env_obs(
        config_name,
        piper_env_observation(sample, "insert USB"),
        select_state=lambda x: x,
    )
    actual = apply_input_transform(compose(inputs), online)
    np.testing.assert_allclose(actual["state"][0], expected["state"])
    for key in expected["image"]:
        np.testing.assert_array_equal(actual["image"][key][0], expected["image"][key])
    np.testing.assert_array_equal(
        actual["tokenized_prompt"][0], expected["tokenized_prompt"]
    )


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("channels_last", [False, True])
def test_piper_preprocessing_preserves_full_camera_pixels(train, channels_last):
    from rlinf.models.embodiment.openpi.modules.model import (
        Observation,
        preprocess_observation,
    )

    # Distinct pixels include all borders: any crop or resize of a crop changes them.
    image = torch.linspace(-1, 1, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
    source = image.permute(0, 2, 3, 1) if channels_last else image
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    observation = Observation(
        images={key: source.clone() for key in keys},
        image_masks={key: torch.ones(2, dtype=torch.bool) for key in keys},
        state=torch.zeros(2, 32),
    )
    result = preprocess_observation(
        observation, train=train, spatial_augmentation=False
    )
    for key in keys:
        torch.testing.assert_close(
            result.images[key], image.permute(0, 2, 3, 1), rtol=0, atol=0
        )
        torch.testing.assert_close(observation.images[key], source, rtol=0, atol=0)


def test_piper_ten_step_sampler_preserves_fifty_step_action_horizon():
    from types import SimpleNamespace

    from rlinf.models.embodiment.openpi.pi0 import Pi0
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config

    observation = Pi0Config(pi05=True, action_horizon=50).fake_obs()
    times = []

    def suffix(obs, actions, time, cache, mask):
        times.append(float(time[0]))
        return torch.ones_like(actions)

    # Constant velocity has an exact endpoint; this checks the actual Euler loop.
    sampler = SimpleNamespace(
        build_prefix_cache=lambda obs: (None, None, None),
        run_suffix=suffix,
        velocity_from_suffix=lambda value: value,
    )
    actions = Pi0.sample_actions(
        sampler, observation, num_steps=10, noise=torch.zeros(1, 50, 32)
    )
    assert len(times) == 10
    assert times[0] == pytest.approx(1.0) and times[-1] == pytest.approx(0.1)
    torch.testing.assert_close(actions, -torch.ones(1, 50, 32))


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("height,width", [(224, 112), (112, 448)])
def test_piper_preprocessing_preserves_rectangular_camera_borders(train, height, width):
    from rlinf.models.embodiment.openpi.modules.model import (
        Observation,
        preprocess_observation,
    )

    image = torch.full((1, 3, height, width), -1.0)
    image[:, :, :4, :] = image[:, :, -4:, :] = 1.0
    image[:, :, :, :4] = image[:, :, :, -4:] = 1.0
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    observation = Observation(
        images={key: image.clone() for key in keys},
        image_masks={},
        state=torch.zeros(1, 32),
    )
    result = preprocess_observation(
        observation, train=train, spatial_augmentation=False
    )
    # Expected content rectangle: full height for portrait, 56 px high for landscape.
    top, bottom, left, right = (0, 224, 56, 168) if height == 224 else (84, 140, 0, 224)
    for key in keys:
        output = result.images[key]
        assert tuple(output.shape) == (1, 224, 224, 3)
        content = output[:, top:bottom, left:right, :]
        assert torch.all(content[:, 0, :, :] == 1)
        assert torch.all(content[:, -1, :, :] == 1)
        assert torch.all(content[:, :, 0, :] == 1)
        assert torch.all(content[:, :, -1, :] == 1)
        assert torch.all(content[:, content.shape[1] // 2, content.shape[2] // 2] == -1)


def test_piper_preprocessing_rng_keeps_camera_border_positions():
    from rlinf.models.embodiment.openpi.modules.model import (
        Observation,
        preprocess_observation,
    )

    # Black/white markers survive the existing positive color jitter unchanged.
    image = torch.full((1, 3, 224, 224), -1.0)
    image[:, :, :4, :] = image[:, :, -4:, :] = 1.0
    image[:, :, :, :4] = image[:, :, :, -4:] = 1.0
    keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    observation = Observation(
        images={key: image.clone() for key in keys},
        image_masks={},
        state=torch.zeros(1, 32),
    )
    result = preprocess_observation(
        observation,
        train=True,
        rng=torch.Generator().manual_seed(42),
        spatial_augmentation=False,
    )
    for key in keys:
        torch.testing.assert_close(
            result.images[key], image.permute(0, 2, 3, 1), rtol=0, atol=0
        )


def test_piper_h50_worker_saves_inference_assets(tmp_path, monkeypatch):
    import json

    from rlinf.models.embodiment.piper_inference import load_piper_inference_config
    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    root = _piper_test_checkpoint(tmp_path, "pi05_piper_h50_state")
    model_cfg, _ = load_piper_inference_config(root)
    (root / "piper_inference.json").unlink()
    data = tmp_path / "data"
    (data / "meta").mkdir(parents=True)
    (data / "meta/info.json").write_text(json.dumps({"fps": 30}))
    # Full distributed weight serialization is outside this CPU asset-path test.
    monkeypatch.setattr(FSDPSftWorker, "save_checkpoint", lambda *a, **kw: None)
    worker = object.__new__(FSDPVlaSftWorker)
    worker.cfg = OmegaConf.create(
        {"actor": {"model": model_cfg}, "data": {"train_data_paths": str(data)}}
    )
    worker._rank = 0
    worker.data_loader = None
    worker.save_checkpoint(str(root), step=10000)
    restored, _ = load_piper_inference_config(root)
    assert restored.openpi.config_name == "pi05_piper_h50_state"


@pytest.mark.parametrize("config_name", ["pi05_piper", "pi05_piper_h50_state"])
def test_piper_extract_rlt_features_keeps_normalized_units(config_name):
    from types import SimpleNamespace

    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi.tasks.eval import Pi0Eval

    observation = Pi0Config(pi05=True, action_horizon=50).fake_obs()
    observation.state[:, :7] = torch.linspace(-0.6, 0.6, 7)
    normalized_actions = torch.linspace(-0.8, 0.8, 50 * 32).reshape(1, 50, 32)

    def unexpected_decode(*args):
        raise AssertionError(
            "RLT features must remain normalized; deployment decodes once"
        )

    feature_model = SimpleNamespace(
        config_name=config_name,
        action_chunk=50,
        _require_rlt=lambda: None,
        env_obs_to_observation=lambda obs: observation,
        build_prefix_cache=lambda obs: (
            torch.zeros(1, 2, 8),
            torch.ones(1, 2, dtype=torch.bool),
            None,
        ),
        _select_rlt_prefix_embeddings=lambda prefix, mask, prompt: (prefix, mask),
        _encode_rlt_flat=lambda prefix, mask: torch.zeros(1, 8),
        _sample_actions_from_prefix_cache=lambda *args: normalized_actions,
        output_transform=unexpected_decode,
    )
    result = Pi0Eval.extract_rlt_obs(
        feature_model, {"states": torch.full((1, 7), 99.0)}
    )
    torch.testing.assert_close(result["proprio"], observation.state[:, :7])
    torch.testing.assert_close(result["ref_chunk"], normalized_actions[:, :, :7])
    assert result["ref_chunk"].shape == (1, 50, 7)


@pytest.mark.parametrize(
    "config_name,horizon,with_state",
    [("pi05_piper", 10, False), ("pi05_piper_h50_state", 50, True)],
)
def test_piper_registered_profile_and_rlt_actions_stay_normalized(
    monkeypatch, config_name, horizon, with_state
):
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.tasks import eval as eval_module

    config = get_openpi_config(config_name)
    assert config.model.action_horizon == horizon
    assert config.model.discrete_state_input is with_state
    observation = SimpleNamespace(
        state=torch.full((1, 32), 0.25), tokenized_prompt=None
    )
    actions = torch.full((1, horizon, 32), 0.5)
    model = SimpleNamespace(
        config_name=config_name,
        action_chunk=horizon,
        _require_rlt=lambda: None,
        env_obs_to_observation=lambda _: observation,
        build_prefix_cache=lambda _: (None, None, None),
        _select_rlt_prefix_embeddings=lambda *args: (None, None),
        _encode_rlt_flat=lambda *args: torch.zeros(1, 8),
        _sample_actions_from_prefix_cache=lambda *args: actions,
        output_transform=MagicMock(side_effect=AssertionError("Double decode")),
    )
    monkeypatch.setattr(
        eval_module, "preprocess_observation", lambda obs, **kwargs: obs
    )
    features = eval_module.Pi0Eval.extract_rlt_obs(model, {})
    assert features["ref_chunk"].shape == (1, horizon, 7)
    torch.testing.assert_close(features["ref_chunk"], actions[..., :7])
    torch.testing.assert_close(features["proprio"], observation.state[..., :7])
    model.output_transform.assert_not_called()


@pytest.mark.parametrize("version", [29999, 30000, 30001])
@pytest.mark.parametrize("critical", [False, True])
def test_piper_rlt_warmup_records_reference_before_actor_takeover(version, critical):
    from rlinf.algorithms.rlt.route import RLTRouteContext, build_rlt_route

    cfg = OmegaConf.create(
        {
            "env": {"train": {"env_type": "real"}},
            "algorithm": {
                "rlt_schedule": {
                    "enable": True,
                    "warmup_post_collect_updates": 30000,
                }
            },
        }
    )
    actor_actions = torch.full((1, 10, 7), 0.5)
    reference = torch.full((1, 50, 7), -0.25)
    ctx = RLTRouteContext(
        env_obs={},
        rlt_obs={"ref_chunk": reference},
        student_actions=actor_actions,
        result={"forward_inputs": {"ref_chunk": reference}},
        mode="train",
        rlt_switch_flags=torch.tensor([critical]),
        version=version,
    )
    output = build_rlt_route(cfg).route(ctx)
    active = critical and version >= 30000
    torch.testing.assert_close(
        output.actions, actor_actions if active else reference[:, :10]
    )
    forward = output.result["forward_inputs"]
    assert forward["record_transition"].item() == critical
    assert forward["actor_switch"].item() == active
    torch.testing.assert_close(forward["action"], output.actions.flatten(1))


def test_rlt_simulator_expert_target_preserves_frozen_reference_input():
    from types import SimpleNamespace

    from rlinf.algorithms.rlt.route import RLTRouteContext, SimulatorRLTRoute

    reference = torch.zeros(1, 3, 7)
    expert = torch.full((1, 3, 7), 2.0)
    ctx = RLTRouteContext(
        env_obs={},
        rlt_obs={"ref_chunk": reference},
        student_actions=torch.ones_like(reference),
        result={"forward_inputs": {"ref_chunk": reference}},
        mode="train",
        rlt_switch_flags=torch.tensor([True]),
        intervene_requested=torch.tensor([True]),
        expert_model=SimpleNamespace(predict_action_batch=lambda **kw: (expert, {})),
    )
    output = SimulatorRLTRoute(use_schedule=False, warmup_updates=0).route(ctx)
    torch.testing.assert_close(output.actions, expert)
    torch.testing.assert_close(output.result["forward_inputs"]["ref_chunk"], reference)
    assert output.result["intervene_flags"].all()


def test_openpi_non_piper_spatial_augmentation_remains_enabled():
    from rlinf.models.embodiment.openpi.modules.model import (
        Observation,
        preprocess_observation,
    )

    image = torch.linspace(-1, 1, 224 * 224 * 3).reshape(1, 224, 224, 3)
    observation = Observation(
        images={"base_0_rgb": image}, image_masks={}, state=torch.zeros(1, 32)
    )
    augmented = preprocess_observation(
        observation, train=True, image_keys=("base_0_rgb",)
    )
    assert not torch.equal(augmented.images["base_0_rgb"], image)
    preserved = preprocess_observation(
        observation,
        train=True,
        image_keys=("base_0_rgb",),
        spatial_augmentation=False,
    )
    torch.testing.assert_close(preserved.images["base_0_rgb"], image, rtol=0, atol=0)


def test_piper_warmup_loss_masks_padding_and_discounts_real_ticks():
    from rlinf.runners.rlt_warmup_runner import discounted_target, masked_behavior_loss

    prediction = torch.zeros(2, 21, requires_grad=True)
    batch = {
        "actions": torch.ones(2, 3, 7),
        "curr_obs": {"ref_chunk": torch.zeros(2, 3, 7)},
        "human": torch.tensor([[True, False, True], [False, False, False]]),
        "valid_steps": torch.tensor([2, 1]),
        "discount_steps": torch.tensor([2, 1]),
        "terminated": torch.tensor([True, False]),
        "rewards": torch.tensor([1.0, 0.0]),
    }
    loss = masked_behavior_loss(prediction, batch)
    assert float(loss.detach()) == pytest.approx(1 / 3)
    loss.backward()
    assert not prediction.grad[0, 14:].any()
    assert not prediction.grad[1, 7:].any()
    target = discounted_target(batch, torch.tensor([100.0, 2.0]), gamma=0.5)
    torch.testing.assert_close(target, torch.tensor([0.5, 1.0]))


def test_piper_warmup_driver_projection_matches_execution_and_has_clamp_gradients(
    piper_local_warmup_cache, tmp_path
):
    from rlinf.robotics.parts.arms.piper import PiperArm
    from rlinf.runners.rlt_warmup_runner import RLTWarmupRunner, WarmupConfig

    runner = RLTWarmupRunner(
        WarmupConfig(
            cache=str(piper_local_warmup_cache),
            output=str(tmp_path / "run"),
            device="cpu",
            critic_action_projection="driver_limits",
        )
    )
    raw = np.array(
        [
            [-200, -20, -200, -120, -80, -140, -2],
            [200, 200, 20, 120, 80, 140, 80],
            [0, 60, -60, 0, 0, 0, 35],
        ],
        dtype=np.float32,
    )
    normalized = torch.tensor(runner.codec.encode(raw), requires_grad=True)
    projected = runner.critic_actions(normalized)
    low = np.r_[np.rad2deg(PiperArm.JOINT_LIMITS_LOWER), 0]
    high = np.r_[np.rad2deg(PiperArm.JOINT_LIMITS_UPPER), 70]
    expected = runner.codec.encode(np.clip(raw, low, high))
    np.testing.assert_allclose(projected.detach().numpy(), expected, atol=2e-6)
    projected.sum().backward()
    assert not normalized.grad[:2].any()
    assert (normalized.grad[2] == 1).all()
    batch = runner.replay.batch([0], [0], "cpu")
    observed = []
    original = runner.target.sac_q_forward

    def capture(obs, actions):
        observed.append(actions.detach().clone())
        return original(obs, actions)

    runner.target.sac_q_forward = capture
    runner._next_q(batch, deterministic=True)
    assert not observed[0].any()  # projection happens before terminal padding


def test_piper_warmup_local_validation_uses_segment_outcomes_and_offline_export(
    piper_local_warmup_cache, tmp_path
):
    import json

    from rlinf.runners.rlt_warmup_runner import RLTWarmupRunner, WarmupConfig

    runner = RLTWarmupRunner(
        WarmupConfig(
            cache=str(piper_local_warmup_cache),
            output=str(tmp_path / "run"),
            device="cpu",
        )
    )
    report = runner.validate()
    assert [
        (r["episode"], r["phase_id"], r["outcome"]) for r in report["episodes"]
    ] == [(2, 0, "success"), (2, 1, "failure"), (3, 0, "success")]
    assert report["by_phase"]["0"]["terminal_rmse_balanced"] is None
    assert report["by_phase"]["0"]["validation_counts"] == {"success": 2, "failure": 0}
    assert not report["checks"]["local_phase_coverage"]
    runner.history.append(report)
    runner.output.mkdir()
    exported = runner.export()
    metadata = json.loads((runner.output / "actor/piper_rlt_actor.json").read_text())
    assert metadata["format"] == "rlinf-piper-rlt-offline-local-actor-v1"
    assert not exported["ready_for_deterministic_eval"]
    filtered = RLTWarmupRunner(
        WarmupConfig(
            cache=str(piper_local_warmup_cache),
            output=str(tmp_path / "filtered"),
            device="cpu",
            phase_filter=1,
        )
    )
    assert filtered.binding != runner.binding
    assert set(filtered.validate()["by_phase"]) == {"1"}
    runner.replay.selected_rows[0] = runner.replay.selected_rows[0][:1]
    missing_train_phase = runner.validate()
    assert missing_train_phase["by_phase"]["1"]["train_counts"] == {
        "success": 0,
        "failure": 0,
    }
    assert (
        missing_train_phase["by_phase"]["1"]["constant_baseline_rmse_balanced"] is None
    )
    assert not missing_train_phase["checks"]["local_phase_coverage"]


def test_piper_warmup_zero_joint_budget_stops_at_critic(tmp_path):
    from rlinf.runners.rlt_warmup_runner import load_warmup_config

    path = tmp_path / "pre_joint.yaml"
    path.write_text("training:\n  joint_steps: 0\n")
    cfg = load_warmup_config(path)
    assert cfg.training.joint_steps == 0
    with pytest.raises(ValueError, match="nonnegative"):
        load_warmup_config(path, ["training.joint_steps=-1"])


def test_piper_warmup_target_includes_reference_gaps_and_scores_only_the_task_end():
    from rlinf.runners.rlt_warmup_runner import discounted_target

    batch = {
        "valid_steps": torch.tensor([5, 3]),
        "discount_steps": torch.tensor([12, 6]),
        "terminated": torch.tensor([False, True]),
        "rewards": torch.tensor([0.0, 1.0]),
    }
    target = discounted_target(batch, torch.tensor([2.0, 100.0]), gamma=0.5)
    torch.testing.assert_close(target, torch.tensor([0.5**12 * 2, 0.5**5]))


def test_piper_warmup_human_conditioning_preserves_reference_and_padding():
    from rlinf.runners.rlt_warmup_runner import actor_training_observation

    reference = torch.full((3, 5, 7), 0.1)
    batch = {
        "actions": torch.full((3, 3, 7), 0.6),
        "curr_obs": {"ref_chunk": reference, "z_rl": torch.ones(3, 2048)},
        "next_obs": {"ref_chunk": reference},
        "human": torch.tensor(
            [[True, False, True], [False, False, False], [True, True, True]]
        ),
        "valid_steps": torch.tensor([2, 3, 3]),
    }
    observation = actor_training_observation(batch)
    # Mixed HIL, a reference-only sample and a complete human demonstration.
    expected = torch.full((3, 5, 7), 0.1)
    expected[0, 0] = 0.6
    expected[2, :3] = 0.6
    torch.testing.assert_close(observation["ref_chunk"], expected)
    torch.testing.assert_close(reference, torch.full_like(reference, 0.1))
    torch.testing.assert_close(batch["next_obs"]["ref_chunk"], reference)
    assert observation["z_rl"] is batch["curr_obs"]["z_rl"]


@pytest.mark.parametrize("dropout_end", [0.0, 1.0])
def test_piper_warmup_updates_use_human_conditioning_and_recorded_critic_actions(
    piper_warmup_cache, tmp_path, dropout_end
):
    import json

    from rlinf.data.datasets.rlt_warmup import file_sha256, json_fingerprint
    from rlinf.runners.rlt_warmup_runner import (
        RLTWarmupRunner,
        WarmupConfig,
        WarmupTraining,
    )

    manifest_path = piper_warmup_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["episodes"]:
        if entry["split"] != "train":
            continue
        path = piper_warmup_cache / entry["file"]
        with np.load(path) as shard:
            arrays = {key: shard[key] for key in shard.files}
        arrays["human"][:, 0] = True
        arrays["actions"][:, 0] = 0.6
        np.savez(path, **arrays)
        entry["sha256"] = file_sha256(path)
    manifest.pop("content_fingerprint")
    manifest["content_fingerprint"] = json_fingerprint(manifest)
    manifest_path.write_text(json.dumps(manifest))
    runner = RLTWarmupRunner(
        WarmupConfig(
            cache=str(piper_warmup_cache),
            output=str(tmp_path / "run"),
            hardware=str(
                Path(__file__).resolve().parents[2]
                / "examples/embodiment/config/piper_usb_hardware.yaml"
            ),
            device="cpu",
            threads=1,
            training=WarmupTraining(
                bc_steps=1,
                critic_steps=1,
                joint_steps=2,
                actor_every=1,
                batch_size=4,
                dropout_end=dropout_end,
            ),
        )
    )
    actor_inputs, critic_actions = [], []
    actor_hook = runner.model.backbone.register_forward_pre_hook(
        lambda module, args: actor_inputs.append(args[0].detach().clone())
    )
    critic_hook = runner.model.q_head.register_forward_pre_hook(
        lambda module, args: critic_actions.append(args[1].detach().clone())
    )
    try:
        for step in range(1, 5):
            state = runner.replay.rng_state()
            batch = runner.replay.sample(4)
            runner.replay.restore_rng(state)
            actor_inputs.clear()
            critic_actions.clear()
            metrics = runner.update()
            if step > 1:
                # Bellman bootstrap uses the VLA proposal at the next state.
                torch.testing.assert_close(
                    actor_inputs[0][:, :70],
                    batch["next_obs"]["ref_chunk"][:, :10].flatten(1),
                )
                # The TD loss evaluates actions that were actually recorded.
                torch.testing.assert_close(
                    critic_actions[0], batch["actions"].flatten(1)
                )
            if step != 2:
                expected = batch["curr_obs"]["ref_chunk"][:, :10].clone()
                expected[:, 0] = 0.6
                if step == 4 and dropout_end == 1.0:
                    expected.zero_()
                torch.testing.assert_close(
                    actor_inputs[-1][:, :70], expected.flatten(1)
                )
                assert np.isfinite(metrics["actor_grad_norm"])
            for episode in runner.replay.episodes:
                torch.testing.assert_close(
                    episode["ref_chunk"],
                    torch.full_like(episode["ref_chunk"], 0.1),
                )
    finally:
        actor_hook.remove()
        critic_hook.remove()
    for entry in manifest["episodes"]:
        assert file_sha256(piper_warmup_cache / entry["file"]) == entry["sha256"]


def test_piper_warmup_gate_requires_validation_of_the_exported_update():
    from rlinf.runners.rlt_warmup_runner import offline_warmup_passed

    history = [{"step": step, "passed": True, "scheduled": True} for step in (2, 4, 6)]
    assert offline_warmup_passed(
        history, step=6, budget=6, interval=2, required_passes=3
    )
    assert not offline_warmup_passed(
        history, step=6, budget=7, interval=2, required_passes=3
    )
    # The final optimizer update must not inherit readiness from older weights.
    assert not offline_warmup_passed(
        history, step=7, budget=7, interval=2, required_passes=3
    )
    history.append({"step": 7, "passed": False, "scheduled": False})
    assert not offline_warmup_passed(
        history, step=7, budget=7, interval=2, required_passes=3
    )
    history[-1]["passed"] = True
    assert offline_warmup_passed(
        history, step=7, budget=7, interval=2, required_passes=3
    )


def test_piper_warmup_freezes_the_correct_heads_and_resumes_exactly(
    piper_warmup_cache, tmp_path
):
    import copy
    import json

    from rlinf.models.embodiment.mlp_policy import get_model as get_actor
    from rlinf.runners.rlt_warmup_runner import (
        RLTWarmupRunner,
        WarmupConfig,
        WarmupTraining,
    )

    cfg = WarmupConfig(
        cache=str(piper_warmup_cache),
        output=str(tmp_path / "full"),
        hardware=str(
            Path(__file__).resolve().parents[2]
            / "examples/embodiment/config/piper_usb_hardware.yaml"
        ),
        device="cpu",
        threads=1,
        training=WarmupTraining(
            bc_steps=2,
            critic_steps=2,
            joint_steps=4,
            batch_size=4,
            eval_every=2,
            save_every=2,
        ),
    )
    runner = RLTWarmupRunner(cfg)
    q_before = copy.deepcopy(runner.model.q_head.state_dict())
    actor_before = copy.deepcopy(runner.model.backbone.state_dict())
    runner.update()
    runner.update()
    for key, value in runner.model.q_head.state_dict().items():
        torch.testing.assert_close(value, q_before[key], rtol=0, atol=0)
    assert any(
        not torch.equal(value, actor_before[key])
        for key, value in runner.model.backbone.state_dict().items()
    )
    actor_before = copy.deepcopy(runner.model.backbone.state_dict())
    runner.update()
    runner.update()
    for key, value in runner.model.backbone.state_dict().items():
        torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)
    assert any(
        not torch.equal(value, q_before[key])
        for key, value in runner.model.q_head.state_dict().items()
    )
    # Full public run and interrupted/resumed run must produce identical weights and counters.
    full = RLTWarmupRunner(cfg)
    report = full.run()
    resumed_cfg = copy.deepcopy(cfg)
    resumed_cfg.output = str(tmp_path / "resumed")
    RLTWarmupRunner(resumed_cfg).run(until_step=3)
    # A killed log write must not invalidate a complete optimizer checkpoint.
    with (Path(resumed_cfg.output) / "metrics.jsonl").open("a") as log:
        log.write('{"step": 4, "validation":')
    resumed = RLTWarmupRunner(resumed_cfg)
    restored_report = resumed.run(resume=True)
    assert report["actor_steps"] == restored_report["actor_steps"] == 3
    assert report["critic_steps"] == restored_report["critic_steps"] == 6
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(
            value, resumed.model.state_dict()[key], rtol=0, atol=0
        )
    assert full.replay.rng_state() == resumed.replay.rng_state()
    assert (
        not report["ready_for_deterministic_eval"]
        and not report["ready_for_online_exploration"]
    )
    assert not report["validation"]["checks"]["real_data"]
    assert "human_rmse" in report["validation"]
    assert report["validation"]["human_validation_ticks"] >= 0
    metadata = json.loads((Path(report["actor"]) / "piper_rlt_actor.json").read_text())
    exported = get_actor(OmegaConf.create(metadata["model"]), torch_dtype=torch.float32)
    exported.load_state_dict(
        torch.load(
            Path(report["actor"]) / "model_state_dict/full_weights.pt",
            weights_only=True,
        )
    )
    batch = full.replay.sample(4)
    torch.testing.assert_close(
        exported.sac_forward(batch["curr_obs"], deterministic=True)[0],
        full.model.sac_forward(batch["curr_obs"], deterministic=True)[0],
        rtol=0,
        atol=0,
    )
    resumed_cfg.training.gamma = 0.9
    with pytest.raises(ValueError, match="config, implementation or cache changed"):
        RLTWarmupRunner(resumed_cfg).run(resume=True)


@pytest.mark.parametrize("corruption", ["{broken}\n", '{broken}\n{"step": 9}\n'])
def test_piper_warmup_resume_rejects_complete_corrupt_log_records(
    piper_warmup_cache, tmp_path, corruption
):
    from rlinf.runners.rlt_warmup_runner import RLTWarmupRunner, WarmupConfig

    cfg = WarmupConfig(
        cache=str(piper_warmup_cache),
        output=str(tmp_path / "run"),
        hardware=str(
            Path(__file__).resolve().parents[2]
            / "examples/embodiment/config/piper_usb_hardware.yaml"
        ),
        device="cpu",
        threads=1,
    )
    RLTWarmupRunner(cfg).run(until_step=1)
    with (Path(cfg.output) / "metrics.jsonl").open("a") as log:
        log.write(corruption)
    import json

    with pytest.raises(json.JSONDecodeError):
        RLTWarmupRunner(cfg).run(resume=True, until_step=1)


def test_piper_warmup_resume_binds_actor_implementation(
    piper_warmup_cache, tmp_path, monkeypatch
):
    import rlinf.runners.rlt_warmup_runner as module

    cfg = module.WarmupConfig(
        cache=str(piper_warmup_cache),
        output=str(tmp_path / "run"),
        hardware=str(
            Path(__file__).resolve().parents[2]
            / "examples/embodiment/config/piper_usb_hardware.yaml"
        ),
        device="cpu",
        threads=1,
    )
    module.RLTWarmupRunner(cfg).run(until_step=1)
    original = module.implementation_fingerprints

    def changed(paths):
        hashes = original(paths)
        hashes["rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py"] = "changed"
        return hashes

    monkeypatch.setattr(module, "implementation_fingerprints", changed)
    with pytest.raises(ValueError, match="config, implementation or cache changed"):
        module.RLTWarmupRunner(cfg).run(resume=True, until_step=1)


def test_piper_warmup_validation_measures_human_targets_without_padding(
    piper_warmup_cache, tmp_path
):
    import json

    from rlinf.data.datasets.rlt_warmup import file_sha256, json_fingerprint
    from rlinf.runners.rlt_warmup_runner import RLTWarmupRunner, WarmupConfig

    manifest_path = piper_warmup_cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["episodes"][2]
    shard_path = piper_warmup_cache / entry["file"]
    with np.load(shard_path) as shard:
        arrays = {key: shard[key] for key in shard.files}
    mask = np.arange(10)[None, :] < arrays["valid_steps"][:, None]
    arrays["human"] = mask
    arrays["actions"][mask] = 0.5
    np.savez(shard_path, **arrays)
    entry["sha256"] = file_sha256(shard_path)
    manifest.pop("content_fingerprint")
    manifest["content_fingerprint"] = json_fingerprint(manifest)
    manifest_path.write_text(json.dumps(manifest))
    cfg = WarmupConfig(
        cache=str(piper_warmup_cache),
        output=str(tmp_path / "run"),
        hardware=str(
            Path(__file__).resolve().parents[2]
            / "examples/embodiment/config/piper_usb_hardware.yaml"
        ),
        device="cpu",
        threads=1,
    )
    runner = RLTWarmupRunner(cfg)
    # A constant 0.1 actor exactly matches the reference, but misses human 0.5.
    with torch.no_grad():
        for parameter in runner.model.parameters():
            parameter.zero_()
        runner.model.actor_mean.bias.fill_(float(np.arctanh(0.1)))
    report = runner.validate()
    assert report["actor_rmse"] == pytest.approx(0, abs=1e-7)
    assert report["human_rmse"] == pytest.approx(0.4)
    assert report["human_validation_ticks"] == int(mask.sum())
