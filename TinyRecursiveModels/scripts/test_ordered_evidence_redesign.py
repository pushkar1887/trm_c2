"""Focused regression tests for the ordered-evidence/compositional V2 redesign.

This repository cannot assume pytest is installed, so the tests use the same plain-script
contract as ``scripts/test_relmap_integration.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch


def _base_config(**overrides):
    from models.recursive_reasoning.trm_fvr_v2 import FVR_C2_Config

    values = dict(
        batch_size=2,
        seq_len=900,
        puzzle_emb_ndim=128,
        num_puzzle_identifiers=4,
        vocab_size=12,
        H_cycles=1,
        L_cycles=1,
        H_layers=1,
        L_layers=1,
        hidden_size=128,
        expansion=2,
        num_heads=4,
        pos_encodings="rope",
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
    )
    values.update(overrides)
    return FVR_C2_Config(**values)


def test_sample_batch_uses_supplied_rng_only() -> None:
    import numpy as np

    from puzzle_dataset import _sample_batch

    group_order = np.array([0], dtype=np.int64)
    puzzle_indices = np.array([0, 10], dtype=np.int64)
    group_indices = np.array([0, 1], dtype=np.int64)

    np.random.seed(1)
    first = _sample_batch(
        np.random.Generator(np.random.Philox(seed=123)),
        group_order, puzzle_indices, group_indices, 0, 4,
    )[1]
    np.random.seed(999)
    second = _sample_batch(
        np.random.Generator(np.random.Philox(seed=123)),
        group_order, puzzle_indices, group_indices, 0, 4,
    )[1]

    assert np.array_equal(first, second), (
        "_sample_batch must use only its supplied Generator; global NumPy state changed the batch")


def test_explicit_lodo_contract_runs_in_eval_and_overrides_rng() -> None:
    from models.recursive_reasoning.trm_fvr_v2 import TinyRecursiveReasoningModel_ACTV1_Inner

    cfg = _base_config(
        c2_enabled=True,
        c2_leave_one_demo_weight=0.0,
        c2_lodo_force_build=True,
        c2_lodo_max_samples=4,
    )
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(cfg).eval()
    context_inputs = torch.stack([
        torch.full((900,), 2, dtype=torch.long),
        torch.full((900,), 3, dtype=torch.long),
        torch.full((900,), 4, dtype=torch.long),
    ]).unsqueeze(0)
    batch = {
        "inputs": torch.full((1, 900), 2, dtype=torch.long),
        "labels": torch.full((1, 900), 2, dtype=torch.long),
        "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
        "context_inputs": context_inputs,
        "context_outputs": context_inputs.clone(),
        "context_mask": torch.ones((1, 3), dtype=torch.bool),
        "_force_lodo_eval": True,
        "_lodo_holdout_idx": torch.tensor([1], dtype=torch.long),
        "_lodo_aux_valid": torch.tensor([True]),
    }

    torch.manual_seed(1)
    first = inner._build_lodo_batch(batch)
    torch.manual_seed(999)
    second = inner._build_lodo_batch(batch)

    assert first is not None and second is not None, "explicit eval LODO contract was ignored"
    assert torch.equal(first["inputs"], context_inputs[:, 1])
    assert torch.equal(first["inputs"], second["inputs"]), "holdout changed with global torch RNG"
    assert not bool(first["context_mask"][0, 1]), "held-out demo remained in active support"


def test_runner_freezes_lodo_contract_with_local_rng() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    batch = {
        "context_mask": torch.tensor([
            [True, True, True],
            [True, False, True],
            [True, True, False],
        ]),
    }
    torch.manual_seed(1)
    first = runner._freeze_lodo_contract(batch, seed=77, max_samples=2)
    torch.manual_seed(999)
    second = runner._freeze_lodo_contract(batch, seed=77, max_samples=2)

    assert torch.equal(first["_lodo_holdout_idx"], second["_lodo_holdout_idx"])
    assert torch.equal(first["_lodo_aux_valid"], second["_lodo_aux_valid"])
    assert int(first["_lodo_aux_valid"].sum()) == 2
    selected = first["context_mask"].gather(
        1, first["_lodo_holdout_idx"].view(-1, 1)).squeeze(1)
    assert bool(selected[first["_lodo_aux_valid"]].all())
    assert torch.is_tensor(first["_force_lodo_eval"])
    assert first["_force_lodo_eval"].dtype == torch.bool
    assert first["_force_lodo_eval"].shape == first["context_mask"].shape[:1]
    assert bool(first["_force_lodo_eval"].all())
    assert all(torch.is_tensor(value) for value in first.values()), (
        "frozen batch metadata must remain compatible with ACT carry allocation")


def test_lodo_contract_rejects_non_integer_holdout_indices() -> None:
    from models.recursive_reasoning.trm_fvr_v2 import TinyRecursiveReasoningModel_ACTV1_Inner

    cfg = _base_config(c2_enabled=True, c2_lodo_force_build=True)
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(cfg).eval()
    context = torch.full((1, 2, 900), 2, dtype=torch.long)
    batch = {
        "inputs": context[:, 0],
        "labels": context[:, 0],
        "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
        "context_inputs": context,
        "context_outputs": context.clone(),
        "context_mask": torch.ones((1, 2), dtype=torch.bool),
        "_force_lodo_eval": True,
        "_lodo_holdout_idx": torch.tensor([0.5]),
        "_lodo_aux_valid": torch.tensor([True]),
    }
    try:
        inner._build_lodo_batch(batch)
    except ValueError as exc:
        assert "Long" in str(exc)
    else:
        raise AssertionError("floating-point LODO holdout index was silently truncated")


def test_evaluation_mode_restores_training_state() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    module = torch.nn.Linear(2, 2).train()
    with runner._evaluation_mode(module):
        assert module.training is False
    assert module.training is True

    module.eval()
    with runner._evaluation_mode(module):
        assert module.training is False
    assert module.training is False


def test_seeded_evaluation_restores_rng_and_repeats_exactly() -> None:
    import importlib.util
    import random

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    module = torch.nn.Dropout(p=0.5).train()
    torch.manual_seed(11)
    np.random.seed(12)
    random.seed(13)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    def sample():
        with runner._seeded_evaluation(module, seed=77):
            assert module.training is False
            return torch.rand(4), np.random.rand(4), tuple(random.random() for _ in range(4))

    first = sample()
    second = sample()
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert module.training is True
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert random.getstate() == python_before


def test_runner_builds_independent_rank0_eval_loader() -> None:
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    sentinel = object()
    calls = []
    original = runner.pretrain.create_dataloader

    def fake(config, split, rank, world, **kwargs):
        calls.append((config, split, rank, world, kwargs))
        return sentinel, object()

    runner.pretrain.create_dataloader = fake
    try:
        config = SimpleNamespace(global_batch_size=8)
        loader = runner.build_eval_loader(config)
    finally:
        runner.pretrain.create_dataloader = original

    assert loader is sentinel
    assert len(calls) == 1
    _, split, rank, world, kwargs = calls[0]
    assert (split, rank, world) == ("train", 0, 1)
    assert kwargs["global_batch_size"] == 8


def test_collect_eval_batches_restores_all_global_rng_state() -> None:
    import importlib.util
    import random
    from types import SimpleNamespace

    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    batch = {
        "inputs": torch.full((2, 900), 2, dtype=torch.long),
        "context_inputs": torch.full((2, 2, 900), 2, dtype=torch.long),
        "context_outputs": torch.full((2, 2, 900), 2, dtype=torch.long),
        "context_mask": torch.ones((2, 2), dtype=torch.bool),
    }
    loader = iter([("train", batch, None)])
    args = SimpleNamespace(eval_seed=1234, batch=2, eval_batches=1, log_every=10)
    torch.manual_seed(11); np.random.seed(12); random.seed(13)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    python_before = random.getstate()
    frozen = runner.collect_eval_batches(args, loader, torch.device("cpu"))
    assert len(frozen) == 1
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert random.getstate() == python_before


def test_injection_scale_zero_removes_target_relmaps_in_both_flows() -> None:
    from models.recursive_reasoning.relation_map import REL_MAP_CHANNELS
    from models.recursive_reasoning.trm_fvr_v2 import TinyRecursiveReasoningModel_ACTV1_Inner

    inputs = torch.full((1, 900), 2, dtype=torch.long)
    rel_maps = torch.ones((1, 900, REL_MAP_CHANNELS), dtype=torch.float32)
    for ordered in (False, True):
        cfg = _base_config(
            c2_enabled=False,
            c2_relmap=True,
            c2_ordered_evidence_flow=ordered,
            c2_bounded_evidence_fusion=False,
        )
        inner = TinyRecursiveReasoningModel_ACTV1_Inner(cfg).eval()
        with torch.no_grad():
            inner.relmap_proj.weight.fill_(0.125)
        baseline = inner.grid_encoder(inputs, None)
        inner._demo_injection_scale = 0.0
        conditioned = inner._condition_grid_features(inputs, rel_maps=rel_maps)[0]
        assert torch.equal(conditioned, baseline), (
            f"injection-off leaked target relmap residual in ordered={ordered} flow")


def test_injection_scale_zero_removes_pid_task_modulation() -> None:
    from models.recursive_reasoning.trm_fvr_v2 import TinyRecursiveReasoningModel_ACTV1_Inner

    cfg = _base_config(c2_enabled=True, c2_modulate_pid=True)
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(cfg).eval()
    with torch.no_grad():
        inner.pid_task_modulator.weight.fill_(0.25)
        inner.pid_task_gate.fill_(1.0)
    grid_features = torch.zeros((1, 900, cfg.hidden_size), dtype=torch.float32)
    puzzle_ids = torch.zeros(1, dtype=torch.long)
    task_vec = torch.ones((1, cfg.hidden_size), dtype=torch.float32)
    baseline = inner._prepend_puzzle_embeddings(
        grid_features, puzzle_ids, use_sparse_training_buffer=False, pid_task_vec=None)
    inner._demo_injection_scale = 0.0
    recovered = inner._prepend_puzzle_embeddings(
        grid_features, puzzle_ids, use_sparse_training_buffer=False, pid_task_vec=task_vec)
    assert torch.equal(recovered, baseline), "injection-off leaked C2 PID modulation"


def test_injection_scale_zero_removes_visual_adapter_residual() -> None:
    from models.recursive_reasoning.trm_fvr_v2 import TinyRecursiveReasoningModel_ACTV1_Inner

    class FakeVisualAdapter(torch.nn.Module):
        def forward(self, base_features, **_kwargs):
            return base_features + 3.0, {"fake_visual_metric": torch.tensor(1.0)}

    cfg = _base_config(c2_enabled=False, c2_relmap=False)
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(cfg).eval()
    inner.visual_rule_adapter = FakeVisualAdapter()
    inputs = torch.full((1, 900), 2, dtype=torch.long)
    context = inputs[:, None, :].expand(-1, 2, -1).clone()
    baseline = inner.grid_encoder(inputs, None)
    inner._demo_injection_scale = 0.0
    conditioned = inner._condition_grid_features(
        inputs,
        context_inputs=context,
        context_outputs=context,
        context_mask=torch.ones((1, 2), dtype=torch.bool),
    )[0]
    assert torch.equal(conditioned, baseline), "injection-off leaked visual-rule residual"


def test_source_and_lodo_manifests_are_content_addressed() -> None:
    import importlib.util
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.py").write_text("alpha\n", encoding="utf-8")
        (root / "b.yaml").write_text("beta: 1\n", encoding="utf-8")
        first = runner._build_source_manifest(root, ("a.py", "b.yaml"))
        second = runner._build_source_manifest(root, ("a.py", "b.yaml"))
        assert first["fingerprint"] == second["fingerprint"]
        (root / "a.py").write_text("changed\n", encoding="utf-8")
        third = runner._build_source_manifest(root, ("a.py", "b.yaml"))
        assert first["fingerprint"] != third["fingerprint"]

    batch = {
        "puzzle_identifiers": torch.tensor([7, 9]),
        "_lodo_holdout_idx": torch.tensor([1, 0]),
        "_lodo_aux_valid": torch.tensor([True, False]),
    }
    folds_a = runner._build_lodo_contract_manifest([batch], eval_seed=123)
    folds_b = runner._build_lodo_contract_manifest([batch], eval_seed=123)
    assert folds_a == folds_b
    assert folds_a["batches"][0]["holdout_idx"] == [1, 0]
    assert folds_a["batches"][0]["aux_valid"] == [True, False]


def test_failure_artifact_records_rank_stage_batch_and_rng() -> None:
    import importlib.util
    import tempfile
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    with tempfile.TemporaryDirectory() as tmp:
        state = SimpleNamespace(
            args=SimpleNamespace(save_checkpoint_dir=tmp, task_metrics_out=""),
            raw={"test": True},
            dist=SimpleNamespace(rank=2),
        )
        batch = {
            "inputs": torch.tensor([[2, 3]]),
            "puzzle_identifiers": torch.tensor([17]),
        }
        path = runner._write_failure_artifact(
            state, batch, step=5, stage="forward_loss", detail="non-finite test")
        assert path is not None and path.exists()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["step"] == 5
        assert payload["rank"] == 2
        assert payload["stage"] == "forward_loss"
        assert torch.equal(payload["batch"]["puzzle_identifiers"], torch.tensor([17]))
        assert "torch_rng_state" in payload and "numpy_rng_state" in payload


def test_nonfinite_gradient_detection_happens_before_allreduce() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    finite = torch.nn.Parameter(torch.tensor([1.0]))
    finite.grad = torch.tensor([2.0])
    poisoned = torch.nn.Parameter(torch.tensor([1.0]))
    poisoned.grad = torch.tensor([float("nan")])
    unused = torch.nn.Parameter(torch.tensor([1.0]))
    assert runner._has_nonfinite_grad([finite, poisoned, unused]) is True
    poisoned.grad = torch.tensor([3.0])
    assert runner._has_nonfinite_grad([finite, poisoned, unused]) is False
    sparse = torch.nn.Parameter(torch.zeros(3))
    sparse.grad = torch.sparse_coo_tensor(
        torch.tensor([[1]]), torch.tensor([float("nan")]), size=(3,))
    assert runner._has_nonfinite_grad([sparse]) is True


def test_where_metrics_separate_macro_micro_and_support_contracts() -> None:
    import models.losses_fvr as losses

    assert hasattr(losses, "_where_metrics_per_task"), "missing per-task WHERE metric contract"
    target = torch.tensor([[3, 2, 2, 2], [3, 4, 4, 4]])
    inputs = torch.tensor([[2, 2, 2, 2], [2, 4, 4, 4]])
    # Row 0: perfect. Row 1: misses its changed cell and fires on all three copy cells.
    q = torch.tensor([[0.9, 0.1, 0.1, 0.1], [0.1, 0.9, 0.9, 0.9]])
    out = losses._where_metrics_per_task(q, target, inputs)
    assert out["f1_per_task"].shape == (2,)
    assert abs(float(out["macro_f1"]) - 0.5) < 1e-6
    assert float(out["micro_f1"]) != float(out["macro_f1"])


def test_where_support_contrast_is_finite_with_no_matched_rows() -> None:
    import models.losses_fvr as losses

    q = torch.tensor([[0.5, 0.5]])
    target = torch.tensor([[3, 2]])
    inputs = torch.tensor([[2, 2]])
    correct = losses._where_metrics_per_task(q, target, inputs)
    shuffled = losses._where_metrics_per_task(q, target, inputs)
    value = losses._where_support_contrast_per_task(
        correct, shuffled, torch.zeros(1, dtype=torch.bool), margin=0.05)
    assert torch.isfinite(value)
    assert float(value) == 0.0


def test_positive_where_loss_trains_copy_only_rows_closed() -> None:
    import models.losses_fvr as losses

    inputs = torch.tensor([[2, 3, 4, 5]])
    target = inputs.clone()
    open_gate = losses._where_metrics_per_task(torch.full((1, 4), 0.9), target, inputs)
    closed_gate = losses._where_metrics_per_task(torch.full((1, 4), 0.1), target, inputs)
    assert float(open_gate["proper_loss"]) > float(closed_gate["proper_loss"])
    assert float(open_gate["proper_loss"]) > 0.0


def test_evidence_schema_fingerprint_is_order_and_semantics_sensitive() -> None:
    import models.recursive_reasoning.trm_fvr_v2 as model

    assert hasattr(model, "evidence_schema_fingerprint"), "missing evidence schema fingerprint"
    a = _base_config(c2_relmap=True, c2_transition_hint=True)
    b = _base_config(c2_relmap=True, c2_transition_hint=True)
    assert torch.equal(model.evidence_schema_fingerprint(a), model.evidence_schema_fingerprint(b))
    b.c2_transition_hint = False
    b.c2_task_palette_feature = True  # both are width 10; semantics must still differ
    assert not torch.equal(model.evidence_schema_fingerprint(a), model.evidence_schema_fingerprint(b))

    # Input-side semantic factors must also invalidate a checkpoint even though they add no
    # output evidence columns and therefore leave the color projection width unchanged.
    c = _base_config(c2_relmap=True, c2_rule_factor_hint=False)
    d = _base_config(c2_relmap=True, c2_rule_factor_hint=True)
    assert model.evidence_total(c) == model.evidence_total(d)
    assert not torch.equal(model.evidence_schema_fingerprint(c), model.evidence_schema_fingerprint(d))


def test_legacy_schema_migration_resets_semantic_consumers() -> None:
    import pretrain

    schema = "model.inner.evidence_schema_fingerprint"
    state = {
        "model.inner.color_evidence_proj.weight": torch.ones(10, 10),
        "model.inner.color_head_mlp_in.weight": torch.ones(4, 14),
        "model.inner.rule_factor_proj.weight": torch.ones(8, 13),
        "model.inner.pairdelta_input_encoder.spatial_mlp.2.weight": torch.ones(8, 8),
        "model.inner.lm_head.weight": torch.ones(12, 8),
    }
    model_state = {schema: torch.arange(32, dtype=torch.uint8)}
    dropped = pretrain._prepare_evidence_schema_state(
        state, model_state, schema, allow_legacy=True)
    assert "model.inner.lm_head.weight" in state
    assert set(dropped) == {
        "model.inner.color_evidence_proj.weight",
        "model.inner.color_head_mlp_in.weight",
        "model.inner.rule_factor_proj.weight",
        "model.inner.pairdelta_input_encoder.spatial_mlp.2.weight",
    }
    assert all(key not in state for key in dropped)

    try:
        pretrain._prepare_evidence_schema_state(
            {"model.inner.lm_head.weight": torch.ones(1)}, model_state, schema,
            allow_legacy=False)
    except RuntimeError as exc:
        assert "predates evidence-schema fingerprints" in str(exc)
    else:
        raise AssertionError("missing fingerprint must require explicit legacy permission")


def test_requested_auxiliary_losses_fail_when_evidence_is_missing() -> None:
    import models.losses_fvr as losses

    outputs = {
        "c2_aux_gate_where_values": torch.zeros(1, 4),
        "c2_aux_labels": torch.full((1, 4), 2),
        "c2_aux_inputs": torch.full((1, 4), 2),
    }
    try:
        losses._require_auxiliary_outputs(
            outputs,
            ("c2_shuffle_gate_where_values", "c2_lodo_shuffle_valid"),
            "C2 gate support contrast",
        )
    except RuntimeError as exc:
        assert "c2_shuffle_gate_where_values" in str(exc)
        assert "c2_lodo_shuffle_valid" in str(exc)
    else:
        raise AssertionError("support contrast must not silently become zero")

    try:
        losses._require_auxiliary_outputs(
            outputs,
            ("c2_aux_canonical_bind_logits", "c2_aux_canonical_bind_support"),
            "canonical VALUE bind auxiliary loss",
        )
    except RuntimeError as exc:
        assert "c2_aux_canonical_bind_logits" in str(exc)
    else:
        raise AssertionError("bind auxiliary loss must not silently become zero")


def test_hierarchical_context_keys_are_collision_free_and_touch_sensitive() -> None:
    import models.recursive_reasoning.relation_map as rel

    assert hasattr(rel, "hierarchical_context_keys"), "missing collision-free context keys"
    sig = torch.tensor(rel.CELL_SIG_NONE).view(1, 1, -1).repeat(1, 2, 1)
    sig[..., rel.SigCol.SELF_COLOR] = 3
    sig[..., rel.SigCol.OBJ_COLOR] = 4
    sig[..., rel.SigCol.ENCL_COLOR_FF] = 5
    sig[..., rel.SigCol.NEAREST_SEED_COLOR] = 6
    sig[..., rel.SigCol.TOUCH_COLOUR_MODE] = torch.tensor([[2, 7]])
    sig[..., rel.SigCol.TOUCH_COLOUR_COUNT] = 1
    valid = torch.ones((1, 2), dtype=torch.bool)
    keys = rel.hierarchical_context_keys(sig, valid)
    assert keys.shape == (1, 2, 5)
    assert int(keys[0, 0, 0]) == int(keys[0, 1, 0]), "K0 should be source-colour only"
    assert int(keys[0, 0, 2]) != int(keys[0, 1, 2]), "touch colour must disambiguate K2"
    assert int(keys.max()) < 2**63 - 1

    # Background is a semantic role, not ARC colour zero. Same source colour, different
    # object attribution, must diverge at K1 even when the background colour is nonzero.
    role_sig = torch.tensor(rel.CELL_SIG_NONE).view(1, 1, -1).repeat(1, 2, 1)
    role_sig[..., rel.SigCol.SELF_COLOR] = 5
    role_sig[0, 0, rel.SigCol.OBJ_COLOR] = 5   # foreground object
    role_sig[0, 1, rel.SigCol.OBJ_COLOR] = 10  # outside background sentinel
    role_keys = rel.hierarchical_context_keys(role_sig, torch.ones((1, 2), dtype=torch.bool))
    assert int(role_keys[0, 0, 0]) == int(role_keys[0, 1, 0])
    assert int(role_keys[0, 0, 1]) != int(role_keys[0, 1, 1])


def test_hierarchical_value_binding_uses_context_and_copy_backoff() -> None:
    import models.recursive_reasoning.relation_map as rel

    base = torch.tensor(rel.CELL_SIG_NONE).view(1, 1, -1).repeat(1, 3, 1)
    base[..., rel.SigCol.SELF_COLOR] = 3
    base[..., rel.SigCol.ENCL_COLOR_FF] = 5
    base[..., rel.SigCol.NEAREST_SEED_COLOR] = 6
    base[..., rel.SigCol.TOUCH_COLOUR_COUNT] = 1
    base[0, 0, rel.SigCol.TOUCH_COLOUR_MODE] = 2
    base[0, 1, rel.SigCol.TOUCH_COLOUR_MODE] = 7
    base[0, 2, rel.SigCol.TOUCH_COLOUR_MODE] = 9
    dst = torch.tensor([[4, 8, 3]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    changed = torch.tensor([[True, True, False]])
    target = base[0, [0, 1, 2]].clone()
    result = rel.hierarchical_value_binding(base, dst, valid, changed, target, valid[0], tau=0.1)
    pred = result["distribution"].argmax(dim=-1)
    assert pred.tolist() == [4, 8, 3]
    assert int(result["collision_count"]) == 0


def test_hierarchical_value_binding_separates_changed_and_copy_support() -> None:
    import models.recursive_reasoning.relation_map as rel

    signature = torch.tensor(rel.CELL_SIG_NONE).view(1, 1, -1).repeat(1, 3, 1)
    signature[..., rel.SigCol.SELF_COLOR] = 3
    signature[..., rel.SigCol.ENCL_COLOR_FF] = 5
    signature[..., rel.SigCol.NEAREST_SEED_COLOR] = 6
    signature[..., rel.SigCol.TOUCH_COLOUR_COUNT] = 1
    signature[0, :, rel.SigCol.TOUCH_COLOUR_MODE] = torch.tensor([2, 7, 9])
    destination = torch.tensor([[4, 8, 3]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    changed = torch.tensor([[True, True, False]])

    result = rel.hierarchical_value_binding(
        signature, destination, valid, changed, signature[0].clone(), valid[0], tau=0.1)

    # P1 outcome-specific backoff: each outcome reports the DEEPEST level where IT was observed.
    # Cells 0/1 see their own changed evidence all the way to K4 (their support twin matches every
    # deeper column) and inherit copy evidence from the shared K1 ancestor; cell 2 sees its own
    # copy at K4 and the ancestors' changed pair at K1. The old contract zeroed ancestor support
    # at the deepest joint level -- that was the Phase-1 Block-1 bug, not a feature.
    assert result["changed_support_count"].tolist() == [1.0, 1.0, 2.0]
    assert result["copy_support_count"].tolist() == [1.0, 1.0, 1.0]
    assert result["changed_supported"].tolist() == [True, True, True]
    assert result["copy_supported"].tolist() == [True, True, True]
    assert result["changed_level_used"].tolist() == [4, 4, 1]
    assert result["copy_level_used"].tolist() == [1, 1, 4]


def test_positive_where_gate_is_nonnegative_and_gates_global_context() -> None:
    from models.recursive_reasoning.trm_fvr_v2 import TestConditionedC2

    cfg = _base_config(
        c2_per_token_gate=True,
        c2_token_gate_where=True,
        c2_positive_where_gate=True,
        c2_ordered_evidence_flow=True,
        c2_rel_where_hint=True,
        c2_gate_init=0.0,
        c2_rel_where_topk=2,
    )
    c2 = TestConditionedC2(cfg).eval()
    target = torch.randn(1, 16, 128)
    ci = torch.full((1, 1, 16), 2, dtype=torch.long)
    co = ci.clone(); co[..., 0] = 3
    cin = torch.randn(1, 1, 16, 128)
    cout = torch.randn(1, 1, 16, 128)
    cm = torch.ones((1, 1), dtype=torch.bool)
    prior = torch.rand(1, 16, 2)
    with torch.no_grad():
        out, metrics, _ = c2(target, ci, co, cin, cout, cm, target_where_hint=prior)
    q = metrics["c2_gate_where_values"]
    assert bool((q >= 0).all() and (q <= 1).all()), "WHERE selector must be in [0,1]"
    assert torch.equal(out, target), "zero gate strengths must make the flag-on path an exact no-op"
    with torch.no_grad():
        c2.gate_patch.fill_(1.0)
        c2.gate_global.fill_(1.0)
        c2.gate_patch_token.weight.zero_()
        c2.gate_patch_token.bias.fill_(-30.0)
        out_closed, _, _ = c2(target, ci, co, cin, cout, cm, target_where_hint=prior)
    assert torch.allclose(out_closed, target, atol=1e-6), "global context bypassed the closed WHERE gate"


def test_ordered_flow_implies_support_relmap_projection() -> None:
    from models.recursive_reasoning.trm_fvr_v2 import TinyRecursiveReasoningModel_ACTV1_Inner

    cfg = _base_config(
        c2_relmap=True,
        c2_rel_where_hint=True,
        c2_ordered_evidence_flow=True,
        c2_relmap_demos=False,
    )
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(cfg)
    assert hasattr(inner, "c2_demo_relmap_proj"), (
        "ordered evidence must make support memory relation-aware without a second enabling flag")


def test_positive_where_gate_rejects_nonzero_strength_init() -> None:
    from models.recursive_reasoning.trm_fvr_v2 import TestConditionedC2

    cfg = _base_config(
        c2_relmap=True,
        c2_rel_where_hint=True,
        c2_per_token_gate=True,
        c2_token_gate_where=True,
        c2_positive_where_gate=True,
        c2_ordered_evidence_flow=True,
        c2_gate_init=0.3,
    )
    try:
        TestConditionedC2(cfg)
    except ValueError as exc:
        assert "c2_gate_init=0" in str(exc)
    else:
        raise AssertionError("positive WHERE gate accepted a nonzero step-0 update strength")


def _where_gate_fixture(*, selector_detach: bool):
    from models.recursive_reasoning.trm_fvr_v2 import TestConditionedC2

    cfg = _base_config(
        c2_per_token_gate=True,
        c2_token_gate_where=True,
        c2_positive_where_gate=True,
        c2_ordered_evidence_flow=True,
        c2_rel_where_hint=True,
        c2_gate_selector_detach=selector_detach,
        c2_gate_init=0.0,
        c2_rel_where_topk=2,
    )
    c2 = TestConditionedC2(cfg).train()
    with torch.no_grad():
        c2.gate_patch.fill_(0.7)
        c2.gate_global.fill_(0.5)
    target = torch.randn(1, 16, 128)
    ci = torch.full((1, 1, 16), 2, dtype=torch.long)
    co = ci.clone(); co[..., 0] = 3
    cin = torch.randn(1, 1, 16, 128)
    cout = torch.randn(1, 1, 16, 128)
    cm = torch.ones((1, 1), dtype=torch.bool)
    prior = torch.rand(1, 16, 2)
    return c2, (target, ci, co, cin, cout, cm, prior)


def test_where_selector_detach_preserves_forward_values() -> None:
    torch.manual_seed(71)
    coupled, args = _where_gate_fixture(selector_detach=False)
    torch.manual_seed(83)
    detached, _ = _where_gate_fixture(selector_detach=True)
    detached.load_state_dict(coupled.state_dict())
    coupled.eval(); detached.eval()
    with torch.no_grad():
        out_c, metrics_c, _ = coupled(*args[:-1], target_where_hint=args[-1])
        out_d, metrics_d, _ = detached(*args[:-1], target_where_hint=args[-1])
    assert torch.equal(out_c, out_d), "selector detach changed forward C2 values"
    assert torch.equal(
        metrics_c["c2_gate_where_values"], metrics_d["c2_gate_where_values"]), (
            "selector detach changed the supervised WHERE tensor")


def test_where_selector_detach_separates_transport_and_where_gradients() -> None:
    torch.manual_seed(97)
    c2, args = _where_gate_fixture(selector_detach=True)
    out, metrics, _ = c2(*args[:-1], target_where_hint=args[-1])

    transport_grad, patch_strength_grad = torch.autograd.grad(
        out.square().mean(),
        (c2.gate_patch_token.weight, c2.gate_patch),
        retain_graph=True,
        allow_unused=True,
    )
    assert transport_grad is None or torch.count_nonzero(transport_grad) == 0, (
        "candidate/transport loss leaked into the WHERE selector")
    assert patch_strength_grad is not None and float(patch_strength_grad.abs()) > 0.0, (
        "detaching the selector also blocked the trainable C2 transport strength")

    where_grad = torch.autograd.grad(
        metrics["c2_gate_where_values"].mean(),
        c2.gate_patch_token.weight,
        allow_unused=True,
    )[0]
    assert where_grad is not None and int(torch.count_nonzero(where_grad)) > 0, (
        "explicit WHERE supervision no longer reaches the selector")


def test_canonical_binder_routes_pure_recolour_and_stays_zero_init() -> None:
    from models.recursive_reasoning.trm_fvr_v2 import TinyRecursiveReasoningModel_ACTV1_Inner

    cfg = _base_config(
        c2_canonical_value_binder=True,
        c2_relmap=True,
        c2_dual_output_head=True,
        c2_geometry_aux_head=False,
        c2_task_palette_feature=False,
        c2_color_head_mlp_dim=0,
    )
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(cfg).eval()
    x = torch.full((1, 1, 900), 2, dtype=torch.long)
    y = x.clone()
    x[0, 0, 62:64] = 3
    y[0, 0, 62:64] = 7
    batch = {
        "inputs": x[:, 0].clone(),
        "context_inputs": x,
        "context_outputs": y,
        "context_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    dist, stats, support, changed_support, copy_support, per_cell = inner._canonical_value_binding(
        batch, 1, 900, torch.device("cpu"))
    assert float(stats["c2_canonical_bind_same_position_route"]) > 0.9
    assert int(dist[0, 62].argmax()) == 5  # output token 7 -> colour index 5
    assert bool(support[0, 62])
    assert bool(changed_support[0, 62]) and not bool(copy_support[0, 62])
    assert bool(copy_support[0, 0]) and not bool(changed_support[0, 0])
    assert torch.count_nonzero(inner.color_evidence_proj.weight) == 0


def test_rule_factors_detect_slide_plus_recolour_as_two_operations() -> None:
    import models.recursive_reasoning.core_prior as prior

    assert hasattr(prior, "evidence_rule_factors"), "missing independent operation factors"
    side = 6
    x = torch.full((side, side), 2, dtype=torch.long)
    y = torch.full_like(x, 2)
    x[2:4, 1:3] = 3       # colour 1 object (token 3)
    y[2:4, 3:5] = 6       # same shape, moved right and recoloured
    result = prior.evidence_rule_factors(x.reshape(1, -1), y.reshape(1, -1), side)
    names = {name: i for i, name in enumerate(prior.RULE_FACTOR_NAMES)}
    scores = result["scores"]
    assert float(scores[names["colour_recolour"]]) > 0.9
    assert float(scores[names["move_right"]]) > 0.9
    assert float(result["same_shape_transport"]) > 0.9


def test_rule_colour_factor_detects_background_recolour() -> None:
    import models.recursive_reasoning.core_prior as prior

    side = 6
    x = torch.full((side, side), 2, dtype=torch.long)
    y = torch.full((side, side), 7, dtype=torch.long)
    x[2:4, 2:4] = 3
    y[2:4, 2:4] = 3
    result = prior.evidence_rule_factors(x.reshape(1, -1), y.reshape(1, -1), side)
    idx = prior.RULE_FACTOR_INDEX
    assert float(result["scores"][idx["colour_recolour"]]) > 0.9
    assert float(result["scores"][idx["extent_same"]]) > 0.9


def test_object_correspondence_ignores_colour_but_preserves_shape() -> None:
    import models.recursive_reasoning.core_prior as prior

    side = 6
    x = torch.full((side, side), 2, dtype=torch.long)
    y = torch.full_like(x, 2)
    x[1:3, 1:3] = 3
    y[1:3, 3:5] = 8
    corr = prior.object_correspondences(x.reshape(-1), y.reshape(-1), side)
    assert len(corr["matched"]) == 1
    assert corr["matched"][0]["dx"] == 2.0
    assert corr["coverage"] == 1.0


def test_pairdelta_can_keep_identity_demos_as_negative_evidence() -> None:
    from models.recursive_reasoning.pair_delta_v2 import demo_delta_features

    x = torch.full((1, 1, 900), 2, dtype=torch.long)
    cm = torch.ones((1, 1), dtype=torch.bool)
    _, legacy_valid = demo_delta_features(x, x.clone(), cm)
    _, identity_valid = demo_delta_features(x, x.clone(), cm, include_identity=True)
    assert not bool(legacy_valid[0, 0]), "legacy contract must remain unchanged"
    assert bool(identity_valid[0, 0]), "identity demo should be usable when explicitly enabled"


def test_pairdelta_spatial_features_separate_translation_from_recolour() -> None:
    from models.recursive_reasoning.pair_delta_v2 import (
        PDS_DIRECTION_CONSISTENCY,
        PDS_DX,
        PDS_DY,
        PDS_SAME_SHAPE_TRANSPORT,
        spatial_delta_features,
    )

    side = 6
    x = torch.full((2, 1, side * side), 2, dtype=torch.long)
    y = x.clone()
    # Row 0: intact 2x2 object translated two cells to the right.
    x[0, 0].view(side, side)[2:4, 1:3] = 3
    y[0, 0].view(side, side)[2:4, 1:3] = 2
    y[0, 0].view(side, side)[2:4, 3:5] = 3
    # Row 1: same-position pure recolour, which must not be misrouted as movement.
    x[1, 0].view(side, side)[2:4, 1:3] = 3
    y[1, 0].view(side, side)[2:4, 1:3] = 7
    cm = torch.ones((2, 1), dtype=torch.bool)
    spatial, valid = spatial_delta_features(x, y, cm)
    assert bool(valid.all())
    assert abs(float(spatial[0, 0, PDS_DY])) < 1e-6
    assert float(spatial[0, 0, PDS_DX]) > 0.3
    assert float(spatial[0, 0, PDS_DIRECTION_CONSISTENCY]) > 0.9
    assert float(spatial[0, 0, PDS_SAME_SHAPE_TRANSPORT]) > 0.9
    assert abs(float(spatial[1, 0, PDS_DY])) < 1e-6
    assert abs(float(spatial[1, 0, PDS_DX])) < 1e-6


def test_pairdelta_spatial_branch_is_zero_init_noop() -> None:
    from models.recursive_reasoning.pair_delta_v2 import PairDeltaEncoder

    torch.manual_seed(17)
    legacy = PairDeltaEncoder(hidden_dim=32, n_slots=2, n_heads=2, include_spatial=False).eval()
    torch.manual_seed(29)
    spatial = PairDeltaEncoder(hidden_dim=32, n_slots=2, n_heads=2, include_spatial=True).eval()
    spatial.load_state_dict(legacy.state_dict(), strict=False)
    x = torch.full((1, 1, 36), 2, dtype=torch.long)
    y = x.clone()
    x[0, 0].view(6, 6)[2:4, 1:3] = 3
    y[0, 0].view(6, 6)[2:4, 3:5] = 3
    cm = torch.ones((1, 1), dtype=torch.bool)
    with torch.no_grad():
        a = legacy(x, y, cm)
        b = spatial(x, y, cm)
    assert torch.equal(a["rule_vec"], b["rule_vec"])
    assert torch.equal(a["rule_slots"], b["rule_slots"])
    assert float(b["spatial_feature_norm"]) > 0.0


def test_extent_route_preserves_floor_or_allows_candidate() -> None:
    import models.recursive_reasoning.trm_fvr_v2 as model

    assert hasattr(model, "extent_conditioned_structure"), "missing extent-conditioned structure route"
    floor = torch.randn(2, 5, 3)
    candidate = torch.randn(2, 5, 3)
    routed = model.extent_conditioned_structure(floor, candidate, torch.tensor([1.0, 0.0]))
    assert torch.equal(routed[0], floor[0])
    assert torch.equal(routed[1], candidate[1])


def test_bind_per_ex_undefined_subsets_are_nan() -> None:
    """A row with no supported changed cells must report NaN changed acc/exact (never acc=0 +
    exact=1); a row with no copy cells must report NaN copy acc/exact. Scalars stay finite."""
    import math

    import models.losses_fvr as losses

    # Row 0: pure copy (input == target everywhere). Row 1: pure change (all cells recoloured).
    target = torch.tensor([[3, 3, 3, 3], [4, 4, 4, 4]])
    inputs = torch.tensor([[3, 3, 3, 3], [3, 3, 3, 3]])
    support = torch.ones_like(target, dtype=torch.bool)
    logits = torch.zeros(2, 4, 10)
    logits[..., 1] = 5.0   # predicts colour index 1 -> token 3 everywhere
    total, _, _, chg_acc, cpy_acc, cov, per_ex = losses._value_ctx_bind_aux_ce(
        logits, target, inputs, support, 3.0, 2.0)
    assert torch.isfinite(total), "scalar loss must stay finite"
    assert torch.isfinite(chg_acc) and torch.isfinite(cpy_acc) and torch.isfinite(cov)
    r0 = {k: float(v[0]) for k, v in per_ex.items()}
    r1 = {k: float(v[1]) for k, v in per_ex.items()}
    assert math.isnan(r0["c2_bind_changed_acc_per_ex"]), "copy-only row: changed acc must be NaN"
    assert math.isnan(r0["c2_bind_changed_exact_per_ex"]), "copy-only row: changed exact must be NaN (was vacuous 1.0)"
    assert r0["c2_bind_copy_acc_per_ex"] == 1.0, "copy-only row: copy acc is defined (predicts 3 == target)"
    assert r0["c2_bind_copy_exact_per_ex"] == 1.0
    assert math.isnan(r1["c2_bind_copy_acc_per_ex"]), "all-changed row: copy acc must be NaN"
    assert math.isnan(r1["c2_bind_copy_exact_per_ex"]), "all-changed row: copy exact must be NaN"
    assert not math.isnan(r1["c2_bind_changed_acc_per_ex"])


def test_canonical_bind_residual_loss_trains_delivery_not_standalone_logits() -> None:
    import models.losses_fvr as losses

    # Cell 0 changes token 3 -> 4. The binder alone prefers the target, but it is still too weak
    # to overcome the deployed base logit's wrong class. Cell 1 is a copy cell whose residual
    # would flip a correct base prediction; Repair B must report that flip without training on it.
    target = torch.tensor([[4, 3]])
    inputs = torch.tensor([[3, 3]])
    base = torch.zeros(1, 2, 10, requires_grad=True)
    with torch.no_grad():
        base[0, 0, 1] = 5.0
        base[0, 1, 1] = 3.0
    bind = torch.zeros(1, 2, 10, requires_grad=True)
    with torch.no_grad():
        bind[0, 0, 2] = 1.0
        bind[0, 1, 2] = 4.0
    changed_support = torch.tensor([[True, False]])
    copy_support = torch.tensor([[False, True]])

    result = losses._canonical_bind_residual_ce(
        base, bind, target, inputs, changed_support, copy_support, changed_w=3.0)

    assert float(result["changed_acc"]) == 0.0, "metric must score base + residual, not bind alone"
    assert float(result["base_wrong_margin"]) == 5.0
    assert float(result["corrected_changed_frac"]) == 0.0
    assert float(result["caused_copy_flip_frac"]) == 1.0
    assert float(result["changed_support_coverage"]) == 1.0
    assert float(result["copy_support_coverage"]) == 1.0

    result["loss"].backward()
    assert base.grad is None, "Repair B must not move the current candidate/base through this loss"
    assert int(torch.count_nonzero(bind.grad[0, 0])) > 0, "changed residual received no delivery gradient"
    assert int(torch.count_nonzero(bind.grad[0, 1])) == 0, "copy diagnostics leaked into the changed-only loss"


def test_nonfinite_diagnostics_name_component_and_tensor_failure() -> None:
    import models.losses_fvr as losses

    component_health = losses._loss_component_health({
        "main": torch.tensor(1.25),
        "bind": torch.tensor(float("nan")),
        "preserve": torch.tensor(float("inf")),
    })
    assert component_health["values"]["main"] == 1.25
    assert component_health["nonfinite"] == {"bind": "nan", "preserve": "+inf"}

    tensor_health = losses._tensor_health(torch.tensor([0.0, float("nan"), float("inf"), -float("inf")]))
    assert tensor_health["nan"] == 1
    assert tensor_health["posinf"] == 1
    assert tensor_health["neginf"] == 1
    assert tensor_health["finite"] == 1
    assert tensor_health["finite_min"] == 0.0
    assert tensor_health["finite_max"] == 0.0


def test_fixed_eval_failure_identity_includes_batch_seed_and_pids() -> None:
    import scripts.test_run_v2 as runner

    identity = runner._fixed_eval_batch_identity(
        {"puzzle_identifiers": torch.tensor([17, 42])}, batch_index=3, seed=1237)
    assert identity == "fixed_eval_batch=3 seed=1237 puzzle_identifiers=[17, 42]"


def test_where_per_task_exposes_undefined_denominator_masks() -> None:
    """has_changed/has_copy must mark the rows whose F1/FPR are undefined, so the per-ex surfacing
    can emit NaN there. where_mask_exact stays MEANINGFUL on copy-only rows (quiet gate = correct)."""
    import models.losses_fvr as losses

    # Row 0: has one changed cell. Row 1: copy-only (nothing changes). Row 2: every cell changes.
    target = torch.tensor([[3, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
    inputs = torch.tensor([[2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]])
    q = torch.tensor([[0.9, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.9, 0.9]])
    out = losses._where_metrics_per_task(q, target, inputs)
    assert "has_copy" in out, "has_copy mask missing (needed for the FPR NaN contract)"
    assert out["has_changed"].tolist() == [True, False, True]
    assert out["has_copy"].tolist() == [True, True, False]
    # Copy-only row with a quiet gate is a CORRECT exact WHERE prediction, not vacuous.
    assert bool(out["where_mask_exact"][1])
    # All-changed row: FPR undefined (no negatives) -> surfacing must NaN it via has_copy.


def test_mechanism_panel_requires_finite_bind_exactness() -> None:
    """Vacuous (now NaN) bind exactness must be EXCLUDED from the strict conditional subset."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    nan = float("nan")
    rows = [
        # Legit strict-conditional member: all three exact flags finite and 1.
        {"where_f1": 0.9, "where_fpr": 0.0, "bind_changed_acc": 0.9, "bind_coverage": 0.9,
         "candidate_strict_exact": 1.0, "where_mask_exact": 1.0,
         "bind_changed_exact": 1.0, "bind_copy_exact": 1.0},
        # Copy-only row: bind_changed_exact is NaN -> must NOT enter the conditional subset,
        # even though every finite field looks perfect.
        {"where_f1": nan, "where_fpr": 0.0, "bind_changed_acc": nan, "bind_coverage": 0.9,
         "candidate_strict_exact": 1.0, "where_mask_exact": 1.0,
         "bind_changed_exact": nan, "bind_copy_exact": 1.0},
    ]
    cond = [r for r in rows
            if isinstance(r.get("where_mask_exact"), float) and r["where_mask_exact"] >= 0.5
            and isinstance(r.get("bind_changed_exact"), float)
            and r["bind_changed_exact"] == r["bind_changed_exact"] and r["bind_changed_exact"] >= 0.5
            and isinstance(r.get("bind_copy_exact"), float)
            and r["bind_copy_exact"] == r["bind_copy_exact"] and r["bind_copy_exact"] >= 0.5]
    assert len(cond) == 1, "NaN bind exactness leaked into the strict conditional"
    T.mechanism_conditioned_exact(rows)   # must run clean on NaN-bearing rows (prints the panel)


def test_paired_report_guards_family_mismatch_and_bootstrap_samples() -> None:
    """The paired report must refuse: (a) n_boot < 1; (b) the same case labelled with different
    families in treatment vs control (the stratified bootstrap would silently mis-stratify)."""
    import csv
    import importlib.util
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    def _write(path, family):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=T._TASK_CSV_COLS, extrasaction="ignore")
            w.writeheader()
            row = {c: 0.0 for c in T._TASK_CSV_COLS}
            row.update(eval_case_id="case_0", puzzle_identifier=1, family=family)
            w.writerow(row)

    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.csv"); b = os.path.join(d, "b.csv")
        _write(a, "conditional_recolor"); _write(b, "conditional_recolor")
        try:
            T.paired_control_report(a, b, n_boot=0, seed=1)
        except ValueError as exc:
            assert "paired-bootstrap-samples" in str(exc)
        else:
            raise AssertionError("n_boot=0 must raise")
        _write(b, "size_change")
        try:
            T.paired_control_report(a, b, n_boot=10, seed=1)
        except ValueError as exc:
            assert "family label mismatch" in str(exc)
        else:
            raise AssertionError("family mismatch must raise")


def test_task_metrics_finalize_is_nonfatal_and_post_checkpoint() -> None:
    """A diagnostics failure must not propagate (the final checkpoint is already saved when the
    hook runs); the hook must also be a no-op without --task-metrics-out."""
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    T._task_metrics_finalize(SimpleNamespace(args=SimpleNamespace(task_metrics_out="")))  # no-op

    real = T.dump_task_metrics
    T.dump_task_metrics = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        T._task_metrics_finalize(SimpleNamespace(args=SimpleNamespace(
            task_metrics_out="x.csv", paired_control_report="")))   # must swallow + report
    finally:
        T.dump_task_metrics = real
    # Ordering: the finalize call must come AFTER the final save_checkpoint in the source.
    src = (ROOT / "scripts" / "test_run_v2.py").read_text(encoding="utf-8")
    tail = src[src.index("save_checkpoint(S, step, final=True)"):]
    assert "_task_metrics_finalize(S)" in tail, "diagnostics must run after the final checkpoint"


def test_fusion_compression_math_and_patience() -> None:
    """Block 3 contract: raw<1e-6 or non-finite -> 1.0; strict >; streak resets on a dip; the
    warning first fires at patience and keeps firing while the condition persists."""
    import importlib.util
    import math

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    assert T._fusion_compression(10.0, 2.0) == 5.0
    assert T._fusion_compression(0.0, 0.0) == 1.0
    assert T._fusion_compression(float("nan"), 2.0) == 1.0
    assert T._fusion_compression(2.0, float("nan")) == 1.0
    assert math.isclose(T._fusion_compression(1.0, 0.0), 1e6)

    def run(seq, warn=5.0, patience=3):
        streaks, fires = {}, []
        for i, c in enumerate(seq):
            if T._fusion_compression_update(streaks, {"rel": c}, warn, patience):
                fires.append(i)
        return fires

    assert run([6.0, 6.0, 6.0]) == [2]
    assert run([6.0, 6.0, 4.0, 6.0, 6.0, 6.0]) == [5]
    assert run([6.0, 6.0, 6.0, 6.0]) == [2, 3]
    assert run([5.0, 5.0, 5.0]) == []            # == threshold is NOT exceeded
    assert run([100.0] * 4, warn=0.0) == []      # warn<=0 disables


def _p1_sig_cell(rel, **cols) -> torch.Tensor:
    """A fully-specified conditioning-signature cell for level-targeted P1 fixtures. Defaults pin
    every key column so a single overridden column moves EXACTLY one hierarchy level."""
    c = torch.tensor(rel.CELL_SIG_NONE, dtype=torch.long).clone()
    c[rel.SigCol.SELF_COLOR] = 3
    c[rel.SigCol.OBJ_COLOR] = 3
    c[rel.SigCol.ENCL_COLOR_FF] = 5
    c[rel.SigCol.TOUCH_COLOUR_MODE] = 2
    c[rel.SigCol.TOUCH_COLOUR_COUNT] = 1
    c[rel.SigCol.NEAREST_SEED_COLOR] = 6
    c[rel.SigCol.OBJ_SIZE_RANK] = 1
    c[rel.SigCol.OBJ_HOLES] = 0
    c[rel.SigCol.LOCAL_ROW3] = 1
    c[rel.SigCol.LOCAL_COL3] = 1
    for k, v in cols.items():
        c[getattr(rel.SigCol, k)] = v
    return c


def test_p1_changed_backoff_survives_deeper_copy_only_context() -> None:
    import models.recursive_reasoning.relation_map as rel

    target = _p1_sig_cell(rel).view(1, -1)
    # Support A: CHANGED, matches the target through K2 but differs at K3 (nearest seed).
    # Support B: COPY, matches the target through K4 exactly.
    support = torch.stack([_p1_sig_cell(rel, NEAREST_SEED_COLOR=9), _p1_sig_cell(rel)]).view(1, 2, -1)
    dst = torch.tensor([[7, 0]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    changed = torch.tensor([[True, False]])
    out = rel.hierarchical_value_binding(
        support, dst, valid, changed, target, torch.ones(1, dtype=torch.bool), tau=3.0)
    assert int(out["changed_level_used"][0]) == 2, \
        "deeper copy-only K3/K4 must not erase the K2 changed support"
    assert bool(out["changed_supported"][0]) and float(out["changed_support_count"][0]) == 1.0
    assert int(out["copy_level_used"][0]) == 4
    assert bool(out["copy_supported"][0]) and float(out["copy_support_count"][0]) == 1.0


def test_p1_copy_backoff_survives_deeper_changed_only_context() -> None:
    import models.recursive_reasoning.relation_map as rel

    target = _p1_sig_cell(rel).view(1, -1)
    # Support A: COPY, matches through K1 only (different enclosure -> K2 miss).
    # Support B: CHANGED, matches through K4 exactly.
    support = torch.stack([_p1_sig_cell(rel, ENCL_COLOR_FF=4), _p1_sig_cell(rel)]).view(1, 2, -1)
    dst = torch.tensor([[0, 8]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    changed = torch.tensor([[False, True]])
    out = rel.hierarchical_value_binding(
        support, dst, valid, changed, target, torch.ones(1, dtype=torch.bool), tau=3.0)
    assert int(out["copy_level_used"][0]) == 1, \
        "deeper changed-only K2..K4 must not erase the K1 copy support"
    assert bool(out["copy_supported"][0]) and float(out["copy_support_count"][0]) == 1.0
    assert int(out["changed_level_used"][0]) == 4


def test_p1_context_moves_posterior_but_not_marginal() -> None:
    import models.recursive_reasoning.relation_map as rel

    support = torch.stack([_p1_sig_cell(rel, ENCL_COLOR_FF=5), _p1_sig_cell(rel, ENCL_COLOR_FF=4)]).view(1, 2, -1)
    dst = torch.tensor([[4, 8]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    changed = torch.tensor([[True, True]])
    tv = torch.ones(1, dtype=torch.bool)
    out5 = rel.hierarchical_value_binding(
        support, dst, valid, changed, _p1_sig_cell(rel, ENCL_COLOR_FF=5).view(1, -1), tv, tau=3.0)
    out4 = rel.hierarchical_value_binding(
        support, dst, valid, changed, _p1_sig_cell(rel, ENCL_COLOR_FF=4).view(1, -1), tv, tau=3.0)
    assert torch.allclose(out5["marginal_distribution"], out4["marginal_distribution"], atol=1e-6), \
        "K0 marginal must be context-free"
    assert int(out5["distribution"][0].argmax()) == 4
    assert int(out4["distribution"][0].argmax()) == 8


def test_p1_route_zero_does_not_scale_or_zero_distribution() -> None:
    from models.recursive_reasoning.trm_fvr_v2 import TinyRecursiveReasoningModel_ACTV1_Inner

    cfg = _base_config(
        c2_canonical_value_binder=True, c2_relmap=True, c2_dual_output_head=True,
        c2_geometry_aux_head=False, c2_task_palette_feature=False, c2_color_head_mlp_dim=0)
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(cfg).eval()
    x = torch.full((1, 1, 900), 2, dtype=torch.long)
    y = x.clone()
    x[0, 0, 62:64] = 3
    y[0, 0, 92:94] = 3     # pure movement (down one row): recolour factor ~0 -> route ~0
    batch = {"inputs": x[:, 0].clone(), "context_inputs": x, "context_outputs": y,
             "context_mask": torch.ones((1, 1), dtype=torch.bool)}
    dist, stats, support, chg_s, cpy_s, per_cell = inner._canonical_value_binding(
        batch, 1, 900, torch.device("cpu"))
    assert float(stats["c2_canonical_bind_same_position_route"]) < 0.1
    row_mass = dist.sum(-1)
    live = row_mass > 0
    assert bool(live.any())
    assert torch.all((row_mass[live] - 1.0).abs() <= 1e-5), \
        "route must not scale P_bind (confidence, not probability mass)"
    assert bool(support.any()), "route~0 must not gate support flags off"


def test_p1_reliability_and_route_are_separate_tensors() -> None:
    from models.recursive_reasoning.trm_fvr_v2 import TinyRecursiveReasoningModel_ACTV1_Inner

    cfg = _base_config(
        c2_canonical_value_binder=True, c2_relmap=True, c2_dual_output_head=True,
        c2_geometry_aux_head=False, c2_task_palette_feature=False, c2_color_head_mlp_dim=0)
    inner = TinyRecursiveReasoningModel_ACTV1_Inner(cfg).eval()
    x = torch.full((1, 1, 900), 2, dtype=torch.long)
    y = x.clone()
    x[0, 0, 62:64] = 3
    y[0, 0, 62:64] = 7
    batch = {"inputs": x[:, 0].clone(), "context_inputs": x, "context_outputs": y,
             "context_mask": torch.ones((1, 1), dtype=torch.bool)}
    _d, _s, _sup, _cs, _ps, per_cell = inner._canonical_value_binding(
        batch, 1, 900, torch.device("cpu"))
    route, rel_t = per_cell["route"], per_cell["reliability"]
    assert route.shape == (1, 900) and rel_t.shape == (1, 900)
    assert route.data_ptr() != rel_t.data_ptr()
    assert not torch.equal(route, rel_t), "reliability (n/(n+tau)) must not be the route"
    assert float(rel_t.max()) < 1.0 and float(rel_t.min()) >= 0.0


def test_p1_distributions_finite_and_normalized() -> None:
    import models.recursive_reasoning.relation_map as rel

    target = torch.stack([_p1_sig_cell(rel), _p1_sig_cell(rel, ENCL_COLOR_FF=4)])
    support = torch.stack([_p1_sig_cell(rel, NEAREST_SEED_COLOR=9), _p1_sig_cell(rel)]).view(1, 2, -1)
    dst = torch.tensor([[7, 0]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    changed = torch.tensor([[True, False]])
    out = rel.hierarchical_value_binding(
        support, dst, valid, changed, target, torch.ones(2, dtype=torch.bool), tau=3.0)
    for key in ("distribution", "marginal_distribution"):
        d = out[key]
        assert torch.isfinite(d).all(), f"{key} not finite"
        assert torch.all((d.sum(-1) - 1.0).abs() <= 1e-6), f"{key} not normalized"


def test_p1_invalid_cells_stay_all_zero() -> None:
    import models.recursive_reasoning.relation_map as rel

    target = torch.stack([_p1_sig_cell(rel), _p1_sig_cell(rel)])
    support = _p1_sig_cell(rel).view(1, 1, -1)
    dst = torch.tensor([[7]])
    valid = torch.ones((1, 1), dtype=torch.bool)
    changed = torch.tensor([[True]])
    t_valid = torch.tensor([True, False])
    out = rel.hierarchical_value_binding(support, dst, valid, changed, target, t_valid, tau=3.0)
    assert float(out["distribution"][1].abs().sum()) == 0.0
    assert float(out["marginal_distribution"][1].abs().sum()) == 0.0
    assert float(out["support_count"][1]) == 0.0 and float(out["support_reliability"][1]) == 0.0

    try:
        rel.hierarchical_value_binding(support, dst, valid, changed, target, t_valid, tau=0.0)
    except ValueError as exc:
        assert "tau" in str(exc)
    else:
        raise AssertionError("tau <= 0 must be rejected")


def test_p1_fixed_replacement_claim_is_support_only() -> None:
    import models.losses_fvr as losses

    # Cells: 0 changed+supported, 1 changed+UNsupported, 2 copy+supported, 3 copy+supported.
    t = torch.tensor([[5, 5, 3, 2]])
    inp = torch.tensor([[3, 3, 3, 2]])
    base_logits = torch.zeros(1, 4, 10)
    base_logits[0, 0, 1] = 5.0   # base predicts token 3 -> WRONG on changed cell 0
    base_logits[0, 1, 1] = 5.0   # WRONG on changed cell 1
    base_logits[0, 2, 1] = 5.0   # token 3 -> right on copy cell 2
    base_logits[0, 3, 0] = 5.0   # token 2 -> right on copy cell 3
    dist = torch.zeros(1, 4, 10)
    dist[0, 0, 3] = 1.0          # binder right on cell 0 (token 5)
    dist[0, 1, 3] = 1.0          # binder WOULD be right on cell 1 -- but it is unsupported
    dist[0, 2, 1] = 1.0
    dist[0, 3, 0] = 1.0
    chg_sup = torch.tensor([[True, False, False, False]])
    cpy_sup = torch.tensor([[False, False, True, True]])
    ones = torch.ones(1, 4)
    out = losses._p1_value_extraction_metrics(
        base_logits, dist, dist.clone(), t, inp, chg_sup, cpy_sup, ones, ones)
    # Claim is support-derived: cell 1 stays on the (wrong) base even though the binder knows better.
    assert abs(float(out["p1_fixed_replacement_gain_per_ex"][0]) - 0.5) < 1e-6
    assert abs(float(out["p1_fixed_replacement_copy_loss_per_ex"][0])) < 1e-6
    assert float(out["p1_raw_bind_changed_acc_per_ex"][0]) == 1.0   # on the supported changed cell
    assert float(out["p1_changed_supported_cells_per_ex"][0]) == 1.0


def test_p1_empty_subsets_are_nan() -> None:
    import math

    import models.losses_fvr as losses

    # Row 0: pure copy (no changed cells). Row 1: pure change (no copy cells).
    t = torch.tensor([[3, 3], [5, 5]])
    inp = torch.tensor([[3, 3], [3, 3]])
    base_logits = torch.zeros(2, 2, 10)
    dist = torch.zeros(2, 2, 10); dist[..., 3] = 1.0
    sup = torch.ones(2, 2, dtype=torch.bool)
    ones = torch.ones(2, 2)
    out = losses._p1_value_extraction_metrics(
        base_logits, dist, dist.clone(), t, inp, sup, sup, ones, ones)
    assert math.isnan(float(out["p1_raw_bind_changed_acc_per_ex"][0]))
    assert math.isnan(float(out["p1_fixed_replacement_gain_per_ex"][0]))
    assert math.isnan(float(out["p1_effective_changed_coverage_per_ex"][0]))
    assert not math.isnan(float(out["p1_raw_bind_copy_acc_per_ex"][0]))
    assert math.isnan(float(out["p1_raw_bind_copy_acc_per_ex"][1]))
    assert math.isnan(float(out["p1_fixed_replacement_copy_loss_per_ex"][1]))
    assert not math.isnan(float(out["p1_raw_bind_changed_acc_per_ex"][1]))


def test_p1_semver_bump_rejects_v3_canonical_checkpoint() -> None:
    import pretrain
    import models.recursive_reasoning.trm_fvr_v2 as trm

    cfg = _base_config(
        c2_canonical_value_binder=True, c2_relmap=True, c2_dual_output_head=True,
        c2_geometry_aux_head=False, c2_task_palette_feature=False, c2_color_head_mlp_dim=0)
    assert trm.EVIDENCE_SCHEMA_SEMVER == 4, "P1 route/support semantics change requires semver 4"
    current = trm.evidence_schema_fingerprint(cfg)
    saved = trm.EVIDENCE_SCHEMA_SEMVER
    trm.EVIDENCE_SCHEMA_SEMVER = 3
    try:
        stale = trm.evidence_schema_fingerprint(cfg)
    finally:
        trm.EVIDENCE_SCHEMA_SEMVER = saved
    assert not torch.equal(current, stale)
    schema = "model.inner.evidence_schema_fingerprint"
    state = {schema: stale, "model.inner.lm_head.weight": torch.ones(1)}
    try:
        pretrain._prepare_evidence_schema_state(
            state, {schema: current}, schema, allow_legacy=True)
    except RuntimeError as exc:
        assert "semantically incompatible" in str(exc)
    else:
        raise AssertionError("a v3 canonical fingerprint must be rejected, not silently accepted")


def test_p3a_contract_check_covers_all_valid_cells() -> None:
    """P3A Block 0: finiteness/normalization failures must be counted on EVERY valid cell,
    not only cells the binder claims -- an unclaimed NaN or broken simplex is still a defect."""
    import models.losses_fvr as losses

    B, L = 1, 4
    base_logits = torch.zeros(B, L, 10)
    dist = torch.full((B, L, 10), 0.1)
    marg = torch.full((B, L, 10), 0.1)
    target = torch.full((B, L), 3, dtype=torch.long)
    inputs = torch.full((B, L), 2, dtype=torch.long)
    no_sup = torch.zeros(B, L, dtype=torch.bool)     # NOTHING is claimed anywhere
    ones = torch.ones(B, L)
    dist[0, 1, 0] = float("nan")                      # non-finite on an UNCLAIMED valid cell
    marg[0, 2, :] = 0.05                              # marginal sums to 0.5 on an unclaimed valid cell
    out = losses._p1_value_extraction_metrics(
        base_logits, dist, marg, target, inputs, no_sup, no_sup, ones, ones)
    assert float(out["p1_finite_fail_per_ex"][0]) == 1.0, (
        "a NaN on an unclaimed valid cell must count as a finite failure")
    assert float(out["p1_norm_fail_per_ex"][0]) == 1.0, (
        "a broken marginal simplex on an unclaimed valid cell must count as a norm failure")

    # Invalid cells stay exempt: same defects on cells with target/input < 2 must count zero.
    dist2 = torch.full((B, L, 10), 0.1)
    dist2[0, 1, 0] = float("nan")
    bad_target = torch.tensor([[3, 0, 1, 3]])         # cells 1,2 invalid
    out2 = losses._p1_value_extraction_metrics(
        base_logits, dist2, dist2.clone(), bad_target, inputs, no_sup, no_sup, ones, ones)
    assert float(out2["p1_finite_fail_per_ex"][0]) == 0.0
    assert float(out2["p1_norm_fail_per_ex"][0]) == 0.0


def test_p3a_p1_verdict_unavailable_without_primary_family() -> None:
    """P3A Block 0: without conditional_recolor rows the P1 verdict is UNAVAILABLE (return None,
    JSON verdict field) -- never an ALL-families substitute stamped PASS/FAIL."""
    import importlib.util
    import json
    import tempfile

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    def _report(rows):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "p1.json")
            res = T.p1_value_extraction_report(rows, {"canonical_key_collisions": 0.0}, path)
            with open(path) as f:
                return res, json.load(f)

    common = {"raw_bind_changed_acc": 0.5, "raw_bind_copy_acc": 0.9,
              "effective_changed_coverage": 0.5, "effective_copy_coverage": 0.5,
              "raw_bind_margin": 0.5, "marginal_bind_changed_acc": 0.4,
              "fixed_replacement_gain": 0.10, "fixed_replacement_copy_loss": 0.0,
              "p1_norm_fail": 0.0, "p1_finite_fail": 0.0, "p1_changed_supported_cells": 10.0}

    # (a) finite gains exist, but only in OTHER families -> UNAVAILABLE, not an ALL fallback.
    res, rep = _report([dict(common, family="other")])
    assert res is None and rep["verdict"] == "UNAVAILABLE" and rep["pass"] is None

    # (b) conditional_recolor rows exist but no finite gain in the family -> UNAVAILABLE.
    res, rep = _report([
        dict(common, family="conditional_recolor", fixed_replacement_gain=float("nan")),
        dict(common, family="other"),
    ])
    assert res is None and rep["verdict"] == "UNAVAILABLE" and rep["pass"] is None

    # (c) control: a finite conditional_recolor gain -> a real boolean verdict with gates.
    res, rep = _report([dict(common, family="conditional_recolor")])
    assert isinstance(res, bool) and isinstance(rep["pass"], bool) and "gates" in rep
    assert rep["family"] == "conditional_recolor"


# ---------------------------------------------------------------------------------------------
# P3A support-conditioned WHERE (Blocks 1-5): the ten pre-registered fixtures.
# ---------------------------------------------------------------------------------------------
def _p3a_c2_config(**overrides):
    values = dict(
        c2_per_token_gate=True,
        c2_token_gate_where=True,
        c2_positive_where_gate=True,
        c2_ordered_evidence_flow=True,
        c2_rel_where_hint=True,
        c2_relmap=True,
        c2_gate_selector_detach=True,
        c2_gate_init=0.0,
        c2_rel_where_topk=2,
    )
    values.update(overrides)
    return _base_config(**values)


def _p3a_support_tensors(seed: int = 11):
    torch.manual_seed(seed)
    target = torch.randn(1, 16, 128)
    ci = torch.full((1, 1, 16), 2, dtype=torch.long)
    co = ci.clone(); co[..., 0] = 3; co[..., 5] = 4
    cin = torch.randn(1, 1, 16, 128)
    cout = torch.randn(1, 1, 16, 128)
    cm = torch.ones((1, 1), dtype=torch.bool)
    return target, ci, co, cin, cout, cm


def test_p3a_flag_off_forward_is_unchanged_and_rejects_query() -> None:
    """P3A test 1: with the new flags off, the C2 forward is bitwise identical whether or not the
    config even mentions them, and passing target_query_features is a hard error (no silent lane)."""
    from models.recursive_reasoning.trm_fvr_v2 import TestConditionedC2

    torch.manual_seed(41)
    legacy = TestConditionedC2(_p3a_c2_config()).eval()          # new fields at their defaults
    torch.manual_seed(43)
    explicit = TestConditionedC2(_p3a_c2_config(
        c2_isolated_relmap_query=False, c2_support_interaction_gate=False,
        c2_lodo_zero_support=False)).eval()
    explicit.load_state_dict(legacy.state_dict())
    target, ci, co, cin, cout, cm = _p3a_support_tensors()
    prior = torch.rand(1, 16, 2)
    with torch.no_grad():
        out_a, m_a, _ = legacy(target, ci, co, cin, cout, cm, target_where_hint=prior)
        out_b, m_b, _ = explicit(target, ci, co, cin, cout, cm, target_where_hint=prior)
    assert torch.equal(out_a, out_b), "explicit-False P3A flags changed the legacy forward"
    assert torch.equal(m_a["c2_gate_where_values"], m_b["c2_gate_where_values"])
    try:
        legacy(target, ci, co, cin, cout, cm, target_query_features=target.clone())
    except ValueError as exc:
        assert "c2_isolated_relmap_query" in str(exc)
    else:
        raise AssertionError("flag-off forward silently accepted an x_query tensor")


def test_p3a_isolated_query_moves_gate_not_recurrent_input() -> None:
    """P3A test 2: with C2 update strengths at zero, changing x_query changes the WHERE gate but
    the returned features stay EXACTLY x_base -- the relmap lane cannot touch recurrence."""
    from models.recursive_reasoning.trm_fvr_v2 import TestConditionedC2

    torch.manual_seed(47)
    c2 = TestConditionedC2(_p3a_c2_config(c2_isolated_relmap_query=True)).eval()
    target, ci, co, cin, cout, cm = _p3a_support_tensors()
    q1 = target + 0.3 * torch.randn_like(target)
    q2 = target - 0.3 * torch.randn_like(target)
    with torch.no_grad():
        out1, m1, _ = c2(target, ci, co, cin, cout, cm, target_query_features=q1)
        out2, m2, _ = c2(target, ci, co, cin, cout, cm, target_query_features=q2)
    assert torch.equal(out1, target) and torch.equal(out2, target), (
        "zero update strengths must return x_base exactly under the isolated query")
    assert not torch.equal(m1["c2_gate_where_values"], m2["c2_gate_where_values"]), (
        "x_query must reach the WHERE gate")
    try:
        c2(target, ci, co, cin, cout, cm)
    except ValueError as exc:
        assert "target_query_features" in str(exc)
    else:
        raise AssertionError("isolated query accepted a forward without x_query")


def test_p3a_interaction_gate_needs_both_target_and_support() -> None:
    """P3A test 3: gate_input = norm(x_query) * norm(patch_context) responds to EITHER factor,
    and with no support at all it collapses to the (zero) bias -- constant q=0.5 everywhere."""
    from models.recursive_reasoning.trm_fvr_v2 import TestConditionedC2

    torch.manual_seed(53)
    c2 = TestConditionedC2(_p3a_c2_config(
        c2_isolated_relmap_query=True, c2_support_interaction_gate=True)).eval()
    with torch.no_grad():
        c2.gate_patch_token.weight.normal_(mean=0.0, std=0.05)   # visible sensitivity
    target, ci, co, cin, cout, cm = _p3a_support_tensors()
    q = target + 0.3 * torch.randn_like(target)
    q_other = target - 0.3 * torch.randn_like(target)
    cout_other = cout + 0.5 * torch.randn_like(cout)
    with torch.no_grad():
        _, m_base, _ = c2(target, ci, co, cin, cout, cm, target_query_features=q)
        _, m_query, _ = c2(target, ci, co, cin, cout, cm, target_query_features=q_other)
        _, m_supp, _ = c2(target, ci, co, cin, cout_other, cm, target_query_features=q)
        _, m_empty, _ = c2(target, ci, co, cin, cout,
                           torch.zeros((1, 1), dtype=torch.bool), target_query_features=q)
    g = lambda m: m["c2_gate_where_values"]
    assert not torch.equal(g(m_base), g(m_query)), "gate ignored the target factor"
    assert not torch.equal(g(m_base), g(m_supp)), "gate ignored the support factor"
    assert torch.allclose(g(m_empty), torch.full_like(g(m_empty), 0.5), atol=1e-6), (
        "with zero support the multiplicative gate input must collapse to the bias (q=0.5)")


def test_p3a_zero_support_outputs_are_finite() -> None:
    """P3A test 4: an all-false context mask produces finite features and a finite in-[0,1] gate."""
    from models.recursive_reasoning.trm_fvr_v2 import TestConditionedC2

    torch.manual_seed(59)
    c2 = TestConditionedC2(_p3a_c2_config(
        c2_isolated_relmap_query=True, c2_support_interaction_gate=True,
        c2_bounded_evidence_fusion=True)).eval()
    target, ci, co, cin, cout, _cm = _p3a_support_tensors()
    zero_cm = torch.zeros((1, 1), dtype=torch.bool)
    with torch.no_grad():
        out, m, _ = c2(target, ci, co, cin, cout, zero_cm, target_query_features=target)
    assert bool(torch.isfinite(out).all()), "zero-support features must stay finite"
    q = m["c2_gate_where_values"]
    assert bool(torch.isfinite(q).all() and (q >= 0).all() and (q <= 1).all())


def test_p3a_zero_support_update_is_exactly_zero() -> None:
    """P3A test 5: even with OPEN update strengths, zero support cannot move the features -- the
    empty-memory cross-attention and masked mean are exact zeros through bias-free projections."""
    from models.recursive_reasoning.trm_fvr_v2 import TestConditionedC2

    torch.manual_seed(61)
    c2 = TestConditionedC2(_p3a_c2_config(
        c2_isolated_relmap_query=True, c2_support_interaction_gate=True,
        c2_bounded_evidence_fusion=True)).eval()
    with torch.no_grad():
        c2.gate_patch.fill_(1.0)
        c2.gate_global.fill_(1.0)
        c2.gate_patch_token.weight.normal_(mean=0.0, std=0.05)
    target, ci, co, cin, cout, _cm = _p3a_support_tensors()
    zero_cm = torch.zeros((1, 1), dtype=torch.bool)
    with torch.no_grad():
        out, _m, _ = c2(target, ci, co, cin, cout, zero_cm, target_query_features=target)
    assert torch.equal(out, target), "zero support must produce an exactly-zero C2 update"


def test_p3a_where_loss_reaches_cross_attention() -> None:
    """P3A test 6: the WHERE objective back-propagates through patch_context into cross-attention
    under the multiplicative interaction gate (selection stays trainable end-to-end)."""
    from models.recursive_reasoning.trm_fvr_v2 import TestConditionedC2

    torch.manual_seed(67)
    c2 = TestConditionedC2(_p3a_c2_config(
        c2_isolated_relmap_query=True, c2_support_interaction_gate=True)).train()
    target, ci, co, cin, cout, cm = _p3a_support_tensors()
    _out, m, _ = c2(target, ci, co, cin, cout, cm, target_query_features=target)
    where_proxy = m["c2_gate_where_values"].square().mean()   # any WHERE objective shape works
    grads = torch.autograd.grad(
        where_proxy,
        (c2.cross_attn.q_proj.weight, c2.cross_attn.o_proj.weight, c2.gate_patch_token.weight),
        allow_unused=True)
    for name, g in zip(("cross_attn.q_proj", "cross_attn.o_proj", "gate_patch_token"), grads):
        assert g is not None and int(torch.count_nonzero(g)) > 0, (
            f"WHERE loss no longer reaches {name} under the interaction gate")


def test_p3a_transport_loss_cannot_train_selector_under_interaction_gate() -> None:
    """P3A test 7: with c2_gate_selector_detach (mandatory under the interaction gate), transport
    losses reach the update strengths and content projections but never the selector tensors."""
    from models.recursive_reasoning.trm_fvr_v2 import TestConditionedC2

    torch.manual_seed(71)
    c2 = TestConditionedC2(_p3a_c2_config(
        c2_isolated_relmap_query=True, c2_support_interaction_gate=True)).train()
    with torch.no_grad():
        c2.gate_patch.fill_(0.7)
        c2.gate_global.fill_(0.5)
        c2.gate_patch_token.weight.normal_(mean=0.0, std=0.05)
    target, ci, co, cin, cout, cm = _p3a_support_tensors()
    prior = torch.rand(1, 16, 2)   # engages where_gate_weights so its detach test is non-vacuous
    out, _m, _ = c2(target, ci, co, cin, cout, cm,
                    target_where_hint=prior, target_query_features=target)
    sel_w, hint_w, strength = torch.autograd.grad(
        out.square().mean(),
        (c2.gate_patch_token.weight, c2.where_gate_weights, c2.gate_patch),
        allow_unused=True)
    assert sel_w is None or int(torch.count_nonzero(sel_w)) == 0, (
        "transport loss leaked into gate_patch_token")
    assert hint_w is None or int(torch.count_nonzero(hint_w)) == 0, (
        "transport loss leaked into where_gate_weights")
    assert strength is not None and float(strength.abs()) > 0.0, (
        "detach also severed the trainable update strength")


def test_p3a_counterfactual_metrics_share_one_row_set() -> None:
    """P3A test 8: correct/shuffled/zero are scored on the intersection row set only -- an excluded
    row with a PERFECT correct-F1 must not lift any aggregate, and per-ex entries outside the set
    (or with undefined denominators) are NaN."""
    import math

    import models.losses_fvr as losses

    t = torch.tensor([[5, 5, 3, 0],       # row 0: cells 0,1 changed; cell 2 copy; cell 3 invalid
                      [3, 3, 3, 3],       # row 1: copy-only (F1 undefined, FPR defined)
                      [5, 5, 5, 5]])      # row 2: all changed, EXCLUDED from the shared set
    inp = torch.tensor([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]])
    correct = torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 0.], [1., 1., 1., 1.]])
    shuffle = torch.zeros(3, 4)
    zero = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
    rows = torch.tensor([True, True, False])
    m = losses._where_counterfactual_metrics(correct, shuffle, zero, t, inp, rows)
    assert abs(float(m["where_correct_macro_f1"]) - 2.0 / 3.0) < 1e-6, (
        "the excluded perfect row leaked into the correct macro F1")
    assert float(m["where_shuffle_macro_f1"]) == 0.0
    assert float(m["where_zero_macro_f1"]) == 0.0
    assert abs(float(m["where_correct_minus_zero_f1"]) - 2.0 / 3.0) < 1e-6
    assert float(m["where_shared_changed_rows"]) == 1.0
    f1 = m["where_correct_f1_per_ex"]
    assert abs(float(f1[0]) - 2.0 / 3.0) < 1e-6
    assert math.isnan(float(f1[1])), "copy-only row must have NaN F1"
    assert math.isnan(float(f1[2])), "excluded row must have NaN F1"
    assert math.isnan(float(m["where_shuffle_f1_per_ex"][2]))
    assert math.isnan(float(m["where_zero_f1_per_ex"][2]))
    fpr = m["where_correct_fpr_per_ex"]
    assert float(fpr[0]) == 0.0 and float(fpr[1]) == 0.0 and math.isnan(float(fpr[2]))


def test_p3a_gradient_probe_never_touches_parameters() -> None:
    """P3A test 9: the probe core takes gradient norms without populating .grad or moving any
    parameter, and its recommendation puts the weighted ratio exactly on target for stable scales."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    torch.manual_seed(73)
    lin = torch.nn.Linear(4, 1, bias=False)
    before = lin.weight.detach().clone()
    x = torch.randn(8, 4)

    def step_fn(i):
        y = lin(x * float(i + 1))
        return y.square().mean(), (0.1 * y).square().mean()

    result = T._gradient_probe_core(step_fn, [lin.weight], steps=3, target_ratio=0.5)
    assert torch.equal(lin.weight.detach(), before), "the probe moved a parameter"
    assert lin.weight.grad is None, "the probe populated .grad (an optimizer could consume it)"
    # L_support = 0.01 * L_where pointwise -> G_support = 0.01 * G_where at every step, so the
    # recommended lambda must land the achieved ratio exactly on the 0.5 target.
    assert abs(result["recommended_lambda_support"] - 50.0) < 1e-3
    assert abs(result["achieved_weighted_ratio"] - 0.5) < 1e-6
    assert result["pass"] is True
    assert len(result["g_where_per_step"]) == 3

    # Live-caught edge: the support contrast's empty-set contract returns a DETACHED zero when a
    # batch has no matched rows -- the probe must record G_support=0 for that step, not crash.
    def step_fn_detached(i):
        y = lin(x)
        return y.square().mean(), torch.zeros(())

    result2 = T._gradient_probe_core(step_fn_detached, [lin.weight], steps=2, target_ratio=0.5)
    assert result2["g_support_per_step"] == [0.0, 0.0]
    assert torch.equal(lin.weight.detach(), before)


def test_p3a_where_selector_scope_excludes_forbidden_parameters() -> None:
    """P3A test 10: the v3-where-selector scope selects exactly the selector surface and its
    forbidden-tensor gate trips on anything that should stay frozen."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_run_v2", str(ROOT / "scripts" / "test_run_v2.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    p = torch.nn.Parameter(torch.zeros(1))
    allowed = [
        "model.inner.relmap_proj.weight",
        "model.inner.c2_demo_relmap_proj.weight",
        "model.inner.c2.demo_proj.weight",
        "model.inner.c2.demo_scalar_proj.weight",
        "model.inner.c2.demo_mix.weight",
        "model.inner.c2.pair_proj.weight",
        "model.inner.c2.pair_mix.weight",
        "model.inner.c2.cross_attn.q_proj.weight",
        "model.inner.c2.patch_proj.weight",
        "model.inner.c2.global_proj.weight",
        "model.inner.c2.gate_patch_token.weight",
        "model.inner.c2.gate_patch_token.bias",
        "model.inner.c2.where_gate_weights",
    ]
    frozen = [
        "model.inner.lm_head.weight",
        "model.inner.L_level.layers.0.mlp.weight",
        "model.inner.color_head.weight",
        "model.inner.color_evidence_proj.weight",
        "model.inner.structure_head.weight",
        "model.inner.structure_relmap_proj.weight",
        "model.inner.c2.gate_patch",
        "model.inner.c2.gate_global",
        "model.inner.pid_task_modulator.weight",
        "model.inner.quarantine_head.0.weight",
        "model.inner.pairdelta_input_encoder.proj.weight",
        "model.inner.delta_rule_input_proj.weight",
        "model.inner.rule_factor_proj.weight",
        "puzzle_emb.weights",
    ]
    named = [(n, p) for n in allowed + frozen]
    params, names, bad = T.select_where_selector_params(named)
    assert sorted(names) == sorted(allowed), (
        f"scope drift: selected {sorted(set(names) ^ set(allowed))} unexpectedly")
    assert not bad, f"clean surface flagged as forbidden: {bad}"
    # Poison: a tensor whose name matches BOTH an allowed pattern and a forbidden one must trip.
    _params, _names, bad2 = T.select_where_selector_params(
        named + [("model.inner.c2.demo_proj_color_head.weight", p)])
    assert bad2, "the forbidden-tensor gate failed to trip on a poisoned selection"


def main() -> None:
    tests = [
        test_sample_batch_uses_supplied_rng_only,
        test_explicit_lodo_contract_runs_in_eval_and_overrides_rng,
        test_runner_freezes_lodo_contract_with_local_rng,
        test_lodo_contract_rejects_non_integer_holdout_indices,
        test_evaluation_mode_restores_training_state,
        test_seeded_evaluation_restores_rng_and_repeats_exactly,
        test_runner_builds_independent_rank0_eval_loader,
        test_collect_eval_batches_restores_all_global_rng_state,
        test_injection_scale_zero_removes_target_relmaps_in_both_flows,
        test_injection_scale_zero_removes_pid_task_modulation,
        test_injection_scale_zero_removes_visual_adapter_residual,
        test_source_and_lodo_manifests_are_content_addressed,
        test_failure_artifact_records_rank_stage_batch_and_rng,
        test_nonfinite_gradient_detection_happens_before_allreduce,
        test_where_metrics_separate_macro_micro_and_support_contracts,
        test_where_support_contrast_is_finite_with_no_matched_rows,
        test_positive_where_loss_trains_copy_only_rows_closed,
        test_evidence_schema_fingerprint_is_order_and_semantics_sensitive,
        test_legacy_schema_migration_resets_semantic_consumers,
        test_requested_auxiliary_losses_fail_when_evidence_is_missing,
        test_hierarchical_context_keys_are_collision_free_and_touch_sensitive,
        test_hierarchical_value_binding_uses_context_and_copy_backoff,
        test_hierarchical_value_binding_separates_changed_and_copy_support,
        test_positive_where_gate_is_nonnegative_and_gates_global_context,
        test_ordered_flow_implies_support_relmap_projection,
        test_positive_where_gate_rejects_nonzero_strength_init,
        test_where_selector_detach_preserves_forward_values,
        test_where_selector_detach_separates_transport_and_where_gradients,
        test_canonical_binder_routes_pure_recolour_and_stays_zero_init,
        test_rule_factors_detect_slide_plus_recolour_as_two_operations,
        test_rule_colour_factor_detects_background_recolour,
        test_object_correspondence_ignores_colour_but_preserves_shape,
        test_pairdelta_can_keep_identity_demos_as_negative_evidence,
        test_pairdelta_spatial_features_separate_translation_from_recolour,
        test_pairdelta_spatial_branch_is_zero_init_noop,
        test_extent_route_preserves_floor_or_allows_candidate,
        test_bind_per_ex_undefined_subsets_are_nan,
        test_canonical_bind_residual_loss_trains_delivery_not_standalone_logits,
        test_nonfinite_diagnostics_name_component_and_tensor_failure,
        test_fixed_eval_failure_identity_includes_batch_seed_and_pids,
        test_where_per_task_exposes_undefined_denominator_masks,
        test_mechanism_panel_requires_finite_bind_exactness,
        test_paired_report_guards_family_mismatch_and_bootstrap_samples,
        test_task_metrics_finalize_is_nonfatal_and_post_checkpoint,
        test_fusion_compression_math_and_patience,
        test_p1_changed_backoff_survives_deeper_copy_only_context,
        test_p1_copy_backoff_survives_deeper_changed_only_context,
        test_p1_context_moves_posterior_but_not_marginal,
        test_p1_route_zero_does_not_scale_or_zero_distribution,
        test_p1_reliability_and_route_are_separate_tensors,
        test_p1_distributions_finite_and_normalized,
        test_p1_invalid_cells_stay_all_zero,
        test_p1_fixed_replacement_claim_is_support_only,
        test_p1_empty_subsets_are_nan,
        test_p1_semver_bump_rejects_v3_canonical_checkpoint,
        test_p3a_contract_check_covers_all_valid_cells,
        test_p3a_p1_verdict_unavailable_without_primary_family,
        test_p3a_flag_off_forward_is_unchanged_and_rejects_query,
        test_p3a_isolated_query_moves_gate_not_recurrent_input,
        test_p3a_interaction_gate_needs_both_target_and_support,
        test_p3a_zero_support_outputs_are_finite,
        test_p3a_zero_support_update_is_exactly_zero,
        test_p3a_where_loss_reaches_cross_attention,
        test_p3a_transport_loss_cannot_train_selector_under_interaction_gate,
        test_p3a_counterfactual_metrics_share_one_row_set,
        test_p3a_gradient_probe_never_touches_parameters,
        test_p3a_where_selector_scope_excludes_forbidden_parameters,
    ]
    failures = []
    for test in tests:
        try:
            test()
            print(f"{test.__name__}: PASS")
        except Exception as exc:
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"{test.__name__}: FAIL - {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
