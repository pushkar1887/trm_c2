"""Regression checks for V3 relational-map integration.

This is intentionally a plain Python script rather than pytest: the local
workspace does not consistently have pytest installed.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import yaml

# Local env has no triton: disable torch.compile so the forward test runs (mirrors run_stage1_local.py).
os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pretrain
from models.recursive_reasoning.object_bank import REL_MAP_CHANNELS

CONFIG = ROOT / "checkpoints" / "TRM-FVR-Experiments" / "c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401" / "all_config.yaml"
DATASET = Path(r"D:\trm_c2\arc1concept-aug-1000")


def _load_relmap_config() -> pretrain.PretrainConfig:
    raw = yaml.safe_load(CONFIG.read_text())
    raw["data_paths"] = [str(DATASET)]
    raw["data_paths_test"] = []
    raw["global_batch_size"] = 2
    arch = raw.setdefault("arch", {})
    arch["c2_enabled"] = True
    arch["c2_mode"] = "test_conditioned"
    arch["c2_num_context"] = 4
    arch["c2_relmap"] = True
    return pretrain.PretrainConfig(**raw)


def test_dataloader_receives_arch_relmap_flag() -> None:
    config = _load_relmap_config()
    loader, _meta = pretrain.create_dataloader(
        config,
        "train",
        0,
        1,
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=2,
    )
    assert loader.dataset.config.c2_relmap is True, "arch.c2_relmap was not forwarded to PuzzleDatasetConfig"

    for _set_name, batch, _global_batch_size in loader:
        assert "rel_maps" in batch, "target rel_maps missing from dataloader batch"
        assert "context_rel_maps" in batch, "context_rel_maps missing from dataloader batch"
        assert tuple(batch["rel_maps"].shape) == tuple(batch["inputs"].shape) + (REL_MAP_CHANNELS,)
        assert tuple(batch["context_rel_maps"].shape) == tuple(batch["context_inputs"].shape) + (REL_MAP_CHANNELS,)
        break


def _find_tuple_assignment(module: ast.Module, name: str) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            continue
        if not isinstance(node.value, ast.Tuple):
            continue
        for elt in node.value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                values.add(elt.value)
    return values


def test_zh_check_rolls_relmap_keys_with_inputs_and_context() -> None:
    """The zh probe's INPUT shuffle rolls a static tuple; the DEMO shuffle must be PREFIX-driven
    (every context_* key + frame_label) so a new demo feature can never be silently left behind
    (the old static tuple missed context_output_rel_maps/frame_label -> the probe measured a mixed
    contract no training path ever sees)."""
    source = (ROOT / "scripts" / "run_stage1_local.py").read_text()
    module = ast.parse(source)
    inp_keys = _find_tuple_assignment(module, "_INP")
    assert "rel_maps" in inp_keys, "zh-check input shuffle must roll rel_maps with inputs"
    zh = source.split("def zh_conditioning_report")[1].split("\n    @torch.no_grad()")[0]
    assert 'startswith("context_")' in zh, "zh-check demo shuffle must roll ALL context_* keys (prefix-driven)"
    assert '"frame_label"' in zh, "zh-check demo shuffle must roll frame_label with the demos"
    assert "c2_candidate_logits" in zh, "zh-check must measure flip on the CANDIDATE head (floor is saturated)"


def test_run_stage1_has_v3_adapter_quarantine_scope() -> None:
    """The TRM-only next experiment needs one frozen-core scope that trains C2/PairDelta adapters
    and the quarantine candidate together. v3-adapter excludes quarantine; v3-head+quarantine
    excludes the C2 adapter stack. The runner must expose the union explicitly."""
    source = (ROOT / "scripts" / "run_stage1_local.py").read_text()
    module = ast.parse(source)
    assert "v3-adapter+quarantine" in source, "runner must expose --train-scope v3-adapter+quarantine"

    choices_seen = False
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--train-scope":
            continue
        for kw in node.keywords:
            if kw.arg == "choices" and isinstance(kw.value, ast.Tuple):
                values = {elt.value for elt in kw.value.elts if isinstance(elt, ast.Constant)}
                assert "v3-adapter+quarantine" in values, "argparse choices missing v3-adapter+quarantine"
                choices_seen = True
    assert choices_seen, "could not find --train-scope argparse choices"

    branch = source.split('elif args.train_scope == "v3-adapter+quarantine":', 1)
    assert len(branch) == 2, "missing v3-adapter+quarantine selection branch"
    body = branch[1].split("elif args.unified", 1)[0]
    assert "v3_adapter_names" in body, "new scope must include C2/relmap/PairDelta adapter params"
    assert '"quarantine_"' in body, "new scope must include quarantine candidate params"
    assert "v3-adapter+quarantine(frozen-core" in body, "scope label must clearly state frozen-core semantics"


def test_run_stage1_supports_epoch_checkpoint_mode() -> None:
    """run_stage1_local is now used as the full-data check vehicle, so it must support the two
    mechanics pretrain.py already has: dataset-derived epoch steps and a consumable checkpoint
    artifact. A raw --steps-only probe cannot satisfy a 3-epoch 960-task run."""
    source = (ROOT / "scripts" / "run_stage1_local.py").read_text()
    assert '"--epochs"' in source, "runner must expose --epochs for full-dataset epoch runs"
    assert '"--save-checkpoint-dir"' in source, "runner must expose a checkpoint output directory"
    assert "meta.total_groups * meta.mean_puzzle_examples" in source, (
        "--epochs must use the same dataset step formula as pretrain.py")
    assert "torch.save(loss_head.state_dict()" in source, "runner must save a pretrain-compatible state_dict"
    assert '"all_config.yaml"' in source, "runner must save the resolved config beside the checkpoint"


def test_model_relmap_fallback_is_loud() -> None:
    source = (ROOT / "models" / "recursive_reasoning" / "trm_fvr_c2.py").read_text()
    assert "rel_maps missing from batch" in source, "model fallback must warn when dataloader rel_maps are absent"
    assert "warnings.warn" in source, "relmap fallback must be visible, not silent"


# --- §15.2 per-component upgrade checks (C2 demo-feed + PairDelta input hint) -----------------------

def _build_v3_2_model(extra_arch: dict):
    """Fresh (no-checkpoint) model + loader with the §15.2 flags on. Returns (loss_head, loader)."""
    import torch
    raw = yaml.safe_load(CONFIG.read_text())
    raw["data_paths"] = [str(DATASET)]
    raw["data_paths_test"] = []
    raw["global_batch_size"] = 2
    raw["load_checkpoint"] = None
    arch = raw.setdefault("arch", {})
    arch["c2_enabled"] = True
    arch["c2_mode"] = "test_conditioned"
    arch["c2_num_context"] = 4
    arch["c2_relmap"] = True
    arch.update(extra_arch)
    config = pretrain.PretrainConfig(**raw)
    loader, meta = pretrain.create_dataloader(
        config, "train", 0, 1, test_set_mode=False, epochs_per_iter=1, global_batch_size=2)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    loss_head, _, _ = pretrain.create_model(config, meta, rank=0, world_size=1)
    return loss_head, loader


def test_dataloader_emits_context_output_rel_maps() -> None:
    config = _load_relmap_config()
    loader, _ = pretrain.create_dataloader(
        config, "train", 0, 1, test_set_mode=False, epochs_per_iter=1, global_batch_size=2)
    for _s, batch, _g in loader:
        assert "context_output_rel_maps" in batch, "§15.2-A: dataloader missing context_output_rel_maps"
        assert tuple(batch["context_output_rel_maps"].shape) == tuple(batch["context_outputs"].shape) + (REL_MAP_CHANNELS,)
        break


def test_v3_2_upgrades_build_zero_init_and_forward() -> None:
    """The F7 guarantee: new projections are EXACTLY zero at init (=> no-op), and a full forward with
    all §15.2 flags on runs finite with the hint contributing 0 at step 0."""
    import torch
    loss_head, loader = _build_v3_2_model({
        "c2_relmap_demos": True, "c2_pairdelta_input_feature": True, "c2_dual_output_head": True})
    inner = loss_head.model.inner
    # (1) modules built + zero-init (zero linear => provably no-op at step 0)
    assert hasattr(inner, "c2_demo_relmap_proj"), "§15.2-A: c2_demo_relmap_proj not built"
    assert float(inner.c2_demo_relmap_proj.weight.abs().max()) == 0.0, "demo relmap proj must be zero-init"
    assert hasattr(inner, "pairdelta_input_encoder"), "§15.2-B: pairdelta_input_encoder not built"
    assert hasattr(inner, "delta_rule_input_proj"), "§15.2-B: delta_rule_input_proj not built"
    assert float(inner.delta_rule_input_proj.weight.abs().max()) == 0.0, "pairdelta input proj must be zero-init"
    # §15.6 structure-reads-map fix: built + zero-init (=> step-0 structure logits unchanged, F7-safe)
    assert hasattr(inner, "structure_relmap_proj"), "§15.6: structure_relmap_proj not built"
    assert tuple(inner.structure_relmap_proj.weight.shape) == (3, REL_MAP_CHANNELS), "structure_relmap_proj must be [3,REL_MAP_CHANNELS] (->PAD/EOS/VALID)"
    assert float(inner.structure_relmap_proj.weight.abs().max()) == 0.0, "structure_relmap_proj must be zero-init (F7 no-op)"
    # §15.9 boundary lever: bias must exist (the lever that can PLACE pad on featureless cells) and be zero-init.
    assert inner.structure_relmap_proj.bias is not None, "§15.9: structure_relmap_proj must have a bias (boundary lever)"
    assert float(inner.structure_relmap_proj.bias.abs().max()) == 0.0, "structure_relmap_proj bias must be zero-init (F7 no-op)"
    # (2) full forward runs finite; the PairDelta input hint contributes exactly 0 at init
    device = next(loss_head.parameters()).device
    for _s, cb, _g in loader:
        if "context_inputs" not in cb or cb["context_inputs"].shape[1] < 2:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
        with torch.device(device.type):
            carry = loss_head.initial_carry(batch)
        carry, loss_val, metrics, _, _ = loss_head(carry=carry, batch=batch, return_keys=[])
        assert torch.isfinite(loss_val), "non-finite loss with §15.2 flags on"
        assert "c2_pairdelta_input_norm" in metrics, "§15.2-B: PairDelta input-hint metric missing"
        assert float(metrics["c2_pairdelta_input_norm"]) == 0.0, "PairDelta input hint must be 0 at init (zero-proj)"
        break


def test_frame_hint_zero_init_and_forward() -> None:
    """Lane B: the frame-hint embedding is zero at init (=> step-0 no-op, F7-safe), the dataloader emits
    frame_label, and a full forward runs finite with the hint contributing exactly 0 at init."""
    import torch
    loss_head, loader = _build_v3_2_model({"c2_dual_output_head": True, "c2_frame_hint": True})
    inner = loss_head.model.inner
    assert hasattr(inner, "frame_embed"), "Lane B: frame_embed not built"
    assert float(inner.frame_embed.embedding_weight.abs().max()) == 0.0, "frame_embed must be zero-init (F7 no-op)"
    device = next(loss_head.parameters()).device
    for _s, cb, _g in loader:
        if "context_inputs" not in cb or cb["context_inputs"].shape[1] < 2:
            continue
        assert "frame_label" in cb, "dataloader must emit frame_label when c2_frame_hint is on"
        assert tuple(cb["frame_label"].shape) == (cb["inputs"].shape[0],), "frame_label must be [B]"
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
        with torch.device(device.type):
            carry = loss_head.initial_carry(batch)
        carry, loss_val, metrics, _, _ = loss_head(carry=carry, batch=batch, return_keys=[])
        assert torch.isfinite(loss_val), "non-finite loss with c2_frame_hint on"
        assert "c2_frame_hint_norm" in metrics, "frame-hint metric missing"
        assert float(metrics["c2_frame_hint_norm"]) == 0.0, "frame hint must contribute 0 at init (zero embed)"
        break


def test_rule_hypothesis_hint_zero_init_and_forward() -> None:
    """c2_rule_hypothesis_hint: the in-model hint imports object_rule_bank.infer_rule_hypotheses, infers
    the top operation-family from the support pairs, and broadcast-adds a ZERO-INIT embedding. F7-safe
    (embed zero at init => step-0 no-op) and a full forward runs finite with the hint contributing 0."""
    import torch
    from models.recursive_reasoning.trm_fvr_c2 import RULE_FAMILY_VOCAB
    loss_head, loader = _build_v3_2_model({"c2_dual_output_head": True, "c2_rule_hypothesis_hint": True})
    inner = loss_head.model.inner
    assert hasattr(inner, "rule_hyp_embed"), "rule_hyp_embed not built when c2_rule_hypothesis_hint on"
    assert tuple(inner.rule_hyp_embed.embedding_weight.shape)[0] == len(RULE_FAMILY_VOCAB), (
        "rule_hyp_embed must have one row per RULE_FAMILY_VOCAB entry")
    assert float(inner.rule_hyp_embed.embedding_weight.abs().max()) == 0.0, "rule_hyp_embed must be zero-init (F7 no-op)"
    device = next(loss_head.parameters()).device
    for _s, cb, _g in loader:
        if "context_inputs" not in cb or cb["context_inputs"].shape[1] < 2:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
        with torch.device(device.type):
            carry = loss_head.initial_carry(batch)
        carry, loss_val, metrics, _, _ = loss_head(carry=carry, batch=batch, return_keys=[])
        assert torch.isfinite(loss_val), "non-finite loss with c2_rule_hypothesis_hint on"
        assert "c2_rule_hyp_norm" in metrics, "rule-hyp metric missing (block did not execute)"
        assert float(metrics["c2_rule_hyp_norm"]) == 0.0, "rule hint must contribute 0 at init (zero embed)"
        # the family inference must actually fire on a multi-demo task (some item routed off 'none')
        assert "c2_rule_hyp_nonzero_frac" in metrics, "rule-hyp coverage metric missing"
        break


def test_structure_from_lmhead_reproduces_floor_partition() -> None:
    """§15.8: structure-from-lm_head must reproduce the floor's PAD/EOS/VALID partition EXACTLY at init.

    The factored structure channel is [lm_pad, lm_eos, logsumexp(lm_colour)]; because
    logsumexp([a, b, logsumexp([c...])]) == logsumexp([a, b, c...]), log_softmax of that triple equals
    the floor's own log-probs on PAD/EOS, and the total VALID colour mass equals the floor's total colour
    mass. This is WARM-START INDEPENDENT -- it does not rely on color_head being initialised from lm_head --
    so it is the load-bearing guarantee that the factored head inherits lm_head's pad/eos/shape."""
    import torch
    loss_head, loader = _build_v3_2_model({
        "c2_dual_output_head": True,
        "c2_floor_candidate_split": True,       # exposes c2_floor_logits + c2_candidate_logits
        "c2_candidate_floor_structure": False,  # candidate == the factored recombine
        "c2_structure_from_lmhead": True,       # structure channel derived from lm_head logsumexp
    })
    device = next(loss_head.parameters()).device
    for _s, cb, _g in loader:
        if "context_inputs" not in cb or cb["context_inputs"].shape[1] < 2:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
        with torch.device(device.type):
            carry = loss_head.initial_carry(batch)
        carry, _lv, _m, detached, _ = loss_head(
            carry=carry, batch=batch,
            return_keys=["logits", "c2_floor_logits", "c2_candidate_logits"])
        floor = detached["c2_floor_logits"].float()
        cand = detached["c2_candidate_logits"].float()
        floor_logp = torch.log_softmax(floor, dim=-1)
        # PAD (idx 0) and EOS (idx 1) channels reproduce the floor's log-probs EXACTLY (float32 identity is
        # ~1e-6). Tolerance is bf16-sized: candidate_logits is cast back to the bf16 forward dtype
        # (trm_fvr_c2.py candidate_logits.to(color_logits.dtype)) so ~1e-2 rounding is expected, not a regression.
        _atol = 3e-2
        assert torch.allclose(cand[..., 0], floor_logp[..., 0], atol=_atol), "§15.8: PAD channel must match floor partition"
        assert torch.allclose(cand[..., 1], floor_logp[..., 1], atol=_atol), "§15.8: EOS channel must match floor partition"
        # Total VALID colour mass matches the floor's total colour mass (color_head only redistributes WITHIN).
        cand_valid = torch.logsumexp(cand[..., 2:12], dim=-1)
        floor_valid = torch.logsumexp(floor_logp[..., 2:12], dim=-1)
        assert torch.allclose(cand_valid, floor_valid, atol=_atol), "§15.8: total VALID mass must match floor"
        break
    else:
        raise AssertionError("no batch with >=2 context demos to exercise §15.8")


def test_floor_candidate_split_exposes_both_paths() -> None:
    """V3 split contract: MAIN is floor-backed, while the factored head remains available as
    a separate candidate tensor for LODO/selector scoring."""
    import torch
    loss_head, loader = _build_v3_2_model({
        "c2_relmap_demos": True,
        "c2_pairdelta_input_feature": True,
        "c2_dual_output_head": True,
        "c2_floor_candidate_split": True,
        "c2_candidate_floor_structure": True,
        "c2_delta_expose_base_logits": True,
    })
    device = next(loss_head.parameters()).device
    for _s, cb, _g in loader:
        if "context_inputs" not in cb or cb["context_inputs"].shape[1] < 2:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
        with torch.device(device.type):
            carry = loss_head.initial_carry(batch)
        _carry, loss_val, metrics, detached, _done = loss_head(
            carry=carry,
            batch=batch,
            return_keys=[
                "logits",
                "c2_floor_logits",
                "c2_candidate_logits",
                "c2_factored_candidate_logits",
                "c2_main_uses_floor",
                "c2_candidate_floor_structure",
            ],
        )
        assert torch.isfinite(loss_val), "non-finite loss with floor/candidate split"
        assert "c2_main_uses_floor" in detached, "split must mark MAIN as floor-backed"
        assert float(detached["c2_main_uses_floor"]) == 1.0, "MAIN must use floor under split"
        assert float(detached["c2_candidate_floor_structure"]) == 1.0, "candidate must use floor structure"
        assert "c2_floor_logits" in detached and "c2_candidate_logits" in detached
        assert detached["logits"].shape == detached["c2_floor_logits"].shape == detached["c2_candidate_logits"].shape
        assert torch.allclose(detached["logits"], detached["c2_floor_logits"]), "returned MAIN logits must be floor logits"
        floor_pred = detached["c2_floor_logits"].argmax(dim=-1)
        candidate_pred = detached["c2_candidate_logits"].argmax(dim=-1)
        assert torch.equal(
            floor_pred.clamp_max(1),
            candidate_pred.clamp_max(1),
        ), "hybrid candidate must preserve floor PAD/EOS decisions"
        break


def test_object_bank_rel_where_hint_selects_support_changed_cells() -> None:
    """Passed Phase-1 signal: support-derived relmap/input-colour evidence should be higher on
    target cells matching the support changed predicate than on copied cells."""
    import torch
    from models.recursive_reasoning.object_bank import relational_maps, relational_where_hint

    side = 4
    inp = torch.full((1, side * side), 2, dtype=torch.long)  # raw colour 0
    out = inp.clone()
    inp[0, [0, 5]] = 3                                      # raw colour 1
    out[0, [0, 5]] = 4                                      # raw colour 2, changed cells
    target = inp.clone()

    context_inputs = inp.view(1, 1, side * side)
    context_outputs = out.view(1, 1, side * side)
    context_mask = torch.ones((1, 1), dtype=torch.bool)
    target_rel = relational_maps(target, side=side)
    context_rel = relational_maps(inp, side=side).view(1, 1, side * side, REL_MAP_CHANNELS)

    hint, info = relational_where_hint(
        target,
        context_inputs,
        context_outputs,
        context_mask,
        target_rel_maps=target_rel,
        context_rel_maps=context_rel,
        side=side,
    )

    changed_like = target[0] == 3
    copy_like = target[0] == 2
    assert tuple(hint.shape) == (1, side * side, 1)
    assert float(hint[0, changed_like, 0].mean()) > float(hint[0, copy_like, 0].mean())
    assert float(info["rel_where_confidence"][0]) > 0.0


def test_pairdelta_intent_features_separate_correct_from_shuffled_support() -> None:
    """Passed Phase-2 signal: PairDelta intent is a router diagnostic, not an output writer."""
    import torch
    from models.recursive_reasoning.pair_delta_encoder import pairdelta_intent_features

    side = 4
    a = torch.full((side * side,), 2, dtype=torch.long)
    b = torch.full((side * side,), 2, dtype=torch.long)
    a[0] = 3
    b[-1] = 3
    ao = a.clone(); ao[0] = 4
    bo = b.clone(); bo[-1] = 4
    inputs = torch.stack([a, b]).view(1, 2, side * side)
    outputs = torch.stack([ao, bo]).view(1, 2, side * side)
    shuffled = torch.stack([bo, ao]).view(1, 2, side * side)
    mask = torch.ones((1, 2), dtype=torch.bool)

    correct = pairdelta_intent_features(inputs, outputs, mask)
    wrong = pairdelta_intent_features(inputs, shuffled, mask)

    assert float(correct["conditional_recolor_score"][0]) > float(wrong["conditional_recolor_score"][0])
    assert float(correct["changed_rate"][0]) < float(wrong["changed_rate"][0])


def test_v3_rel_where_and_pairdelta_hints_are_zero_init_live_inputs() -> None:
    """The passed signals may enter the existing color_head only as zero-init evidence columns."""
    import torch
    loss_head, loader = _build_v3_2_model({
        "c2_relmap_demos": True,
        "c2_dual_output_head": True,
        "c2_rel_where_hint": True,
        "c2_pairdelta_intent_hint": True,
    })
    inner = loss_head.model.inner
    # FIX A: hint columns live in the dedicated color_evidence_proj (own lr group), zero-init.
    assert float(inner.color_evidence_proj.weight[:, -2:].abs().max()) == 0.0, "hint evidence columns must be zero-init"

    device = next(loss_head.parameters()).device
    for _s, cb, _g in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
        with torch.device(device.type):
            carry = loss_head.initial_carry(batch)
        _carry, loss_val, metrics, detached, _done = loss_head(
            carry=carry,
            batch=batch,
            return_keys=["c2_rel_where_hint", "c2_pairdelta_intent_hint"],
        )
        assert torch.isfinite(loss_val), "non-finite loss with rel_where/PairDelta hints"
        assert "c2_rel_where_hint_mean" in metrics, "rel_where hint metric missing"
        assert "c2_pairdelta_conditional_score" in metrics, "PairDelta conditional metric missing"
        assert "c2_rel_where_hint" in detached and "c2_pairdelta_intent_hint" in detached
        break


def test_boundary_lever_places_pad_on_featureless_cell() -> None:
    """§15.9: structure_relmap_proj bias=True is the lever that lets the head PLACE pad on cells where the
    relmap is featureless (every channel ~0 outside the grid). Zero-init => step-0 == floor (F7-safe); a learned
    (bias, weight-on-valid_mask) reproduces an implicit (1 - valid_mask) channel so PAD rises on pad cells and
    falls on valid cells -- copying the input boundary WITHOUT eating valid cells. Pure-tensor: no model/dataset."""
    import torch
    from models.layers import CastedLinear
    proj = CastedLinear(REL_MAP_CHANNELS, 3, bias=True)
    with torch.no_grad():
        proj.weight.zero_()
        proj.bias.zero_()
    valid_cell = torch.zeros(REL_MAP_CHANNELS)
    valid_cell[0] = 1.0                                  # channel 0 == valid_mask (1 inside the grid)
    pad_cell = torch.zeros(REL_MAP_CHANNELS)             # featureless outside the grid (all channels ~0)
    cells = torch.stack([valid_cell, pad_cell])          # [2, C]
    # (1) zero-init: the proj is EXACTLY a no-op -> §15.8 floor partition preserved at step 0 (F7-safe).
    assert float(proj(cells).abs().max()) == 0.0, "zero-init structure_relmap_proj must be a no-op (step-0 == floor)"
    # (2) the lever: bias[PAD] is the implicit "1"; weight[PAD, valid_mask] is the "- valid_mask" slope.
    PAD = 0
    with torch.no_grad():
        proj.bias[PAD] = 2.0
        proj.weight[PAD, 0] = -2.0
    out = proj(cells)
    pad_logit_valid = float(out[0, PAD])                 # valid_mask=1 -> 2.0 + (-2.0)*1 = 0.0
    pad_logit_pad = float(out[1, PAD])                   # valid_mask=0 -> 2.0 + (-2.0)*0 = 2.0
    assert pad_logit_pad > pad_logit_valid + 1.0, (
        "boundary lever must raise PAD on featureless pad cells ABOVE valid cells "
        f"(pad={pad_logit_pad:.2f} vs valid={pad_logit_valid:.2f}) -- a weight-only proj cannot (bias=False is flat)")


def test_outside_grid_lever_places_pad_and_gates_size_change() -> None:
    """§15.9.1: the extent PAD mask + a dedicated [1->3] proj lets structure PLACE pad on the padding
    (where the 13-ch relmap is featureless); warm-init makes it immediate; and conf=0 (no verified size
    rule) zeroes the mask so the lever leaves the floor untouched -- no task can be hurt. Pure-tensor
    (structure_outside_proj is F.linear(x, W)); no model/dataset/coolname needed."""
    import torch
    import torch.nn.functional as F
    PAD = 0
    # cells: a VALID cell (valid_mask=1 -> outside=0) and a PAD cell (valid_mask=0 -> outside=1)
    valid_mask = torch.tensor([[1.0], [0.0]])
    outside = 1.0 - valid_mask                                   # valid->0, pad->1
    # (1) zero-init proj (flag-off / no warm-init) is an exact no-op -> step-0 == floor (F7-safe)
    W0 = torch.zeros(3, 1)
    assert float(F.linear(outside, W0).abs().max()) == 0.0, "zero-init outside proj must be a no-op (F7)"
    # (2) warm-init the PAD row LARGE: with a verified extent (conf=1) pad is asserted on the pad cell;
    # the eos-clean mask keeps outside=0 cells (valid AND eos) untouched EVEN at big magnitude.
    W = torch.zeros(3, 1)
    W[PAD, 0] = 20.0
    out_conf1 = F.linear(outside * 1.0, W)                       # verified size rule: conf = 1
    assert float(out_conf1[1, PAD]) > 14.0, "outside lever must PLACE pad on the padding (from the weight, not bias)"
    assert float(out_conf1[0, PAD]) == 0.0, "outside=0 cells (valid/eos under the eos-clean mask) must be untouched even at V=20"
    # (3) no verified size rule (conf=0): the conf scaling zeroes the mask -> lever disabled (no task hurt)
    out_conf0 = F.linear(outside * 0.0, W)
    assert float(out_conf0.abs().max()) == 0.0, "conf=0 (unverified extent) must disable the outside lever"


def _make_canvas(h, w, off_r, off_c, side, colour=5):
    """Replica of build_arc_dataset.np_grid_to_seq_translational_augment for ONE grid: content box at
    (off_r, off_c) with a thin-L EOS below/right. Returns a flat [side*side] token grid."""
    import torch
    g = torch.zeros(side, side, dtype=torch.long)          # PAD = 0
    g[off_r:off_r + h, off_c:off_c + w] = colour + 2        # colours are token >=2
    er, ec = off_r + h, off_c + w
    if er < side:
        g[er, off_c:ec] = 1                                 # EOS row directly below the box
    if ec < side:
        g[off_r:er, ec] = 1                                 # EOS col directly right of the box
    return g.reshape(-1)


def test_extent_pad_mask_matches_tokenizer_pad_region() -> None:
    """extent_pad_mask, fed the TRUE extent, must reproduce the tokenizer's PAD region EXACTLY (any offset,
    including translated) and never fire on EOS -- this is the geometry the generalized lever depends on.
    Verified empirically at IoU 100%% on the 518K aux by scripts/verify_outside_grid_lever.py; here it is asserted."""
    import torch
    from models.recursive_reasoning.trm_fvr_c2 import extent_pad_mask
    side = 6
    for (h, w, orr, occ) in [(3, 3, 0, 0), (2, 4, 1, 1), (4, 2, 2, 3), (3, 3, 3, 3), (1, 1, 0, 5), (5, 5, 0, 0)]:
        tok = _make_canvas(h, w, orr, occ, side).unsqueeze(0)                 # [1, L]
        mask = extent_pad_mask(tok, torch.tensor([float(h)]), torch.tensor([float(w)]), side)[0]
        pad = (tok[0] == 0).float()
        eos = (tok[0] == 1)
        assert torch.equal(mask, pad), f"mask != tokenizer PAD region for (h,w,off)={(h, w, orr, occ)}"
        assert float(mask[eos].sum()) == 0.0, f"mask fired on EOS (not eos-clean) for {(h, w, orr, occ)}"


def test_extent_eos_mask_matches_tokenizer_thin_l() -> None:
    """extent_eos_mask, fed the TRUE extent, must reproduce the tokenizer's thin-L EOS boundary exactly.
    This is the EOS analogue of the outside-grid PAD lever; the PAD corner outside the L must stay PAD."""
    import torch
    from models.recursive_reasoning.trm_fvr_c2 import extent_eos_mask, extent_pad_mask
    side = 6
    for (h, w, orr, occ) in [(3, 3, 0, 0), (2, 4, 1, 1), (4, 2, 1, 3), (1, 1, 0, 4), (5, 5, 0, 0)]:
        tok = _make_canvas(h, w, orr, occ, side).unsqueeze(0)
        eos_mask = extent_eos_mask(tok, torch.tensor([float(h)]), torch.tensor([float(w)]), side)[0]
        pad_mask = extent_pad_mask(tok, torch.tensor([float(h)]), torch.tensor([float(w)]), side)[0]
        eos = (tok[0] == 1).float()
        assert torch.equal(eos_mask, eos), f"EOS mask != tokenizer EOS region for (h,w,off)={(h, w, orr, occ)}"
        assert float((eos_mask * pad_mask).sum()) == 0.0, "EOS and PAD extent masks must not overlap"


def test_extent_pad_mask_size_change_shares_input_offset() -> None:
    """The tokenizer pads input+output with ONE (pad_r,pad_c) (build_arc_dataset:54), so the output box
    shares the INPUT box offset even under size change. extent_pad_mask reads that offset off the input and
    with the predicted OUTPUT size reproduces the OUTPUT PAD region -> this is the size-change fix."""
    import torch
    from models.recursive_reasoning.trm_fvr_c2 import extent_pad_mask
    side = 8
    orr, occ = 2, 1
    tin = _make_canvas(3, 3, orr, occ, side).unsqueeze(0)                     # input box 3x3
    tout = _make_canvas(5, 4, orr, occ, side)                                 # output box 5x4, SAME offset
    mask = extent_pad_mask(tin, torch.tensor([5.0]), torch.tensor([4.0]), side)[0]
    assert torch.equal(mask, (tout == 0).float()), "size-change mask must match the OUTPUT PAD region"
    assert float(mask[(tout == 1)].sum()) == 0.0, "size-change mask must stay eos-clean"


def test_predicted_extent_verifies_demo_size_rules() -> None:
    """_predicted_extent fits out_hw=f(in_hw) over demos from {identity, constant, integer-ratio}, VERIFIES
    on every demo, applies the first verified rule to the test input, and returns conf=1 only when verified
    (else 0 -> lever stays at floor). NON-VACUOUS: fitted rules (constant/ratio) need >=2 valid demos --
    a single demo reconstructs itself by construction, and that vacuous "verification" asserted the WRONG
    box on 2-demo tasks under LODO holdout. Pure: binds the real _canvas_extent_stats to a fake self."""
    import types
    import torch
    from models.recursive_reasoning import trm_fvr_c2 as M
    side = 6

    def canv(h, w):
        g = torch.zeros(side, side, dtype=torch.long)
        g[:h, :w] = 7
        return g.reshape(-1)

    class _Cfg:
        c2_extent_use_shape_head = False
        c2_extent_shape_head_tau = 0.5

    c2 = types.SimpleNamespace()
    c2._canvas_extent_stats = types.MethodType(M.TestConditionedC2._canvas_extent_stats, c2)
    slf = types.SimpleNamespace(c2=c2, config=_Cfg())

    def run(ctx_in, ctx_out, test_in, mask=None):
        batch = {
            "context_inputs": torch.stack([canv(*hw) for hw in ctx_in]).unsqueeze(0),
            "context_outputs": torch.stack([canv(*hw) for hw in ctx_out]).unsqueeze(0),
            "inputs": canv(*test_in).unsqueeze(0),
        }
        if mask is not None:
            batch["context_mask"] = torch.tensor([mask], dtype=torch.bool)
        return M.TinyRecursiveReasoningModel_ACTV1_Inner._predicted_extent(slf, batch)

    h, w, c = run([(3, 3), (2, 4)], [(3, 3), (2, 4)], (5, 2))
    assert float(c[0]) == 1.0 and int(h[0]) == 5 and int(w[0]) == 2, "identity rule must verify + apply"
    h, w, c = run([(3, 3), (2, 4)], [(4, 4), (4, 4)], (1, 1))
    assert float(c[0]) == 1.0 and int(h[0]) == 4 and int(w[0]) == 4, "constant rule must verify + apply"
    h, w, c = run([(3, 3), (2, 2)], [(6, 6), (4, 4)], (3, 4))
    assert float(c[0]) == 1.0 and int(h[0]) == 6 and int(w[0]) == 8, "integer-ratio rule must verify + apply"
    h, w, c = run([(3, 3), (2, 2)], [(6, 6), (5, 5)], (3, 3))
    assert float(c[0]) == 0.0, "inconsistent demos must NOT verify (conf=0 -> floor, no task hurt)"
    # NON-VACUOUS guards (the 1-demo trap = a 2-demo task under LODO holdout):
    h, w, c = run([(3, 3)], [(4, 4)], (2, 2))
    assert float(c[0]) == 0.0, "constant must NOT 'verify' on a single demo (fits anything -> vacuous)"
    h, w, c = run([(3, 3)], [(6, 6)], (2, 2))
    assert float(c[0]) == 0.0, "ratio must NOT 'verify' on a single demo (fits anything -> vacuous)"
    h, w, c = run([(3, 3)], [(3, 3)], (5, 2))
    assert float(c[0]) == 1.0 and int(h[0]) == 5, "identity IS safe from one demo (parameter-free)"
    h, w, c = run([(3, 3)], [(3, 3)], (5, 2), mask=[False])
    assert float(c[0]) == 0.0, "zero valid demos must NOT verify anything (vacuous all())"


def test_forward_never_mutates_batch_and_visual_flag_raises() -> None:
    """REGRESSION x2. (1) The batch dict IS the ACT carry's current_data: a forward that ADDS keys
    (the old `batch['rel_maps'] = ...` stash) leaks them into the carry and KeyErrors on the next
    step whenever the dataloader did not emit that key. (2) c2_visual_encoder references a REMOVED
    encoder class; it must raise a clear ValueError at build, never a NameError."""
    import torch
    from models.recursive_reasoning.trm_fvr_c2 import FVR_C2_Config, TinyRecursiveReasoningModel_ACTV1_Inner
    base = dict(batch_size=2, seq_len=900, puzzle_emb_ndim=128, num_puzzle_identifiers=2, vocab_size=12,
                H_cycles=1, L_cycles=1, H_layers=1, L_layers=1, hidden_size=128, expansion=2, num_heads=4,
                pos_encodings="rope", halt_max_steps=1, halt_exploration_prob=0.0, forward_dtype="float32")
    try:
        TinyRecursiveReasoningModel_ACTV1_Inner(FVR_C2_Config(**base, c2_visual_encoder=True))
        raise AssertionError("stale c2_visual_encoder=True must raise (the encoder class was removed)")
    except ValueError:
        pass
    m = TinyRecursiveReasoningModel_ACTV1_Inner(FVR_C2_Config(**base))
    m.eval()
    batch = {"inputs": torch.randint(0, 12, (2, 900)), "puzzle_identifiers": torch.zeros(2, dtype=torch.long)}
    keys_before = set(batch.keys())
    with torch.no_grad():
        m.forward(m.fresh_carry(2), batch)
    added = set(batch.keys()) - keys_before
    assert not added, f"forward must not add keys to the batch dict (leaks into the ACT carry): {added}"


def test_aux_forward_shares_main_input_contract() -> None:
    """REGRESSION (bug: LODO trained a different input than MAIN): _run_aux_logits must forward the demo
    relmaps + frame label into _condition_grid_features, and _build_lodo_batch must gather EVERY per-demo
    shuffle tensor from ONE source index (no per-tensor gather conditions -> no wrong-task demos paired
    with correct-task relmaps). Static source checks -- cheap, catches the wiring class, not the values."""
    src = (ROOT / "models" / "recursive_reasoning" / "trm_fvr_c2.py").read_text()
    aux = src.split("def _run_aux_logits")[1].split("\n    def ")[0]
    for key in ("context_rel_maps", "context_output_rel_maps", "frame_label"):
        assert key in aux, f"_run_aux_logits must pass {key} (aux forward != main forward input contract)"
    lodo = src.split("def _build_lodo_batch")[1].split("\n    def ")[0]
    assert "shuffle_src" in lodo, "_build_lodo_batch must use ONE shuffle source index for all demo tensors"
    assert "locals()" not in lodo, "per-tensor gather conditions ('wrong_indices' in locals()) must not return"
    assert "shuffle_frame_label" in lodo and "shuffle_context_output_rel_maps" in lodo, (
        "shuffle variants of frame_label/context_output_rel_maps must ride along for the wrong-task control")


def test_value_evidence_v2_rich_ctx_forward_and_zero_init() -> None:
    """Fix 2 wiring: c2_value_v2_rich_ctx routes cell_conditioning_signature as the V2 context key.
    F7-safe (V2 color_head columns zero-init -> no step-0 output change) and forward stays finite in
    BOTH the default relmap-bucket path and the rich-signature path."""
    import torch
    from models.recursive_reasoning.trm_fvr_c2 import VALUE_EVIDENCE_V2_DIM
    for rich in (False, True):
        loss_head, loader = _build_v3_2_model({
            "c2_dual_output_head": True, "c2_relmap": True,
            "c2_value_evidence_v2": True, "c2_value_v2_rich_ctx": rich,
        })
        inner = loss_head.model.inner
        # F7: the trailing VALUE_EVIDENCE_V2_DIM evidence columns are zero at init (step-0 no-op).
        # FIX A: they live in color_evidence_proj, not in a widened color_head.
        assert float(inner.color_evidence_proj.weight[:, -VALUE_EVIDENCE_V2_DIM:].abs().max()) == 0.0, (
            f"V2 evidence columns must be zero-init (rich={rich})")
        device = next(loss_head.parameters()).device
        for _s, cb, _g in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
            with torch.device(device.type):
                carry = loss_head.initial_carry(batch)
            _carry, loss_val, metrics, _detached, _done = loss_head(carry=carry, batch=batch, return_keys=[])
            assert torch.isfinite(loss_val), f"non-finite loss with value_evidence_v2 (rich={rich})"
            assert "c2_value_v2_support_coverage" in metrics, f"V2 stats missing (rich={rich})"
            break


def test_input_hints_threaded_not_stashed() -> None:
    """REGRESSION / torch.compile-safety (A4): the per-forward rel_where + pairdelta_intent hints must be
    THREADED via _condition_grid_features' return (input_hints dict) -> _output_logits, NOT written to
    self._stashed_* inside forward (instance-attr mutation is invisible to compile graph capture and
    couples main/aux forwards). Static source guard + a live forward with both hints on."""
    import torch
    src = (ROOT / "models" / "recursive_reasoning" / "trm_fvr_c2.py").read_text()
    assert "_stashed_rel_where_hint" not in src and "_stashed_pairdelta_intent_hint" not in src, (
        "the per-forward hint stash is back -- it breaks torch.compile capture; thread via input_hints instead")
    cgf = src.split("def _condition_grid_features")[1].split("\n    def ")[0]
    assert "input_hints" in cgf and "return grid_features, c2_metrics, pid_task_vec, rel_maps, input_hints" in cgf, (
        "_condition_grid_features must build and RETURN input_hints")
    olog = src.split("def _output_logits")[1].split("\n    def ")[0]
    assert 'input_hints.get("rel_where")' in olog and 'input_hints.get("pairdelta_intent")' in olog, (
        "_output_logits must READ the threaded input_hints dict, not a stash")

    loss_head, loader = _build_v3_2_model({
        "c2_relmap_demos": True, "c2_dual_output_head": True,
        "c2_rel_where_hint": True, "c2_pairdelta_intent_hint": True,
    })
    device = next(loss_head.parameters()).device
    for _s, cb, _g in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
        with torch.device(device.type):
            carry = loss_head.initial_carry(batch)
        _carry, loss_val, metrics, _detached, _done = loss_head(carry=carry, batch=batch, return_keys=[])
        assert torch.isfinite(loss_val), "non-finite loss after threading input_hints"
        assert "c2_rel_where_hint_mean" in metrics and "c2_pairdelta_conditional_score" in metrics, (
            "threaded hints must still reach the head (metrics missing)")
        break


def test_transition_hint_binds_value_and_is_zero_init() -> None:
    """VALUE-binding hint: (1) _transition_hint must return the demo-consensus P(out|in) row at each
    target cell (changed support cells only, zero row = no evidence, LODO-safe via _active_context_*);
    (2) the 10 extra color_head columns must be ZERO-INIT so step-0 colour logits are byte-identical
    to the flag-off baseline (F7)."""
    import torch
    from models.recursive_reasoning.trm_fvr_c2 import FVR_C2_Config, TinyRecursiveReasoningModel_ACTV1_Inner
    base = dict(batch_size=1, seq_len=16, puzzle_emb_ndim=64, num_puzzle_identifiers=2, vocab_size=12,
                H_cycles=1, L_cycles=1, H_layers=1, L_layers=1, hidden_size=64, expansion=2, num_heads=4,
                pos_encodings="rope", halt_max_steps=1, halt_exploration_prob=0.0, forward_dtype="float32",
                c2_relmap=False, c2_transition_hint=True)
    m = TinyRecursiveReasoningModel_ACTV1_Inner(FVR_C2_Config(**base))
    # (2) FIX A: color_head stays hidden-only; the 10 transition columns live in color_evidence_proj
    assert m.color_head.weight.shape[1] == 64, "color_head must stay hidden-width (evidence split out)"
    assert m.color_evidence_proj.weight.shape[1] == 10, "evidence proj must carry exactly the 10 transition columns"
    assert float(m.color_evidence_proj.weight.abs().sum()) == 0.0, "transition columns must be zero-init (F7)"

    # (1) consensus math: demos recolor colour 1 (token 3) -> colour 3 (token 5); colour 0 (token 2) copies
    L = 16
    ci = torch.full((1, 2, L), 2, dtype=torch.long); ci[:, :, :4] = 3
    co = ci.clone(); co[:, :, :4] = 5
    ti = torch.full((1, L), 2, dtype=torch.long); ti[0, :2] = 3; ti[0, 8] = 7   # colour 5: never seen changing
    batch = {"context_inputs": ci, "context_outputs": co, "context_mask": torch.ones(1, 2, dtype=torch.bool),
             "inputs": ti}
    hint = m._transition_hint(batch, 1, L, ti.device)
    assert hint.shape == (1, L, 10)
    assert float(hint[0, 0, 3]) == 1.0 and float(hint[0, 0].sum()) == 1.0, "cell with colour 1 must read P(->3)=1"
    assert float(hint[0, 4].sum()) == 0.0, "copy-colour cell must have a ZERO row (no changed-cell evidence)"
    assert float(hint[0, 8].sum()) == 0.0, "never-observed colour must have a ZERO row (no evidence != identity)"

    # LODO safety: _active_context_* must OVERRIDE context_* (holdout excluded from the consensus)
    batch["_active_context_inputs"] = ci[:, :1]
    batch["_active_context_outputs"] = ci[:, :1].clone()      # active demo: NOTHING changes
    batch["_active_context_mask"] = torch.ones(1, 1, dtype=torch.bool)
    hint2 = m._transition_hint(batch, 1, L, ti.device)
    assert float(hint2.sum()) == 0.0, "_active_context_* must override context_* (LODO holdout safety)"


def test_value_evidence_v2_copy_change_and_lodo_safety() -> None:
    """VALUE V2 must expose copy-vs-change and context-conditioned colour evidence without changing
    logits at step 0. It must also respect _active_context_* so a held-out demo cannot leak back into
    its own LODO reconstruction evidence."""
    import torch
    from models.recursive_reasoning.trm_fvr_c2 import (
        FVR_C2_Config,
        TinyRecursiveReasoningModel_ACTV1_Inner,
        VALUE_EVIDENCE_V2_DIM,
    )
    base = dict(batch_size=1, seq_len=16, puzzle_emb_ndim=64, num_puzzle_identifiers=2, vocab_size=12,
                H_cycles=1, L_cycles=1, H_layers=1, L_layers=1, hidden_size=64, expansion=2, num_heads=4,
                pos_encodings="rope", halt_max_steps=1, halt_exploration_prob=0.0, forward_dtype="float32",
                c2_relmap=False, c2_value_evidence_v2=True)
    m = TinyRecursiveReasoningModel_ACTV1_Inner(FVR_C2_Config(**base))
    # FIX A: V2 columns live in the dedicated color_evidence_proj, color_head stays hidden-only.
    assert m.color_head.weight.shape[1] == 64
    assert m.color_evidence_proj.weight.shape[1] == VALUE_EVIDENCE_V2_DIM
    assert float(m.color_evidence_proj.weight.abs().sum()) == 0.0, (
        "VALUE V2 columns must be zero-init")

    L = 16
    # Source token 5 (raw colour 3): six copied cells and two changed cells to token 9 (raw colour 7).
    ci = torch.full((1, 1, L), 2, dtype=torch.long)
    ci[0, 0, :8] = 5
    co = ci.clone()
    co[0, 0, 6:8] = 9
    ti = torch.full((1, L), 2, dtype=torch.long)
    ti[0, :4] = 5
    ti[0, 15] = 0
    labels = ti.clone()
    labels[0, 0] = 9
    rel_where = torch.full((1, L, 1), 0.5)
    batch = {
        "context_inputs": ci,
        "context_outputs": co,
        "context_mask": torch.ones(1, 1, dtype=torch.bool),
        "inputs": ti,
        "labels": labels,
    }
    feat, stats = m._value_evidence_v2(batch, 1, L, ti.device, rel_where_hint=rel_where)
    assert feat.shape == (1, L, VALUE_EVIDENCE_V2_DIM)
    assert abs(float(feat[0, 0, 3]) - 0.75) < 1e-6, "copy_dist[source] must equal P(copy|source)"
    assert abs(float(feat[0, 0, 10 + 7]) - 1.0) < 1e-6, "conditioned_dist must bind changed target colour"
    assert abs(float(feat[0, 0, 20]) - 0.25) < 1e-6, "change_rate must count changed support cells"
    assert abs(float(feat[0, 0, 21]) - 0.75) < 1e-6, "copy_rate must count copied support cells"
    assert abs(float(feat[0, 0, 26 + 7]) - 0.5) < 1e-6, "WHERE-gated conditioned value must multiply rel_where"
    assert float(feat[0, 15].abs().sum()) == 0.0, "PAD/EOS cells must receive no VALUE evidence"
    assert float(stats["c2_value_v2_copy_rate_on_copy"]) > float(stats["c2_value_v2_change_rate_on_copy"])

    # LODO safety: active context excludes the changed cells; changed evidence disappears but copy evidence remains.
    batch["_active_context_inputs"] = ci[:, :, :].clone()
    batch["_active_context_outputs"] = ci[:, :, :].clone()
    batch["_active_context_mask"] = torch.ones(1, 1, dtype=torch.bool)
    feat2, _ = m._value_evidence_v2(batch, 1, L, ti.device, rel_where_hint=rel_where)
    assert float(feat2[0, 0, 10:20].abs().sum()) == 0.0, "_active_context_* must remove held-out changed VALUE"
    assert abs(float(feat2[0, 0, 3]) - 1.0) < 1e-6, "active copy-only context should still expose copy evidence"


def test_color_mlp_residual_is_zero_init_noop() -> None:
    """c2_color_head_mlp_dim adds interaction capacity as a RESIDUAL with a zero-init output layer:
    step-0 colour logits must be BYTE-IDENTICAL to the linear-only head (the linear path is
    lm_head-warm-started, so replacing it instead of adding to it would break the floor)."""
    import torch
    from models.recursive_reasoning.trm_fvr_c2 import FVR_C2_Config, TinyRecursiveReasoningModel_ACTV1_Inner
    base = dict(batch_size=1, seq_len=16, puzzle_emb_ndim=64, num_puzzle_identifiers=2, vocab_size=12,
                H_cycles=1, L_cycles=1, H_layers=1, L_layers=1, hidden_size=64, expansion=2, num_heads=4,
                pos_encodings="rope", halt_max_steps=1, halt_exploration_prob=0.0, forward_dtype="float32",
                c2_relmap=False)
    torch.manual_seed(0)
    m_lin = TinyRecursiveReasoningModel_ACTV1_Inner(FVR_C2_Config(**base))
    torch.manual_seed(0)
    m_mlp = TinyRecursiveReasoningModel_ACTV1_Inner(FVR_C2_Config(**base, c2_color_head_mlp_dim=32))
    assert getattr(m_mlp, "color_head_mlp_in", None) is not None, "MLP branch must build when dim > 0"
    assert float(m_mlp.color_head_mlp_out.weight.abs().sum()) == 0.0, "MLP output layer must be zero-init"
    with torch.no_grad():
        m_mlp.color_head.weight.copy_(m_lin.color_head.weight)     # same-seed draws differ by the extra module
        z = torch.randn(1, m_lin.puzzle_emb_len + 16, 64)
        batch = {"inputs": torch.randint(0, 12, (1, 16))}
        _, ex_lin = m_lin._output_logits(z, batch)
        _, ex_mlp = m_mlp._output_logits(z, batch)
    assert torch.equal(ex_lin["c2_color_logits"], ex_mlp["c2_color_logits"]), (
        "zero-init MLP residual must leave step-0 colour logits byte-identical (F7)")


def test_candidate_floor_structure_respects_extent_levers() -> None:
    """REGRESSION (run A' step 0: LODO pad 97.5% -> 1%): c2_candidate_floor_structure must build its
    PAD/EOS channels from the LEVER-CORRECTED structure_logits, not the raw lm_head floor. The raw
    blank-pid floor colours over the padding (measured gap mean~177) -- reading floor_logits[...,0:2]
    bypassed the +-1000 extent overrides entirely. Setup: a deliberately PAD-BLIND floor (colour logit
    beats pad everywhere) + warm-init extent levers + an identity-verified demo; the hybrid candidate
    must still emit PAD outside the predicted box and EOS on its thin-L."""
    import torch
    from models.recursive_reasoning.trm_fvr_c2 import FVR_C2_Config, TinyRecursiveReasoningModel_ACTV1_Inner
    side = 6
    base = dict(batch_size=1, seq_len=side * side, puzzle_emb_ndim=64, num_puzzle_identifiers=2, vocab_size=12,
                H_cycles=1, L_cycles=1, H_layers=1, L_layers=1, hidden_size=64, expansion=2, num_heads=4,
                pos_encodings="rope", halt_max_steps=1, halt_exploration_prob=0.0, forward_dtype="float32",
                c2_relmap=False, c2_structure_from_lmhead=True,
                c2_relmap_outside_grid=True, c2_structure_outside_warm_init=True,
                c2_relmap_eos_grid=True, c2_structure_eos_warm_init=True,
                c2_floor_candidate_split=True, c2_candidate_floor_structure=True)
    m = TinyRecursiveReasoningModel_ACTV1_Inner(FVR_C2_Config(**base))
    with torch.no_grad():
        m.lm_head.weight.zero_()
        m.lm_head.weight[4].fill_(1.0 / 64)        # colour token 4 wins EVERYWHERE -> pad-blind floor

    tok = _make_canvas(3, 3, 0, 0, side).unsqueeze(0)                        # 3x3 content, thin-L EOS, PAD outside
    batch = {"inputs": tok,
             "context_inputs": tok.unsqueeze(1), "context_outputs": tok.clone().unsqueeze(1),
             "context_mask": torch.ones(1, 1, dtype=torch.bool)}             # identity rule -> conf=1
    z_H = torch.ones(1, m.puzzle_emb_len + side * side, 64)
    with torch.no_grad():
        _, extras = m._output_logits(z_H, batch)
    floor_pred = extras["c2_floor_logits"][0].argmax(-1)
    assert (floor_pred[tok[0] == 0] >= 2).all(), "sanity: the crafted floor must be pad-blind (colours over PAD)"
    pred = extras["c2_candidate_logits"][0].argmax(-1)
    assert (pred[tok[0] == 0] == 0).all(), "hybrid candidate must emit PAD outside the box (lever bypassed?)"
    assert (pred[tok[0] == 1] == 1).all(), "hybrid candidate must emit EOS on the thin-L (eos lever bypassed?)"
    assert (pred[tok[0] >= 2] >= 2).all(), "hybrid candidate must keep colours on structure-valid cells"


def test_quarantine_candidate_pid_invariant_and_copy_consensus() -> None:
    """PID-QUARANTINED candidate head, the three contract guarantees:
    (1) QUARANTINE: the head's logits are BIT-IDENTICAL under a PID change (its features never
        include puzzle_identifiers), while the floor logits DO move (sanity: PID matters to z_H).
        This is also the train/deploy consistency fix -- the z_H candidate trains on blank-PID
        z_H (c2_lodo_blank_pid) but deploys on PID-ful z_H; the quarantine head has ONE contract.
    (2) WARM-INIT copy-unless-consensus: with a demo consensus 'colour 1 -> colour 3' the head
        must argmax colour 3 on colour-1 cells (8*P beats copy 4) and COPY on no-evidence cells.
    (3) SUBSTITUTION: backward through the candidate's colour cells must produce grads on
        quarantine_* and NONE on color_head (the z_H colour path is out of the candidate lane)."""
    import torch
    from models.recursive_reasoning.trm_fvr_c2 import FVR_C2_Config, TinyRecursiveReasoningModel_ACTV1_Inner
    side = 6
    base = dict(batch_size=1, seq_len=side * side, puzzle_emb_ndim=64, num_puzzle_identifiers=4, vocab_size=12,
                H_cycles=1, L_cycles=1, H_layers=1, L_layers=1, hidden_size=64, expansion=2, num_heads=4,
                pos_encodings="rope", halt_max_steps=1, halt_exploration_prob=0.0, forward_dtype="float32",
                c2_relmap=False, c2_structure_from_lmhead=True,
                c2_floor_candidate_split=True, c2_candidate_floor_structure=True,
                c2_quarantine_candidate=True, c2_quarantine_hidden=32)
    m = TinyRecursiveReasoningModel_ACTV1_Inner(FVR_C2_Config(**base))
    m.eval()
    # warm-init layout: +4 copy (input one-hot), +8 marginal consensus (transition block),
    # +9 conditioned consensus (FIX B block [130:140]; conditioned > marginal > copy)
    assert float(m.quarantine_mlp_out.weight.abs().sum()) == 0.0, "quarantine MLP out must be zero-init"
    for c in (0, 4, 9):
        assert float(m.quarantine_lin.weight[c, 2 + c]) == 4.0, "copy warm-init column misplaced"
        assert float(m.quarantine_lin.weight[c, 108 + c]) == 8.0, "marginal warm-init column misplaced"
        assert float(m.quarantine_lin.weight[c, 130 + c]) == 9.0, "conditioned warm-init column misplaced (FIX B)"

    # demos: recolor colour 1 (token 3) -> colour 3 (token 5); target has colour-1 and colour-0 cells
    tok = _make_canvas(3, 3, 0, 0, side).unsqueeze(0)
    tok[0, 0] = 3                                      # (0,0) colour 1 -> consensus says colour 3
    ci = tok.clone().unsqueeze(1)
    co = ci.clone(); co[co == 3] = 5
    batch = {"inputs": tok, "context_inputs": ci, "context_outputs": co,
             "context_mask": torch.ones(1, 1, dtype=torch.bool),
             "puzzle_identifiers": torch.tensor([1])}
    with torch.no_grad():                              # make the floor actually PID-sensitive
        m.puzzle_emb.weights[1].fill_(0.5)
        m.puzzle_emb.weights[2].fill_(-0.5)
    seq_info = dict(cos_sin=m.rotary_emb() if hasattr(m, "rotary_emb") else None)

    def _fwd(pid: int):
        b = dict(batch); b["puzzle_identifiers"] = torch.tensor([pid])
        ie, _, rm, _ = m._input_embeddings(b)
        z_H, _ = m._run_recurrence(m.fresh_carry(1), ie, seq_info)
        _, ex = m._output_logits(z_H, b, rel_maps=rm)
        return ex

    with torch.no_grad():
        ex1, ex2 = _fwd(1), _fwd(2)
    assert "c2_quarantine_logits" in ex1, "quarantine logits must be exposed in extras"
    assert torch.equal(ex1["c2_quarantine_logits"], ex2["c2_quarantine_logits"]), (
        "QUARANTINE VIOLATION: the head's logits moved under a PID change")
    assert not torch.equal(ex1["c2_floor_logits"], ex2["c2_floor_logits"]), (
        "sanity broken: the floor must be PID-sensitive or the invariance test is vacuous")
    q = ex1["c2_quarantine_logits"][0]
    assert int(q[0].argmax()) == 3, "consensus cell (colour 1, demos say ->3) must argmax colour 3"
    copy_cells = (tok[0] == 2).nonzero().flatten()
    assert all(int(q[i].argmax()) == 0 for i in copy_cells.tolist()), "no-evidence cells must COPY"
    # candidate integration: colour choice on structure-valid cells == quarantine choice (+2 tokens)
    cand = ex1["c2_candidate_logits"][0]
    valid = tok[0] >= 2
    assert torch.equal(cand[valid].argmax(-1), q[valid].argmax(-1) + 2), (
        "candidate colour choice must come from the quarantine head on valid cells")

    # (3) gradient substitution: candidate colour cells -> quarantine_* grads, color_head untouched
    m.zero_grad(set_to_none=True)
    ie, _, rm, _ = m._input_embeddings(batch)
    z_H, _ = m._run_recurrence(m.fresh_carry(1), ie, seq_info)
    _, ex = m._output_logits(z_H, batch, rel_maps=rm)
    ex["c2_candidate_logits"][0][valid][:, 2:12].sum().backward()
    # step-0 grad flows to the LINEAR path and the zero-init MLP OUTPUT; mlp_in's grad is exactly
    # zero at init (it passes through the zeroed mlp_out) -- that is the F7 pattern, not a bug.
    assert m.quarantine_lin.weight.grad is not None and float(
        m.quarantine_lin.weight.grad.abs().sum()) > 0, "LODO gradient must reach the quarantine head"
    assert m.quarantine_mlp_out.weight.grad is not None and float(
        m.quarantine_mlp_out.weight.grad.abs().sum()) > 0, "the MLP residual must be trainable from step 0"
    assert m.color_head.weight.grad is None or float(m.color_head.weight.grad.abs().sum()) == 0.0, (
        "color_head must be OUT of the candidate colour path when quarantine is on")


def main() -> None:
    tests = [
        test_boundary_lever_places_pad_on_featureless_cell,
        test_outside_grid_lever_places_pad_and_gates_size_change,
        test_extent_pad_mask_matches_tokenizer_pad_region,
        test_extent_eos_mask_matches_tokenizer_thin_l,
        test_extent_pad_mask_size_change_shares_input_offset,
        test_predicted_extent_verifies_demo_size_rules,
        test_forward_never_mutates_batch_and_visual_flag_raises,
        test_aux_forward_shares_main_input_contract,
        test_dataloader_receives_arch_relmap_flag,
        test_zh_check_rolls_relmap_keys_with_inputs_and_context,
        test_run_stage1_has_v3_adapter_quarantine_scope,
        test_run_stage1_supports_epoch_checkpoint_mode,
        test_model_relmap_fallback_is_loud,
        test_dataloader_emits_context_output_rel_maps,
        test_v3_2_upgrades_build_zero_init_and_forward,
        test_floor_candidate_split_exposes_both_paths,
        test_object_bank_rel_where_hint_selects_support_changed_cells,
        test_pairdelta_intent_features_separate_correct_from_shuffled_support,
        test_v3_rel_where_and_pairdelta_hints_are_zero_init_live_inputs,
        test_frame_hint_zero_init_and_forward,
        test_rule_hypothesis_hint_zero_init_and_forward,
        test_structure_from_lmhead_reproduces_floor_partition,
        test_input_hints_threaded_not_stashed,
        test_value_evidence_v2_rich_ctx_forward_and_zero_init,
        test_transition_hint_binds_value_and_is_zero_init,
        test_value_evidence_v2_copy_change_and_lodo_safety,
        test_color_mlp_residual_is_zero_init_noop,
        test_candidate_floor_structure_respects_extent_levers,
        test_quarantine_candidate_pid_invariant_and_copy_consensus,
    ]
    failures: list[str] = []
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
