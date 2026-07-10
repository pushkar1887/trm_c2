"""Wiring tests A-E for the C2 Structural Rule Branch.

Verifies that enabling `c2_use_struct_rule_branch=True` with gate init 0.0:
  - Constructs the struct_* modules only when enabled (Tests A, B)
  - Preserves the initialization of all OTHER parameters (Test C)
  - Produces forward-pass logits identical to branch-off within numerical tol (Test D)
  - Routes gradients through the gate and the projection layers (Test E)

Run with:
    trm/Scripts/python.exe scripts/test_struct_rule_branch_wiring.py
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain


REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "struct_rule_branch_v005_aug1000_seed0_evalaug0_pid401" / "wiring_tests_report.md"
BASE_CHECKPOINT = Path("D:/trm_c2/step_518071")
REFERENCE_CONFIG = Path(
    "D:/trm_c2/TinyRecursiveModels/checkpoints/TRM-FVR-Experiments/"
    "c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/all_config.yaml"
)
TRAIN_DATA = "data/arc1concept-aug-0"  # use AUG0 for tests (lightweight)


def build_config(struct_branch_on: bool) -> pretrain.PretrainConfig:
    raw = yaml.safe_load(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    # Point at trm_c2 base checkpoint for testing
    raw["load_checkpoint"] = str(BASE_CHECKPOINT)
    raw["data_paths"] = [TRAIN_DATA]
    raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0
    raw["checkpoint_path"] = str(REPORT_PATH.parent / "noop_checkpoints")
    raw["run_name"] = "wiring_test"
    raw["global_batch_size"] = 1
    raw.setdefault("arch", {})["c2_use_struct_rule_branch"] = bool(struct_branch_on)
    raw["arch"]["c2_struct_rule_gate_init"] = 0.0
    raw["arch"]["c2_structure_fusion_alpha"] = 0.0
    return pretrain.PretrainConfig(**raw)


def construct_model(struct_branch_on: bool, seed: int = 0) -> torch.nn.Module:
    config = build_config(struct_branch_on)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader, train_metadata = pretrain.create_dataloader(
        config, "train", 0, 1,
        test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    del train_loader
    torch.manual_seed(seed)
    np.random.seed(seed)
    loss_head, _, _ = pretrain.create_model(config, train_metadata, rank=0, world_size=1)
    return loss_head, config, train_metadata


def fetch_first_batch(config) -> Dict[str, torch.Tensor]:
    eval_loader, _ = pretrain.create_dataloader(
        config, "test", 0, 1,
        test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    for _set_name, batch, _gbs in eval_loader:
        return {k: v for k, v in batch.items()}
    raise RuntimeError("No batches returned from eval loader.")


def test_a_default_off() -> Dict[str, object]:
    print("\n=== TEST A: Default-off construction ===")
    loss_head, _config, _md = construct_model(struct_branch_on=False, seed=0)
    c2 = loss_head.model.inner.c2
    result = {
        "has_struct_feature_proj": hasattr(c2, "struct_feature_proj"),
        "has_struct_update_proj": hasattr(c2, "struct_update_proj"),
        "has_gate_struct": hasattr(c2, "gate_struct"),
    }
    passes = (not result["has_struct_feature_proj"]
              and not result["has_struct_update_proj"]
              and not result["has_gate_struct"])
    return {"name": "A_default_off", "passes": bool(passes), "detail": result}


def test_b_branch_on() -> Dict[str, object]:
    print("\n=== TEST B: Branch-on construction ===")
    loss_head, _config, _md = construct_model(struct_branch_on=True, seed=0)
    c2 = loss_head.model.inner.c2
    detail = {
        "has_struct_feature_proj": hasattr(c2, "struct_feature_proj"),
        "has_struct_update_proj": hasattr(c2, "struct_update_proj"),
        "has_gate_struct": hasattr(c2, "gate_struct"),
        "gate_struct_init": float(c2.gate_struct.detach().cpu()) if hasattr(c2, "gate_struct") else None,
    }
    passes = (detail["has_struct_feature_proj"]
              and detail["has_struct_update_proj"]
              and detail["has_gate_struct"]
              and detail["gate_struct_init"] == 0.0)
    return {"name": "B_branch_on", "passes": bool(passes), "detail": detail}


def test_c_rng_safety() -> Dict[str, object]:
    print("\n=== TEST C: Same-seed RNG safety ===")
    loss_off, _, _ = construct_model(struct_branch_on=False, seed=0)
    loss_on, _, _ = construct_model(struct_branch_on=True, seed=0)
    c2_off = loss_off.model.inner.c2
    c2_on = loss_on.model.inner.c2
    shared_attr_paths = [
        "demo_proj.weight", "demo_scalar_proj.weight", "demo_mix.weight",
        "pair_proj.weight", "pair_mix.weight",
        "cross_attn.q_proj.weight", "cross_attn.k_proj.weight",
        "cross_attn.v_proj.weight", "cross_attn.o_proj.weight",
        "patch_proj.weight", "global_proj.weight",
    ]
    per_param: Dict[str, float] = {}
    for path in shared_attr_paths:
        a = c2_off
        b = c2_on
        for part in path.split("."):
            a = getattr(a, part)
            b = getattr(b, part)
        diff = (a.detach().float() - b.detach().float()).abs().max().item()
        per_param[path] = diff
    # pid_task_modulator is on the outer inner model when c2_modulate_pid=True
    outer_off = loss_off.model.inner
    outer_on = loss_on.model.inner
    if hasattr(outer_off, "pid_task_modulator") and hasattr(outer_on, "pid_task_modulator"):
        diff = (outer_off.pid_task_modulator.weight.detach().float()
                - outer_on.pid_task_modulator.weight.detach().float()).abs().max().item()
        per_param["pid_task_modulator.weight"] = diff
    max_diff = max(per_param.values()) if per_param else 0.0
    passes = max_diff == 0.0
    return {"name": "C_rng_safety", "passes": bool(passes), "max_diff": max_diff, "per_param": per_param}


def test_d_zero_gate_forward_equiv() -> Dict[str, object]:
    print("\n=== TEST D: Zero-gate forward equivalence ===")
    loss_off, config_off, md = construct_model(struct_branch_on=False, seed=0)
    loss_on, config_on, _ = construct_model(struct_branch_on=True, seed=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return {"name": "D_zero_gate_forward_equiv", "passes": False, "error": "CUDA required"}
    loss_off.eval()
    loss_on.eval()
    batch = fetch_first_batch(config_off)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        with torch.device(device.type):
            carry_off = loss_off.model.initial_carry(batch)
            carry_on = loss_on.model.initial_carry(batch)
        out_off = None
        out_on = None
        n_steps = int(config_off.arch.halt_max_steps)
        for _ in range(n_steps):
            carry_off, out_off = loss_off.model(carry=carry_off, batch=batch)
        for _ in range(n_steps):
            carry_on, out_on = loss_on.model(carry=carry_on, batch=batch)
    logits_off = out_off["logits"].float()
    logits_on = out_on["logits"].float()
    max_abs_diff = (logits_off - logits_on).abs().max().item()
    passes = max_abs_diff < 1e-5
    return {"name": "D_zero_gate_forward_equiv", "passes": bool(passes),
            "max_abs_logit_diff": max_abs_diff, "tolerance": 1e-5}


def test_e_backward_learnability() -> Dict[str, object]:
    print("\n=== TEST E: Backward learnability ===")
    loss_head, config, _md = construct_model(struct_branch_on=True, seed=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        return {"name": "E_backward_learnability", "passes": False, "error": "CUDA required"}
    c2 = loss_head.model.inner.c2
    # Force the gate to a non-zero value so projection layers get gradient signal
    with torch.no_grad():
        c2.gate_struct.fill_(0.01)
    c2.gate_struct.requires_grad_(True)
    loss_head.train()
    batch = fetch_first_batch(config)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.device(device.type):
        carry = loss_head.model.initial_carry(batch)
    # forward returns (new_carry, loss, metrics, detached_outputs, halted_all)
    carry, loss_value, metrics, _outputs, _halted = loss_head(carry=carry, batch=batch, return_keys=[])
    if loss_value is None or not loss_value.requires_grad:
        return {"name": "E_backward_learnability", "passes": False,
                "error": f"loss not differentiable; metrics_keys={list(metrics.keys())[:10]}"}
    loss_value.backward()
    grad_gate = c2.gate_struct.grad
    grad_proj = c2.struct_update_proj.weight.grad if hasattr(c2, "struct_update_proj") else None
    grad_feat = c2.struct_feature_proj.weight.grad if hasattr(c2, "struct_feature_proj") else None
    detail = {
        "gate_struct_grad_present": grad_gate is not None,
        "gate_struct_grad_finite": bool(grad_gate is not None and torch.isfinite(grad_gate).all().item()),
        "gate_struct_grad_value": float(grad_gate.item()) if grad_gate is not None else None,
        "struct_update_proj_grad_present": grad_proj is not None,
        "struct_update_proj_grad_finite": bool(grad_proj is not None and torch.isfinite(grad_proj).all().item()),
        "struct_update_proj_grad_l2": float(grad_proj.norm().item()) if grad_proj is not None else None,
        "struct_feature_proj_grad_present": grad_feat is not None,
        "struct_feature_proj_grad_finite": bool(grad_feat is not None and torch.isfinite(grad_feat).all().item()),
        "struct_feature_proj_grad_l2": float(grad_feat.norm().item()) if grad_feat is not None else None,
    }
    passes = (detail["gate_struct_grad_present"]
              and detail["gate_struct_grad_finite"]
              and detail["struct_update_proj_grad_present"]
              and detail["struct_update_proj_grad_finite"]
              and detail["struct_feature_proj_grad_present"]
              and detail["struct_feature_proj_grad_finite"])
    return {"name": "E_backward_learnability", "passes": bool(passes), "detail": detail}


def write_report(results) -> None:
    lines = [
        "# Phase 2 Wiring Tests Report — C2 Structural Rule Branch",
        "",
        "**Script**: `scripts/test_struct_rule_branch_wiring.py`",
        "**Base checkpoint**: `D:\\trm_c2\\step_518071`",
        "**Data**: `data/arc1concept-aug-0` (lightweight, AUG0 canonical)",
        "",
        "## Results",
        "",
        "| Test | Status | Notes |",
        "|---|---|---|",
    ]
    for r in results:
        status = "PASS" if r["passes"] else "FAIL"
        note_parts = []
        for k, v in r.items():
            if k in ("name", "passes"): continue
            note_parts.append(f"`{k}={v}`")
        notes = "; ".join(note_parts)[:240] or "—"
        lines.append(f"| {r['name']} | {status} | {notes} |")
    lines.append("")
    overall = all(r["passes"] for r in results)
    lines.append(f"**Overall verdict**: {'ALL PASS — safe to proceed to Phase 3 fine-tune' if overall else 'FAIL — do not proceed; investigate failures above'}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[done] report -> {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-d-and-e", action="store_true", help="Skip the GPU forward/backward tests (D, E)")
    args = parser.parse_args()
    results = []
    results.append(test_a_default_off())
    results.append(test_b_branch_on())
    results.append(test_c_rng_safety())
    if not args.skip_d_and_e:
        results.append(test_d_zero_gate_forward_equiv())
        results.append(test_e_backward_learnability())
    write_report(results)
    print()
    for r in results:
        print(f"  {r['name']}: {'PASS' if r['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
