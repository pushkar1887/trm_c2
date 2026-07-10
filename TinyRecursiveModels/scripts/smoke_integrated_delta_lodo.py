"""Smoke: the Phase-B delta-LODO loss integrated into FVRACTLossHead (main pipeline).

Builds the model + loss head exactly like pretrain.py (create_model), enables the delta
branch + factored head, sets arch.loss.c2_delta_lodo_weight>0, and runs ONE real loss-head
step on an aug-1000 batch. Verifies:
  - model emits c2_aux_inputs / c2_aux_logits (the held-out-demo reconstruction)
  - FVRACTLossHead computes the two-region delta-LODO term (metric present, finite, >0)
  - total loss is finite and backprops into the delta branch
This is the integration check before pushing to Kaggle (where pretrain.py runs it under DDP).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrain  # noqa: E402

CONFIG = "checkpoints/TRM-FVR-Experiments/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/all_config.yaml"
CKPT = r"D:\trm_c2\step_518071"
DATASET = r"D:\trm_c2\arc1concept-aug-1000"


def main() -> None:
    raw = yaml.safe_load(Path(CONFIG).resolve().read_text(encoding="utf-8"))
    raw["load_checkpoint"] = CKPT
    raw["data_paths"] = [DATASET]
    raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0
    raw["global_batch_size"] = 8
    raw["run_name"] = "smoke_delta_lodo"
    raw["checkpoint_path"] = "reports/smoke_delta_lodo_noop"
    arch = raw.setdefault("arch", {})
    # enable ALL the components (same flags as the validated runs)
    arch["c2_structure_fusion_alpha"] = 0.0
    arch["c2_delta_rule_branch"] = True
    arch["c2_delta_rule_encoder_dim"] = 256
    arch["c2_delta_rule_slots"] = 8
    arch["c2_delta_rule_logit_residual"] = True
    arch["c2_delta_rule_logit_replace"] = True
    arch["c2_delta_rule_slot_attend"] = True
    arch["c2_delta_rule_factored_head"] = True
    arch["c2_delta_rule_cell_gate_bias"] = -2.0
    arch["c2_delta_expose_rule_vec"] = True          # Stage 0: surface r_full / r_loo / r_shuf
    arch["c2_delta_expose_base_logits"] = True       # Stage 0: surface P_off (for the Stage-2 KL)
    arch["c2_lodo_blank_pid"] = True
    arch["c2_leave_one_demo_weight"] = 0.0          # OLD weak LODO OFF
    arch["c2_lodo_force_build"] = True              # but STILL build the aux batch for my term
    arch["c2_lodo_force_shuffle"] = True            # build the wrong-task aux for the contrast
    arch["c2_lodo_contrast_weight"] = 0.0           # OLD plain-CE contrast OFF
    # NEW: my two-region delta-LODO loss + changed-cell CONTRAST as the cross-demo trainer
    loss = arch.setdefault("loss", {})
    loss["c2_delta_lodo_weight"] = 0.5
    loss["c2_delta_changed_weight"] = 5.0
    loss["c2_delta_color_weight"] = 1.0
    loss["c2_delta_pad_weight"] = 1.0
    loss["c2_delta_eos_weight"] = 3.0
    loss["c2_delta_contrast_weight"] = 1.0
    loss["c2_delta_contrast_margin"] = 0.5
    # Stage 0 panel + Stage 1 rule task-specificity (the margin=0.018 fix)
    loss["c2_delta_diag"] = True
    loss["c2_delta_nce_weight"] = 1.0
    loss["c2_delta_cons_weight"] = 0.5
    loss["c2_delta_nce_tau"] = 0.1

    config = pretrain.PretrainConfig(**raw)
    loader, meta = pretrain.create_dataloader(config, "train", 0, 1, test_set_mode=False,
                                              epochs_per_iter=1, global_batch_size=config.global_batch_size)
    loss_head, _, _ = pretrain.create_model(config, meta, rank=0, world_size=1)
    loss_head.train()                                # _build_lodo_batch requires train mode
    device = torch.device("cuda")

    # grab first batch that has context demos
    batch = None
    for _set, cb, _g in loader:
        if "context_inputs" in cb and cb["context_inputs"].shape[1] >= 2:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
            break
    assert batch is not None, "no batch with context demos"

    with torch.device("cuda"):                       # carry buffers must be built on cuda
        carry = loss_head.initial_carry(batch)
    carry, loss_val, metrics, _, halted = loss_head(carry=carry, batch=batch, return_keys=[])

    print("=== integrated delta-LODO smoke ===")
    print(f"total loss            = {float(loss_val.item()):.4f}  finite={torch.isfinite(loss_val).item()}")
    have = lambda k: k in metrics
    for k in ["lm_loss_main", "c2_aux_weighted_loss", "c2_delta_lodo_weighted_loss",
              "c2_delta_lodo_raw", "c2_delta_lodo_inside", "c2_delta_lodo_outside", "c2_pad_loss",
              "c2_delta_contrast_weighted_loss", "c2_delta_contrast_real_changed",
              "c2_delta_contrast_shuffle_changed", "c2_delta_contrast_gap",
              # Stage 0 panel + Stage 1 rule losses
              "d_strict_exact", "d_content_exact", "d_changed_color_acc", "d_pad_acc", "d_eos_acc",
              "d_rule_margin", "d_same_task_cos", "d_other_task_cos",
              "l_nce", "l_cons", "c2_delta_nce_weighted_loss", "c2_delta_cons_weighted_loss"]:
        print(f"  metric {k:36s} present={have(k)} "
              f"value={float(metrics[k].item()) if have(k) else float('nan'):.4f}")

    assert have("c2_delta_lodo_weighted_loss"), "FATAL: delta-LODO term not in metrics (not wired)."
    dl = float(metrics["c2_delta_lodo_raw"].item())
    assert dl > 0.0, f"FATAL: delta-LODO raw loss is 0 ({dl}) -> aux path not producing/consumed."
    assert have("c2_delta_contrast_real_changed"), "FATAL: contrast term not wired (no shuffle aux?)."
    assert torch.isfinite(loss_val).item(), "FATAL: total loss not finite."
    # Stage 0/1 assertions
    assert have("d_rule_margin"), "FATAL: d_rule_margin missing -> rule vecs not exposed/consumed."
    assert have("l_nce") and have("l_cons"), "FATAL: NCE/cons losses not wired."
    assert torch.isfinite(metrics["l_nce"]).item() and float(metrics["l_nce"].item()) != 0.0, \
        "FATAL: l_nce 0/non-finite -> r_full/r_loo not feeding NCE."
    assert have("d_strict_exact") and have("d_changed_color_acc"), "FATAL: Stage-0 panel missing."

    loss_val.backward()
    # confirm gradient reached the delta branch (the new cross-demo trainer)
    inner = loss_head.model.inner
    g_head = inner.delta_rule_color_head.weight.grad
    g_enc = inner.delta_rule_proj.weight.grad
    gh = float(g_head.norm().item()) if g_head is not None else 0.0
    ge = float(g_enc.norm().item()) if g_enc is not None else 0.0
    print(f"  grad delta_rule_color_head = {gh:.3e}")
    print(f"  grad delta_rule_proj       = {ge:.3e}")
    assert gh > 0.0 or ge > 0.0, "FATAL: no gradient reached the delta branch."
    # Stage 1: NCE/cons must train the rule ENCODER (the bank), not just the branch.
    g_enc_sq = 0.0
    g_enc_n = 0
    g_pair_mlp = 0.0
    for _n, _p in inner.delta_rule_encoder.named_parameters():
        if _p.grad is not None:
            gn = float(_p.grad.norm().item())
            g_enc_sq += gn * gn
            g_enc_n += 1
            if _n.startswith("pair_mlp"):
                g_pair_mlp += gn * gn
    g_enc_total = g_enc_sq ** 0.5
    g_pair_mlp = g_pair_mlp ** 0.5
    print(f"  grad delta_rule_encoder TOTAL = {g_enc_total:.3e} over {g_enc_n} params "
          f"(pair_mlp={g_pair_mlp:.3e})")
    assert g_enc_total > 0.0, "FATAL: NCE/cons did not reach delta_rule_encoder (rule bank)."
    print("\n[PASS] Stage 0 panel + Stage 1 NCE/cons wired; rule encoder (bank) is trained.")


if __name__ == "__main__":
    main()
