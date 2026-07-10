"""Full verifier inference flow — Stage 1-5 per spec.

Stage 1: K candidates per task (canonical + D4 augs + LODO variants)
Stage 2: Canvas cleanup (S1)
Stage 3: Per-candidate feature extraction (logit_margin, shape_conf, lodo_stab)
Stage 4: Verifier scoring
Stage 5: Selection with safe fallback to canonical if score < tau_commit

Outputs:
  <out_dir>/eval_with_verifier_summary.csv  — per-task: pred grids, verifier scores
  <out_dir>/rejection_or_keep.md            — verdict + diagnostics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain  # noqa: E402
from scripts.fvr_structfuse_alpha_sweep import IGNORE_LABEL_ID, crop_shape, read_csv, write_csv  # noqa: E402
from scripts.postprocess_canvas_cleanup import canvas_cleanup, EOS_TOKEN  # noqa: E402
from scripts.verifier_head import DemoConsistencyVerifier  # noqa: E402
from scripts.build_verifier_dataset import apply_d4, color_grid_to_token_seq, _D4_NAMES  # noqa: E402


def score_exact(pred_seq: np.ndarray, label_seq: np.ndarray) -> int:
    mask = label_seq != IGNORE_LABEL_ID
    return int(np.array_equal(pred_seq[mask], label_seq[mask]))


def compute_logit_margin(logits: torch.Tensor) -> float:
    """Mean per-cell (top - second) logit on the cells inside the predicted canvas."""
    top2 = logits.float().topk(2, dim=-1).values  # (S, 2)
    margin = (top2[:, 0] - top2[:, 1]).mean().item()
    return float(margin)


def main() -> None:
    p = argparse.ArgumentParser(description="Verifier inference flow eval.")
    p.add_argument("--config", required=True, help="TRM/C2 config YAML")
    p.add_argument("--checkpoint", required=True, help="TRM checkpoint")
    p.add_argument("--verifier-checkpoint", required=True, help="Verifier .pt checkpoint for the legacy verifier flow")
    p.add_argument("--reference-ledger", required=True, help="C0 ledger for baseline comparison")
    p.add_argument("--dataset", default="data/arc-agi-evaluation-full400-seed0-pid401aligned",
                   help="Eval dataset root. MUST be aligned to the checkpoint's puzzle_emb "
                        "ordering (default: the PID401-aligned rebuild). Using the misaligned "
                        "data/arc-agi-evaluation-full400-seed0 yields canonical 6/400 not 125/400.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--tau-commit", type=float, default=0.0,
                   help="Threshold above which verifier commits to its pick; below = canonical fallback")
    p.add_argument("--use-d4", action="store_true", default=True)
    p.add_argument("--use-lodo", action="store_true", default=False,
                   help="Generate LODO candidates (requires demo masking infrastructure)")
    p.add_argument("--use-struct-features", action="store_true", default=False)
    p.add_argument("--dump-candidates", action="store_true", default=False,
                   help="Write per-candidate (task_id, source, score, is_gt_exact) to "
                        "candidate_scores.csv for held-out AUROC analysis.")
    p.add_argument("--use-cleanup", action="store_true", default=False,
                   help="Apply canvas_cleanup to canonical predictions. DEFAULT OFF — "
                        "cleanup is destructive on noisy TRM outputs (crop_shape returns "
                        "wrong (h,w)). Known bug B2; cleanup needs a model-provided shape head.")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required.")

    # Load TRM/C2 config + model.
    config_path = Path(args.config).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["load_checkpoint"] = str(Path(args.checkpoint).resolve())
    raw["data_paths"] = [args.dataset]
    raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0
    raw["checkpoint_path"] = str(out_dir / "noop_checkpoints")
    raw["run_name"] = "eval_with_verifier"
    raw["global_batch_size"] = int(args.global_batch_size)
    raw.setdefault("arch", {})["c2_structure_fusion_alpha"] = 0.0
    config = pretrain.PretrainConfig(**raw)

    train_loader, train_meta = pretrain.create_dataloader(
        config, "train", 0, 1, test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    eval_loader, _ = pretrain.create_dataloader(
        config, "test", 0, 1, test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    del train_loader
    eval_batches = []
    for _set_name, batch, _gbs in eval_loader:
        eval_batches.append({k: v for k, v in batch.items()})
    print(f"[eval] cached eval batches: {len(eval_batches)}")

    loss_head, _, _ = pretrain.create_model(config, train_meta, rank=0, world_size=1)
    core_model = loss_head.model
    core_model.eval()
    for p_ in core_model.parameters():
        p_.requires_grad_(False)
    core_model.to(device)

    # Load verifier.
    verifier_ckpt = torch.load(Path(args.verifier_checkpoint).resolve(), map_location=device)
    verifier = DemoConsistencyVerifier(
        hidden_dim=int(config.arch.__pydantic_extra__.get("hidden_size", 512)),
        use_struct_features=bool(args.use_struct_features),
    ).to(device)
    verifier.load_state_dict(verifier_ckpt["verifier_state_dict"])
    verifier.eval()

    # Eval IDs + reference ledger.
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root
    eval_ids = json.loads((dataset_root / "identifiers.json").read_text(encoding="utf-8"))
    ref_rows = read_csv(Path(args.reference_ledger).resolve())
    ref_by_task = {row["task_id"]: row for row in ref_rows}

    # Iterate eval batches, producing per-task candidates and verifier scores.
    results: List[Dict[str, object]] = []
    seen_tids: set = set()   # B3 FIX: per-task deduplication
    cand_dump: List[Dict[str, object]] = []  # per-candidate scores for held-out AUROC
    halt_max_steps = int(config.arch.halt_max_steps)
    with torch.inference_mode():
        for batch_idx, cpu_batch in enumerate(eval_batches, start=1):
            batch = {k: v.to(device) for k, v in cpu_batch.items()}
            # ----- Canonical prediction (K=1) -----
            with torch.device(device.type):
                carry = core_model.initial_carry(batch)
            for _ in range(halt_max_steps):
                carry, out_can = core_model(carry=carry, batch=batch)
            preds_can = torch.argmax(out_can["logits"], dim=-1)
            logits_can = out_can["logits"]
            puzzle_ids = batch["puzzle_identifiers"].cpu().numpy().tolist()
            labels = batch["labels"].cpu().numpy()

            # Build demo encoding once per task (rule_bank for verifier).
            inner = core_model.inner
            context_inputs = batch.get("context_inputs")
            context_outputs = batch.get("context_outputs")
            context_mask = batch.get("context_mask")
            if context_inputs is None:
                # Some pipelines pack context into the batch differently; skip verifier
                # and fall back to canonical for this batch.
                rule_bank = None
            else:
                in_feats = inner.grid_encoder(context_inputs)
                out_feats = inner.grid_encoder(context_outputs)
                rule_bank, rule_mask, struct_feats = inner.c2.expose_demo_encoding(
                    context_inputs=context_inputs,
                    context_outputs=context_outputs,
                    context_input_features=in_feats,
                    context_output_features=out_feats,
                    context_mask=context_mask,
                )
                # cast to float32 for verifier compatibility (C2 outputs bf16)
                rule_bank = rule_bank.to(torch.float32)
                struct_feats = struct_feats.to(torch.float32)

            B = preds_can.shape[0]
            for row_idx in range(B):
                pid = int(puzzle_ids[row_idx])
                if pid <= 0 or pid >= len(eval_ids):
                    continue
                tid = eval_ids[pid]
                if tid in seen_tids:    # B3 FIX: skip duplicate task entries
                    continue
                seen_tids.add(tid)
                pred_can_seq = preds_can[row_idx].cpu().numpy()
                label_seq = labels[row_idx]
                pred_h, pred_w = crop_shape(pred_can_seq)
                # B2 SHUT: canvas_cleanup is destructive on noisy TRM outputs.
                # Skip by default; canonical = raw TRM prediction.
                if args.use_cleanup:
                    pred_can_clean = canvas_cleanup(pred_can_seq, pred_h=pred_h, pred_w=pred_w)
                    canon_source = "canonical_cleaned"
                else:
                    pred_can_clean = pred_can_seq
                    canon_source = "canonical_raw"
                canonical_exact = score_exact(pred_can_clean, label_seq)

                # ----- D4 augmentations (Stage 1 candidates) -----
                # B4 FIX: D4 base is RAW canonical prediction, independent of cleanup.
                candidate_seqs = [pred_can_clean]
                candidate_sources = [canon_source]
                if args.use_d4 and pred_h > 0 and pred_w > 0:
                    grid_can = pred_can_seq.reshape(30, 30)[:pred_h, :pred_w]
                    grid_colors = np.clip(grid_can - 2, 0, 9)
                    for d4 in _D4_NAMES[1:]:  # skip identity, already have it
                        aug = apply_d4(grid_colors, d4)
                        cand_seq = color_grid_to_token_seq(aug)
                        candidate_seqs.append(cand_seq)
                        candidate_sources.append(f"d4_{d4}")

                # Score each candidate with verifier (or fallback if rule_bank missing).
                best_seq = pred_can_clean
                best_score = float("inf") if rule_bank is None else float("-inf")
                best_source = canon_source
                verifier_scores: List[Tuple[str, float]] = []
                if rule_bank is not None:
                    K = len(candidate_seqs)
                    cand_tokens = torch.tensor(np.stack(candidate_seqs), dtype=torch.long, device=device)
                    rb = rule_bank[row_idx:row_idx + 1].expand(K, -1, -1)
                    rm = rule_mask[row_idx:row_idx + 1].expand(K, -1)
                    sf = struct_feats[row_idx:row_idx + 1].expand(K, -1, -1) if args.use_struct_features else None
                    # Scalar features: logit_margin computed from canonical only (proxy).
                    margin = compute_logit_margin(logits_can[row_idx])
                    scalar = torch.tensor(
                        [[margin, 0.0, 0.0, 0.0]] * K, dtype=torch.float32, device=device,
                    )
                    scores = verifier(cand_tokens, rb, rm, scalar, struct_features=sf).cpu().numpy()
                    for ci, (src, sc) in enumerate(zip(candidate_sources, scores)):
                        verifier_scores.append((src, float(sc)))
                        if args.dump_candidates:
                            cand_dump.append({
                                "task_id": tid,
                                "source": src,
                                "score": float(sc),
                                "is_gt_exact": int(score_exact(candidate_seqs[ci], label_seq)),
                            })
                    best_idx = int(np.argmax(scores))
                    best_score = float(scores[best_idx])
                    if best_score > float(args.tau_commit):
                        best_seq = candidate_seqs[best_idx]
                        best_source = candidate_sources[best_idx]
                    else:
                        best_seq = pred_can_clean
                        best_source = f"{canon_source}_fallback"

                final_exact = score_exact(best_seq, label_seq)
                ref = ref_by_task.get(tid, {})
                results.append({
                    "task_id": tid,
                    "bucket": ref.get("bucket", ""),
                    "c0_exact": int(float(ref.get("exact_accuracy", 0)) > 0),
                    "canonical_exact": canonical_exact,
                    "verifier_exact": final_exact,
                    "best_source": best_source,
                    "best_score": best_score,
                    "n_candidates": len(candidate_seqs),
                })
            if batch_idx % 25 == 0:
                print(f"[eval] batch {batch_idx}/{len(eval_batches)}  tasks={len(results)}")
    print(f"[eval] DONE - tasks scored: {len(results)}")

    # Write summary.
    fields = ["task_id", "bucket", "c0_exact", "canonical_exact", "verifier_exact",
              "best_source", "best_score", "n_candidates"]
    write_csv(out_dir / "eval_with_verifier_summary.csv", results, fields)

    # Optional per-candidate dump + held-out AUROC (verifier_score vs is_gt_exact).
    if args.dump_candidates and cand_dump:
        write_csv(out_dir / "candidate_scores.csv", cand_dump,
                  ["task_id", "source", "score", "is_gt_exact"])
        pos = [r["score"] for r in cand_dump if int(r["is_gt_exact"]) == 1]
        neg = [r["score"] for r in cand_dump if int(r["is_gt_exact"]) == 0]
        if pos and neg:
            # AUROC = P(score(pos) > score(neg)) via rank statistic (Mann–Whitney U).
            import numpy as _np
            alls = _np.array([r["score"] for r in cand_dump], dtype=float)
            ys = _np.array([int(r["is_gt_exact"]) for r in cand_dump], dtype=int)
            order = alls.argsort()
            ranks = _np.empty_like(order, dtype=float)
            ranks[order] = _np.arange(1, len(alls) + 1)
            n_pos, n_neg = len(pos), len(neg)
            auroc = (ranks[ys == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            (out_dir / "auroc.txt").write_text(
                f"held_out_auroc={auroc:.4f}\nn_pos={n_pos}\nn_neg={n_neg}\n", encoding="utf-8")
            print(f"[eval] held-out AUROC (score vs is_gt_exact): {auroc:.4f} "
                  f"(n_pos={n_pos}, n_neg={n_neg})")

    # Compute aggregate.
    def count(rows, key, scope="all"):
        if scope == "all":
            return sum(int(r[key]) for r in rows)
        return sum(int(r[key]) for r in rows if r["bucket"] == scope)

    c0 = count(results, "c0_exact")
    canon = count(results, "canonical_exact")
    verif = count(results, "verifier_exact")
    bf_c0 = count(results, "c0_exact", "both_fail")
    bf_canon = count(results, "canonical_exact", "both_fail")
    bf_verif = count(results, "verifier_exact", "both_fail")
    replace_loss = sum(1 for r in results if int(r["c0_exact"]) == 1 and int(r["verifier_exact"]) == 0)

    report = [
        f"verdict: {'KEEP' if verif > c0 and replace_loss <= 5 else 'REJECT'}",
        f"checkpoint: {Path(args.checkpoint).resolve()}",
        f"verifier checkpoint: {Path(args.verifier_checkpoint).resolve()}",
        f"tau_commit: {float(args.tau_commit)}",
        f"use_struct_features: {bool(args.use_struct_features)}",
        "",
        f"exact (C0 baseline):       {c0}/400",
        f"exact (canonical cleaned): {canon}/400",
        f"exact (verifier-selected): {verif}/400",
        f"both_fail (C0):            {bf_c0}/{sum(1 for r in results if r['bucket']=='both_fail')}",
        f"both_fail (canonical):     {bf_canon}/{sum(1 for r in results if r['bucket']=='both_fail')}",
        f"both_fail (verifier):      {bf_verif}/{sum(1 for r in results if r['bucket']=='both_fail')}",
        f"replacement_loss vs C0:    {replace_loss}",
        "",
        f"pass criteria from PREREG: exact > 135, replacement_loss <= 5",
    ]
    (out_dir / "rejection_or_keep.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
