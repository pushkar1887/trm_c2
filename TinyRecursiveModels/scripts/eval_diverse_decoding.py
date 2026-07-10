"""Diverse-decoding eval: sample K candidates per task from a frozen TRM and
measure the Oracle@K ceiling and Vote@K (self-consistency) accuracy.

Motivation: the verifier+D4 mechanism is capped at ~127/400 because D4 transforms
of a wrong prediction are still wrong. Temperature sampling on the final-step logits
produces CONTENT-different candidates (it perturbs the cells the model is unsure about).
This script measures whether that headroom exists:

  canonical (greedy argmax)   — the K=1 baseline (expected 125/400 on aligned data)
  Oracle@K                    — fraction of tasks where ANY of K samples is exact
                                (the ceiling a perfect selector could reach)
  Oracle@K+greedy             — greedy ∪ K samples (true reachable ceiling)
  Vote@K                      — per-token majority vote over K samples, then exact
                                (a selector-free self-consistency heuristic)

Since the TRM forward is deterministic, all K samples are drawn from ONE forward's
final logits (drawing K times from the same distribution == K independent decodes).

Usage:
  trm/Scripts/python.exe scripts/eval_diverse_decoding.py \
    --config <all_config.yaml> --checkpoint <ckpt> \
    --reference-ledger <c0 ledger.csv> \
    --dataset data/arc-agi-evaluation-full400-seed0-pid401aligned \
    --out-dir reports/diverse_decoding_T1.0_K8 \
    --temperature 1.0 --num-samples 8 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain  # noqa: E402
from scripts.fvr_structfuse_alpha_sweep import IGNORE_LABEL_ID, read_csv, write_csv  # noqa: E402


def score_exact(pred_seq: np.ndarray, label_seq: np.ndarray) -> int:
    mask = label_seq != IGNORE_LABEL_ID
    return int(np.array_equal(pred_seq[mask], label_seq[mask]))


def main() -> None:
    p = argparse.ArgumentParser(description="Diverse-decoding Oracle@K / Vote@K eval.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference-ledger", required=True)
    p.add_argument("--dataset", default="data/arc-agi-evaluation-full400-seed0-pid401aligned")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--capture-steps", action="store_true", default=False,
                   help="Also collect the argmax prediction at every reasoning step "
                        "(coherent diversity) and report Oracle over the trajectory.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required.")

    # Load config + model (mirror eval_with_verifier setup).
    raw = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    raw["load_checkpoint"] = str(Path(args.checkpoint).resolve())
    raw["data_paths"] = [args.dataset]
    raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0
    raw["checkpoint_path"] = str(out_dir / "noop_checkpoints")
    raw["run_name"] = "eval_diverse_decoding"
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
    eval_batches = [{k: v for k, v in batch.items()} for _s, batch, _g in eval_loader]
    print(f"[eval] cached eval batches: {len(eval_batches)}")

    loss_head, _, _ = pretrain.create_model(config, train_meta, rank=0, world_size=1)
    core_model = loss_head.model
    core_model.eval()
    for p_ in core_model.parameters():
        p_.requires_grad_(False)
    core_model.to(device)

    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset)
    if not dataset_root.is_absolute():
        dataset_root = repo_root / dataset_root
    eval_ids = json_load(dataset_root / "identifiers.json")
    ref_by_task = {r["task_id"]: r for r in read_csv(Path(args.reference_ledger).resolve())}

    halt_max_steps = int(config.arch.halt_max_steps)
    T = float(args.temperature)
    K = int(args.num_samples)

    results: List[Dict[str, object]] = []
    seen = set()
    with torch.inference_mode():
        for bi, cpu_batch in enumerate(eval_batches, start=1):
            batch = {k: v.to(device) for k, v in cpu_batch.items()}
            with torch.device(device.type):
                carry = core_model.initial_carry(batch)
            out = None
            step_greedy = []   # coherent argmax prediction at each reasoning step
            for _step in range(halt_max_steps):
                carry, out = core_model(carry=carry, batch=batch)
                if args.capture_steps:
                    step_greedy.append(torch.argmax(out["logits"], dim=-1).cpu().numpy())  # [B,900]
            logits = out["logits"]            # [B, 900, V]
            greedy = torch.argmax(logits, dim=-1)   # [B, 900]
            puzzle_ids = batch["puzzle_identifiers"].cpu().numpy().tolist()
            labels = batch["labels"].cpu().numpy()

            # Temperature-sampled candidates: [B, 900, K]
            probs = torch.softmax(logits / max(T, 1e-6), dim=-1)  # [B,900,V]
            B, L, V = probs.shape
            flat = probs.reshape(B * L, V)
            samp = torch.multinomial(flat, num_samples=K, replacement=True)  # [B*L, K]
            samp = samp.reshape(B, L, K)

            greedy_np = greedy.cpu().numpy()
            samp_np = samp.cpu().numpy()

            for row in range(B):
                pid = int(puzzle_ids[row])
                if pid <= 0 or pid >= len(eval_ids):
                    continue
                tid = eval_ids[pid]
                if tid in seen:
                    continue
                seen.add(tid)
                label_seq = labels[row]
                g = greedy_np[row]                  # [900]
                samples = [samp_np[row, :, k] for k in range(K)]  # K x [900]

                canon_exact = score_exact(g, label_seq)
                sample_exacts = [score_exact(s, label_seq) for s in samples]
                oracle_k = int(any(sample_exacts))
                oracle_kg = int(canon_exact or oracle_k)

                # Oracle over the reasoning trajectory (coherent intermediate predictions).
                oracle_steps = 0
                if args.capture_steps and step_greedy:
                    oracle_steps = int(any(score_exact(sg[row], label_seq) for sg in step_greedy))

                # Vote@K: per-token majority across K samples.
                stk = np.stack(samples, axis=0)     # [K, 900]
                vote = np.zeros(L, dtype=stk.dtype)
                for j in range(L):
                    vals, counts = np.unique(stk[:, j], return_counts=True)
                    vote[j] = vals[int(np.argmax(counts))]
                vote_exact = score_exact(vote, label_seq)

                # Vote@K+greedy: include greedy in the vote pool (greedy gets 1 vote).
                stk_g = np.concatenate([stk, g[None, :]], axis=0)
                vote_g = np.zeros(L, dtype=stk_g.dtype)
                for j in range(L):
                    vals, counts = np.unique(stk_g[:, j], return_counts=True)
                    vote_g[j] = vals[int(np.argmax(counts))]
                vote_g_exact = score_exact(vote_g, label_seq)

                ref = ref_by_task.get(tid, {})
                results.append({
                    "task_id": tid,
                    "bucket": ref.get("bucket", ""),
                    "canonical_exact": canon_exact,
                    "oracle_k": oracle_k,
                    "oracle_k_greedy": oracle_kg,
                    "vote_k": vote_exact,
                    "vote_k_greedy": vote_g_exact,
                    "oracle_steps": oracle_steps,
                    "n_distinct_samples": len({s.tobytes() for s in samples}),
                })
            if bi % 25 == 0:
                print(f"[eval] batch {bi}/{len(eval_batches)} tasks={len(results)}")
    print(f"[eval] DONE tasks={len(results)}")

    write_csv(out_dir / "diverse_decoding_summary.csv", results,
              ["task_id", "bucket", "canonical_exact", "oracle_k", "oracle_k_greedy",
               "vote_k", "vote_k_greedy", "oracle_steps", "n_distinct_samples"])

    def total(key, b=None):
        return sum(int(r[key]) for r in results if (b is None or r["bucket"] == b))

    buckets = sorted({r["bucket"] for r in results})
    n_tasks = len(results)
    mean_distinct = np.mean([int(r["n_distinct_samples"]) for r in results]) if results else 0.0

    lines = [
        f"# Diverse decoding — T={T}, K={K}, seed={args.seed}",
        f"dataset: {args.dataset}",
        f"checkpoint: {Path(args.checkpoint).resolve()}",
        f"tasks scored: {n_tasks}",
        f"mean distinct samples per task (of {K}): {mean_distinct:.2f}",
        "",
        f"canonical (greedy):   {total('canonical_exact')}/{n_tasks}",
        f"Oracle@{K}:            {total('oracle_k')}/{n_tasks}   (any sample exact)",
        f"Oracle@{K}+greedy:     {total('oracle_k_greedy')}/{n_tasks}   (CEILING for a selector)",
        f"Vote@{K}:              {total('vote_k')}/{n_tasks}",
        f"Vote@{K}+greedy:       {total('vote_k_greedy')}/{n_tasks}",
    ]
    if args.capture_steps:
        lines.append(f"Oracle@steps:          {total('oracle_steps')}/{n_tasks}   "
                     f"(any of {halt_max_steps} reasoning-step predictions exact)")
    lines += [
        "",
        "per-bucket (canonical / Oracle@K+greedy / Vote@K+greedy):",
    ]
    for b in buckets:
        nb = sum(1 for r in results if r["bucket"] == b)
        lines.append(f"  {b:12s} n={nb:3d}  canon={total('canonical_exact', b):3d}  "
                     f"oracle={total('oracle_k_greedy', b):3d}  vote={total('vote_k_greedy', b):3d}")
    # both_fail headline
    bf_canon = total("canonical_exact", "both_fail")
    bf_oracle = total("oracle_k_greedy", "both_fail")
    lines += [
        "",
        f"both_fail headroom: canonical {bf_canon} -> Oracle@K+greedy {bf_oracle} "
        f"(+{bf_oracle - bf_canon} reachable)",
        "",
        f"D4 reference ceiling was 127/400. Oracle@K+greedy here: {total('oracle_k_greedy')}/{n_tasks}.",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)


def json_load(path: Path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
