"""Diagnose a verifier eval from its per-candidate dump (candidate_scores.csv).

Answers:
  1. Overall held-out AUROC (score vs is_gt_exact).
  2. Score separation: mean/median verifier score for correct vs wrong candidates.
  3. Per-source score distribution (canonical_raw vs d4_* confusers).
  4. Ranking quality: per task, did the verifier rank a CORRECT candidate #1?
     - tasks where a correct candidate exists among the K
     - of those, how often argmax(score) is a correct candidate (= would commit right)
  5. The "recoverable" set: tasks where canonical is WRONG but some D4 is correct,
     and whether the verifier's top pick is that correct D4 (the only source of gains).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="candidate_scores.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.candidates, newline="")))
    for r in rows:
        r["score"] = float(r["score"])
        r["is_gt_exact"] = int(r["is_gt_exact"])

    n = len(rows)
    pos = [r["score"] for r in rows if r["is_gt_exact"] == 1]
    neg = [r["score"] for r in rows if r["is_gt_exact"] == 0]
    print(f"candidates: {n}  correct: {len(pos)}  wrong: {len(neg)}")

    # AUROC via Mann-Whitney rank statistic.
    if pos and neg:
        order = sorted(range(n), key=lambda i: rows[i]["score"])
        ranks = [0.0] * n
        for rank, i in enumerate(order, start=1):
            ranks[i] = rank
        sum_pos_ranks = sum(ranks[i] for i in range(n) if rows[i]["is_gt_exact"] == 1)
        npos, nneg = len(pos), len(neg)
        auroc = (sum_pos_ranks - npos * (npos + 1) / 2) / (npos * nneg)
        print(f"held-out AUROC: {auroc:.4f}")
    print(f"score(correct):  mean={_mean(pos):+.4f} median={_median(pos):+.4f}  range=[{_min(pos):+.3f},{_max(pos):+.3f}]")
    print(f"score(wrong):    mean={_mean(neg):+.4f} median={_median(neg):+.4f}  range=[{_min(neg):+.3f},{_max(neg):+.3f}]")

    # Per-source.
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    print("\nper-source mean score (and correct-rate):")
    for src in sorted(by_src):
        s = by_src[src]
        sc = _mean([r["score"] for r in s])
        cr = _mean([r["is_gt_exact"] for r in s])
        print(f"  {src:24s} n={len(s):4d}  mean_score={sc:+.4f}  correct_rate={cr:.3f}")

    # Per-task ranking.
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)

    tasks_with_correct = 0
    argmax_is_correct = 0
    canonical_wrong_d4_right = 0
    recovered = 0
    for tid, cands in by_task.items():
        has_correct = any(c["is_gt_exact"] == 1 for c in cands)
        if not has_correct:
            continue
        tasks_with_correct += 1
        top = max(cands, key=lambda c: c["score"])
        if top["is_gt_exact"] == 1:
            argmax_is_correct += 1
        # canonical wrong but a d4 right?
        canon = [c for c in cands if c["source"].startswith("canonical")]
        canon_correct = bool(canon) and canon[0]["is_gt_exact"] == 1
        d4_correct = any(c["is_gt_exact"] == 1 and c["source"].startswith("d4_") for c in cands)
        if (not canon_correct) and d4_correct:
            canonical_wrong_d4_right += 1
            if top["is_gt_exact"] == 1 and top["source"].startswith("d4_"):
                recovered += 1

    print(f"\nranking quality (tasks where SOME candidate is correct): {tasks_with_correct}")
    if tasks_with_correct:
        print(f"  argmax(score) is a correct candidate: {argmax_is_correct}/{tasks_with_correct} "
              f"({argmax_is_correct/tasks_with_correct:.1%})")
    print(f"\nGAIN POTENTIAL — canonical wrong but a D4 is correct: {canonical_wrong_d4_right} tasks")
    print(f"  of those, verifier's top pick IS that correct D4 (a real gain): {recovered}")
    print(f"\nLOSS RISK — tasks where canonical IS correct:")
    canon_correct_tasks = 0
    canon_correct_but_top_wrong = 0
    for tid, cands in by_task.items():
        canon = [c for c in cands if c["source"].startswith("canonical")]
        if canon and canon[0]["is_gt_exact"] == 1:
            canon_correct_tasks += 1
            top = max(cands, key=lambda c: c["score"])
            if top["is_gt_exact"] == 0:
                canon_correct_but_top_wrong += 1
    print(f"  canonical-correct tasks: {canon_correct_tasks}")
    print(f"  of those, verifier's top pick is WRONG (would lose at tau=0): {canon_correct_but_top_wrong}")


def _mean(xs): return sum(xs) / len(xs) if xs else float("nan")
def _min(xs): return min(xs) if xs else float("nan")
def _max(xs): return max(xs) if xs else float("nan")
def _median(xs):
    if not xs: return float("nan")
    s = sorted(xs); m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m-1] + s[m]) / 2


if __name__ == "__main__":
    main()
