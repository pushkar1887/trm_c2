"""FILE #6 (V2 rewrite): structured, multi-GPU-ready training/probe driver.

Rewrite of scripts/run_stage1_local.py (the 1714-line single-main() oracle, UNTOUCHED — it stays
the equivalence oracle). Same flags, same behaviour, same panel numbers (gate G3); the monolith is
decomposed into module-level sections so every phase is testable and rank-gateable.

SECTIONS
  §1 CONSTANTS            §2 ARGS (build_parser)       §3 CONFIG BUILD (build_config)
  §4 DISTRIBUTED          §5 MODEL + SCOPE             §6 OPTIMIZER
  §7 EVAL + PROBES        §8 PANEL                     §9 TRAIN LOOP + main()
  §10 LEGACY-GATED        (dead lanes kept runnable behind their flags, never silently)

PRE-REGISTERED DIFFERENCES from the oracle (each is a ledgered DRV fix, invisible on all
ledger commands):
  DRV-1  panel V2TAIL reads the evidence offset from trm_fvr_v2.evidence_slice (schema-owned)
         instead of hand-summed widths imported from the OLD modules. Same number today (G3).
  DRV-4  --per-family-eval setup save/restores the global torch/numpy/random RNG state, so a
         measurement flag no longer perturbs the TRAINING batch stream.
  DRV-5  the lr-group split is UNCONDITIONAL (the oracle skipped it when struct_lr==evidence_lr==
         --lr, silently parking color_evidence_proj in the wd=0.01 default group: lever-erosion
         reappearance #4). Observable only at --lr 1e-3 exactly.
  DRV-2  eval_fixed warns (once) if a metric key is missing from some frozen batches (the
         macro-average silently deflates such keys). Detection only; math unchanged.
  +      default --config is the V2 base yaml (the legacy default re-armed the V2-GUARD trap);
         --data overrides the dataset path (Kaggle); --dump-config prints the resolved config and
         exits (gate G1 hook). Explicit --config keeps every oracle comparison exact.

LAUNCH
  single GPU (Windows local):
    trm\\Scripts\\python.exe scripts\\test_run_v2.py --c2-relmap --v3-clean ...
  multi-GPU (Linux/Kaggle 2xT4 -- nccl; Windows/CPU smoke -- gloo, picked automatically):
    torchrun --nproc_per_node=2 scripts/test_run_v2.py --c2-relmap --v3-clean ...
    WINDOWS ONLY: this torch build lacks libuv and torchrun's own rendezvous store IGNORES
    USE_LIBUV=0 (worker-side env:// init respects it). Launch workers manually instead:
    per rank r: set RANK=r LOCAL_RANK=r WORLD_SIZE=N MASTER_ADDR=127.0.0.1 MASTER_PORT=29531
    USE_LIBUV=0, then run this script N times concurrently (the G5 gate does exactly this).
  NOTE: --batch is the GLOBAL batch size; pretrain divides it per rank (batch 8 @ world 2
  = 4 rows/rank). Eval/probes/checkpoint run on rank 0; guards + aborts are rank-AGREED.
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pretrain  # noqa: E402
from models.color_perm import apply_color_perm  # noqa: E402

# ======================================================================================
# §1 CONSTANTS
# ======================================================================================
# DEFAULT CONFIG FLIP (plan decision 6): this is the V2 driver; defaulting to the legacy
# trm_fvr_c2 yaml would re-arm the V2-GUARD trap on every plain run. The oracle's legacy
# default stays available as LEGACY_CONFIG / an explicit --config.
CONFIG = "config/new_fvr_full_v2_base.yaml"
LEGACY_CONFIG = "checkpoints/TRM-FVR-Experiments/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/all_config.yaml"
CKPT = r"D:\trm_c2\step_518071"
DATASET = r"D:\trm_c2\arc1concept-aug-1000"
PROVENANCE_SOURCE_FILES = (
    "models/recursive_reasoning/trm_fvr_v2.py",
    "models/losses_fvr.py",
    "models/recursive_reasoning/relation_map.py",
    "models/recursive_reasoning/core_prior.py",
    "models/recursive_reasoning/pair_delta_v2.py",
    "puzzle_dataset.py",
    "pretrain.py",
    "scripts/test_run_v2.py",
)


# ======================================================================================
# §2 ARGS — every oracle flag VERBATIM (names + defaults; gate G0), grouped for --help.
# New flags: --data, --dump-config (G1 hook). --config default flipped (see §1).
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="V2 structured training/probe driver (File #6)")

    g = ap.add_argument_group("run")
    g.add_argument("--config", type=str, default=CONFIG,
                   help="Base all_config yaml. DEFAULT = the V2 base (trm_fvr_v2). Pass the legacy "
                        "all_config.yaml (see LEGACY_CONFIG in this file) to run the old trm_fvr_c2 "
                        "model; the V2-only evidence flags then refuse via the V2-GUARD.")
    g.add_argument("--data", type=str, default=DATASET,
                   help="Training dataset path (default: the local aug-1000). Kaggle/Stage runs "
                        "override this instead of editing the file.")
    g.add_argument("--dump-config", action="store_true",
                   help="Resolve args -> (raw, arch, loss), print as yaml, exit 0 BEFORE any "
                        "dataloader/model work. Gate G1 hook + fast command sanity check.")
    g.add_argument("--no-mextra", action="store_true",
                   help="Disable ALL Block-6 metric lines ([START]/[FEED]/[PRESS]/[TREND]/"
                        "[RUN ]/[PERF]/[SCORE]/[MOVERS]/[CONTRACT]) -- the oracle-equivalence "
                        "escape hatch: with this flag the stdout is the verbatim old panel.")
    g.add_argument("--score-rows", type=int, default=48,
                   help="Row cap for the [SCORE] static component scorecard (cost control: the "
                        "verified-frame builder runs a propose/verify walk per row). 0 disables "
                        "SCORE entirely (static + dynamic).")
    g.add_argument("--steps", type=int, default=300)
    g.add_argument("--epochs", type=float, default=None,
                   help="Full-dataset mode: override --steps with the pretrain.py step formula "
                        "epochs * total_groups * mean_puzzle_examples / batch.")
    g.add_argument("--save-checkpoint-dir", type=str, default="",
                   help="If set, save final model state_dict as step_N plus all_config.yaml in this directory.")
    g.add_argument("--save-checkpoint-steps", type=str, default="",
                   help="Comma-separated completed update counts to save before the final checkpoint, e.g. 300. "
                        "Requires --save-checkpoint-dir; empty preserves final-only behavior.")
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--log-every", type=int, default=10)
    g.add_argument("--eval-batches", type=int, default=8,
                   help="how many FIXED batches to measure on each log step (stable LODO eval). "
                        "0.4: default raised 4->8 -- 2 batches made LODO swing +-25pct, hiding trends.")
    g.add_argument("--eval-seed", type=int, default=1234,
                   help="seed for the fixed-eval holdout so the SAME held-out demos are scored every step")
    g.add_argument("--ckpt", type=str, default=CKPT,
                   help="warm-start checkpoint. DEFAULT is the bare GitHub TRM (c2/structure RANDOM!). "
                        "Pass the trained step_1631 to load trained c2/structure/delta so LODO has a real canvas.")

    g = ap.add_argument_group("paired evaluation / diagnostics (Block 1/2)")
    g.add_argument("--task-metrics-out", type=str, default="",
                   help="write ONE CSV row per fixed-eval example (WHERE/bind/candidate per-ex metrics) "
                        "for paired analysis. Also drives the mechanism-conditioned-exact panel. Empty = off.")
    g.add_argument("--paired-control-report", type=str, default="",
                   help="control CSV (a previous run's --task-metrics-out) to compare THIS run's "
                        "--task-metrics-out against: paired per-case deltas + family-stratified bootstrap CI.")
    g.add_argument("--where-gradient-probe-only", action="store_true",
                   help="P3A Block 4: measure grad norms of the WHERE loss vs the RAW support "
                        "contrast on the v3-where-selector parameters over a few batches, recommend "
                        "lambda_support, write JSON, and EXIT. No optimizer step ever runs.")
    g.add_argument("--where-gradient-probe-steps", type=int, default=3,
                   help="Batches to probe (default 3).")
    g.add_argument("--where-gradient-target-ratio", type=float, default=0.5,
                   help="Desired weighted-support/WHERE gradient ratio (default 0.5); "
                        "lambda_support = ratio * G_where / (G_support + 1e-12).")
    g.add_argument("--where-gradient-probe-out", type=str, default="",
                   help="JSON output path (default: <task-metrics-out stem>_where_probe.json, "
                        "else where_gradient_probe.json in the CWD).")
    g.add_argument("--paired-bootstrap-samples", type=int, default=10000,
                   help="paired bootstrap resamples for the control-report CI.")
    g.add_argument("--paired-bootstrap-seed", type=int, default=1234,
                   help="seed for the paired bootstrap resampling (reproducible CI).")
    g.add_argument("--fusion-compression-warn", type=float, default=5.0,
                   help="Block 3: warn when a bounded-fusion lane's raw/applied norm-ratio compression "
                        "exceeds this for --fusion-compression-patience consecutive evals (the rho clamp "
                        "is discarding most of that evidence residual). Diagnostic only -- NEVER aborts. "
                        "<=0 disables the warning (the compression line still prints).")
    g.add_argument("--fusion-compression-patience", type=int, default=3,
                   help="consecutive evals above --fusion-compression-warn before [FUSION-WARN] fires "
                        "(keeps firing while the condition persists; resets on any eval below threshold).")

    g = ap.add_argument_group("losses / stages")
    g.add_argument("--preserve", type=float, default=1.0, help="c2_delta_preserve_weight (Stage 2 KL)")
    g.add_argument("--changed-valid", type=float, default=0.0, help="c2_changed_valid_loss_weight (Stage 3)")
    g.add_argument("--lodo-weight", type=float, default=0.05, help="final gated-output LODO CE weight after warmup")
    g.add_argument("--lodo-pad-weight", type=float, default=0.0,
                   help="PAD term inside c2_delta_lodo canvas CE. Keep tiny under --floor-candidate-split; "
                        "0 isolates colour.")
    g.add_argument("--lodo-eos-weight", type=float, default=0.0,
                   help="EOS term inside c2_delta_lodo canvas CE. Keep tiny under --floor-candidate-split; "
                        "0 isolates colour. NOTE: eos is ~68 cells -> a high weight (3.0) over-predicts the "
                        "frame and EATS valid cells (collapses shape_task). ~1.0 is the balanced value.")
    g.add_argument("--lodo-copy-weight", type=float, default=1.0,
                   help="COPY (unchanged-colour) term inside c2_delta_lodo canvas CE = c2_delta_color_weight. "
                        "RAISE this (~3-4) to PROTECT valid cells from being flipped to PAD by the boundary CE "
                        "(the copy cells fold first -> shape_task collapse). Default 1.0 = prior behaviour.")
    g.add_argument("--lodo-changed-weight", type=float, default=5.0,
                   help="CHANGED (transform) term inside c2_delta_lodo canvas CE = c2_delta_changed_weight. "
                        "Default 5.0 = prior behaviour; the transform cells are the task, keep them up-weighted.")
    g.add_argument("--lodo-warmup-steps", type=int, default=50, help="steps before ramping final gated-output LODO CE")
    g.add_argument("--lodo-ramp-steps", type=int, default=50, help="linear ramp length for final gated-output LODO CE")
    g.add_argument("--contrast-per-row", action="store_true",
                   help="Per-task changed-cell contrast hinge instead of the batch-scalar hinge: each LODO row "
                        "is hinged separately and rows WITHOUT a valid wrong-task shuffle are masked (the "
                        "batch-scalar form dilutes the gap and leaks gradient through same-task shuffles).")
    g.add_argument("--contrast-weight", type=float, default=1.0,
                   help="Final two-region contrast weight. SCHEDULED with the LODO warmup/ramp (was fixed at "
                        "1.0 from step 0 -- full-strength contrast on RANDOM-init adapters drifted z_H and "
                        "broke MAIN strict before the LODO CE even started; measured: strict 100->70 by step "
                        "90 with lodo_w still ramping).")
    g.add_argument("--value-v2-aux-weight", type=float, default=0.0,
                   help="Explicit LODO loss on the VALUE-V2-only color_head contribution. This forces the "
                        "new evidence columns to become predictive instead of being ignored by the full "
                        "colour head. 0 = off.")
    g.add_argument("--value-v2-aux-changed-weight", type=float, default=1.0,
                   help="Changed-cell weight inside the VALUE-V2-only auxiliary CE.")
    g.add_argument("--value-v2-aux-copy-weight", type=float, default=1.0,
                   help="Copy-cell weight inside the VALUE-V2-only auxiliary CE.")
    g.add_argument("--bind-aux-weight", type=float, default=0.0,
                   help="Direct LODO CE weight on the existing value_ctx_bind evidence slice. 0 = off.")
    g.add_argument("--bind-aux-changed-weight", type=float, default=3.0,
                   help="Changed-cell weight inside the support-masked bind auxiliary CE.")
    g.add_argument("--bind-aux-copy-weight", type=float, default=2.0,
                   help="Copy-cell weight inside the support-masked bind auxiliary CE.")
    g.add_argument("--bind-residual-aware", action="store_true",
                   help="Repair B: train the canonical bind slice against detached candidate colour "
                        "logits with that slice removed. Changed-supported cells train; copy-supported "
                        "cells are diagnostics only and remain protected by ordinary LODO/preservation.")
    g.add_argument("--gate-where-weight", type=float, default=0.0,
                   help="LODO changed/copy regression weight for --token-gate-where. 0 = off.")
    g.add_argument("--gate-support-contrast-weight", type=float, default=0.0,
                   help="Hinge weight requiring correct-support gate WHERE loss to beat shuffled support.")
    g.add_argument("--shape-weight", type=float, default=0.3,
                   help="c2_shape_loss_weight for --cross-demo-shape (CE sum over batch; 0.3 = gentle start, "
                        "raise if c2_shape_exact is flat, lower if it perturbs MAIN).")

    g = ap.add_argument_group("selector / candidate lanes")
    g.add_argument("--selector-eval", action="store_true",
                   help="VERIFY-AND-SELECT (Take #1 / composition-C): eval-only floor-safe selector. Per task "
                        "scores the two model candidates -- gg=0 (deterministic floor) vs gg=on (head) -- by LODO "
                        "reconstruction, commits the best (tie->floor), reports the selector>=floor GUARANTEE + "
                        "head-vs-floor + selection headroom. No training/model change.")
    g.add_argument("--floor-candidate-split", action="store_true",
                   help="V3 safety split: MAIN/logits use lm_head FLOOR, while LODO trains/scores the factored "
                        "V3 candidate. Selector metrics compare candidate vs floor and tie to floor.")
    g.add_argument("--candidate-floor-structure", action="store_true",
                   help="With --floor-candidate-split, build the candidate from FLOOR PAD/EOS/VALID structure "
                        "plus V3 colour values on floor-valid cells. This tests colour improvement without "
                        "asking the candidate to relearn canvas structure.")
    g.add_argument("--quarantine-candidate", action="store_true",
                   help="PID-QUARANTINED candidate colour head: a small MLP over EXCLUSIVELY PID-free "
                        "evidence (input one-hot + 3x3 neighbourhood + transition hint + rel-where + "
                        "palette + intent + relmap; z_H never read) supplies the CANDIDATE colour choice; "
                        "the canvas stays the solved floor structure. Closes the three measured cheats by "
                        "construction: cannot memorise (no PID), colour-perm leaves only table-reading, "
                        "and with --train-scope quarantine z_H is not in the gradient path (MAIN risk "
                        "structurally zero -- no warmup needed, lodo weight can be 1.0). Warm-init = "
                        "copy-unless-consensus (step-0 == the D1 deterministic baseline). Needs "
                        "--floor-candidate-split; pair with --candidate-floor-structure + --color-perm.")
    g.add_argument("--quarantine-hidden", type=int, default=256,
                   help="Hidden dim of the quarantine head's MLP residual (c2_quarantine_hidden).")

    g = ap.add_argument_group("training pressure / augmentation")
    g.add_argument("--color-perm", action="store_true",
                   help="Phase 1: per-task colour-palette permutation on TRAINING batches "
                        "(anti-memorisation; fixed-eval stays canonical). MAIN WILL DROP -- that is "
                        "the success signal, not a regression. Use with --no-abort-on-main-drop.")
    g.add_argument("--pid-dropout", type=float, default=0.0,
                   help="NECESSITY PRESSURE (C-prime): fraction of TRAINING-batch rows whose puzzle_identifier "
                        "is replaced by the blank PID(0), so memorisation cannot reduce their MAIN CE and "
                        "reading the demos becomes the only remaining gradient. Runner-side + per-row: the "
                        "frozen eval batches are NEVER dropped, so the panel/abort guard stay honest. Read "
                        "the [pid-null] report: null-pid MAIN strict rising = the pressure is working. "
                        "Use with --unified (a frozen core cannot re-route around a missing PID) + --color-perm.")
    g.add_argument("--pid-null-eval", action="store_true",
                   help="Print the [pid-null] diagnostic (MAIN strict with true PID vs blank PID(0) on the "
                        "frozen eval batches) at baseline+end even without --pid-dropout. null-pid strict "
                        "IS demo-only performance -- the honest generalization number.")
    g.add_argument("--router-gate", action="store_true",
                   help="Phase 3: gate the colour-repair head by a per-task self-consistency score "
                        "(demo-only). CLEAN recolours fire the head; relational/structural tasks shut it. "
                        "Validated: accept>=0.9 -> 88pct coverage vs reject -> 24pct.")
    g.add_argument("--router-threshold", type=float, default=0.9,
                   help="Phase 3: router fires fully at/above this self-consistency; shuts below (band 0.2).")
    g.add_argument("--palette-constrain", action="store_true",
                   help="Phase 3: penalise output colours absent from the input grid (no-invention prior, "
                        "97pct on relational), strength=(1-router_mult) so CLEAN tasks stay free. Pre-wired "
                        "for the relational path; dormant while relational tasks are router-shut.")

    g = ap.add_argument_group("palette")
    g.add_argument("--task-palette", action=argparse.BooleanOptionalAction, default=False,
                   help="V3 colour-head task palette: expose support-in/out + target-input palette as a "
                        "color_head feature AND apply a soft absent-colour bias. Default off so matched "
                        "controls have no hidden evidence; enable explicitly or use --task-palette-bias "
                        "for the ordered probes' bias-only contract.")
    g.add_argument("--task-palette-feature", action="store_true",
                   help="Ablation: expose the task palette to color_head input without applying the logit bias.")
    g.add_argument("--task-palette-bias", action="store_true",
                   help="Ablation: apply the task-palette absent-colour bias without widening color_head input.")
    g.add_argument("--task-palette-strength", type=float, default=4.0,
                   help="Soft logit penalty for colours absent from support inputs/outputs and target input.")
    g.add_argument("--task-palette-hard", action="store_true",
                   help="Hard-mask absent task-palette colours. Use only after target coverage is proven safe.")

    g = ap.add_argument_group("evidence (both models)")
    g.add_argument("--rel-where-hint", action="store_true",
                   help="Fold the PASSED ObjectBank/relmap WHERE evidence into the existing V3 color_head "
                        "as an input feature only. No VALUE writer, no candidate acceptance, no PAD/EOS edits.")
    g.add_argument("--rel-where-topk", type=int, default=1,
                   help="FIX D: expose the top-K rel-where predicate masks (each scaled by its score) as "
                        "evidence columns instead of the single hard winner. 1 = legacy. The WHERE gate "
                        "used by value-v2/quarantine stays channel 0, so their semantics never change.")
    g.add_argument("--algo-where-maps", action="store_true",
                   help="FIX C: 21 algorithmic WHERE/VALUE evidence columns from cell_conditioning_signature "
                        "cols 11/12 -- enclosed flood-fill flag + enclosing-colour one-hot + nearest-seed-"
                        "colour one-hot. Separate zero-init color_evidence_proj columns, never a writer.")
    g.add_argument("--algo-where-touch", action="store_true",
                   help="D6 (B13 close-out): 14 touch-colour evidence columns from cell_conditioning_signature "
                        "cols 14/15 -- touch_colour_mode one-hot (10) + distinct-count bucket one-hot (4). "
                        "The neighbour-conditioned recolor substrate; target-input-only, zero-init, never a writer.")
    g.add_argument("--pairdelta-intent-hint", action="store_true",
                   help="Fold cheap PairDelta intent diagnostics into the existing V3 color_head as a "
                        "per-task input feature only. PairDelta remains a router/evidence signal, not a solver.")
    g.add_argument("--transition-hint", action="store_true",
                   help="VALUE binding: per-cell demo-consensus P(out_colour|in_colour) over CHANGED support "
                        "cells, gathered by each target cell's input colour -> 10 zero-init color_head "
                        "columns. Routes the transition evidence DIRECTLY to the colour decision (one linear "
                        "layer), bypassing the frozen recurrence that attenuates input-side demo injections. "
                        "LODO-safe (active-context only), F7-safe (zero-init), never a writer.")
    g.add_argument("--value-evidence-v2", action="store_true",
                   help="Richer VALUE evidence for the existing color_head: copy-vs-change rates plus "
                        "context-conditioned changed-colour distribution with rel-WHERE gating. Evidence "
                        "only, zero-init, no CTBank writer, no candidate executor, no PAD/EOS edits.")
    g.add_argument("--value-v2-rich-ctx", action="store_true",
                   help="Use ObjectBank cell_conditioning_signature as VALUE-V2's context key instead of "
                        "the coarse relmap bucket. Requires --value-evidence-v2. Default off; evidence "
                        "only, zero-init, no writer.")
    g.add_argument("--canonical-value-binder", action="store_true",
                   help="Use the single collision-free hierarchical copy/change VALUE binder. This "
                        "suppresses competing support-derived ten-class evidence authorities; palette "
                        "bias remains a final validity constraint.")
    g.add_argument("--value-backoff-tau", type=float, default=3.0,
                   help="Dirichlet backoff strength for --canonical-value-binder (default: 3.0).")

    # --- V2-model-only evidence (default --config already IS the V2 base; against a legacy config
    #     these refuse via the V2-GUARD instead of being silently dropped).
    g = ap.add_argument_group("evidence (V2 model only)")
    g.add_argument("--verified-frame-evidence", action="store_true",
                   help="[V2 model] D1/E-1: verified-frame applied grid as 11 color_evidence_proj columns "
                        "(10 one-hot + conf). The Lane-A exactness proof as evidence; zero-init, F7-safe.")
    g.add_argument("--analogy-evidence", action="store_true",
                   help="[V2 model] D2/E-2: analogy per-cell colour distribution as 11 columns "
                        "(10 dist + conf). Zero-init, F7-safe.")
    g.add_argument("--value-ctx-gate", action="store_true",
                   help="[V2 model] D7: context-conditioned copy/change gate, 2 columns "
                        "P(change|src,ctx)/P(copy|src,ctx) with source-marginal backoff. Zero-init.")
    g.add_argument("--value-v2-backoff", action="store_true",
                   help="[V2 model] D4/M1: collision-free bounded VALUE-V2 context key "
                        "(enclosure x seed x bg, no modulo) instead of the hash-%%512 rich-ctx bucket.")
    g.add_argument("--pairdelta-color-evidence", action="store_true",
                   help="[V2 model] D8/File#5: pair-delta cross-demo agreement + positional WHERE prior "
                        "as 14 color_evidence_proj columns (consensus P(dst|src) counted once-per-demo, "
                        "min change rate over demos, row/col band priors). Zero-init, F7-safe.")
    g.add_argument("--pairdelta-structure-evidence", action="store_true",
                   help="[V2 model] D9/File#5: verified {preserve, transpose, bbox} extent-family masks "
                        "-> zero-init [6->3] structure_pairdelta_proj (families the extent engine's "
                        "{identity, constant, ratio} set cannot express). Rides the struct-lr group.")
    g.add_argument("--pairdelta-bidi-evidence", action="store_true",
                   help="[V2 model] D10/SS7: reverse-direction (y->x) evidence as 4 columns -- "
                        "invertibility (bijection => trust the mapping), per-src deletion rate + "
                        "cross-demo min, dst-mass (is this colour a rule OUTPUT). Zero-init, F7-safe.")
    g.add_argument("--pairdelta-input-conf-gate", action="store_true",
                   help="[V2 model] SS7 reuse: multiply the --pairdelta-input rule_vec broadcast by the "
                        "encoder's own rule_confidence (was dead). Low-signal tasks shrink the hint "
                        "instead of feeding the norm growth that breaks MAIN-ON.")
    g.add_argument("--pairdelta-include-identity", action="store_true",
                   help="Treat unchanged support demonstrations as explicit negative/identity evidence in "
                        "PairDeltaV2. Default off preserves the legacy encoder contract.")
    g.add_argument("--pairdelta-spatial", action="store_true",
                   help="Fuse colour-agnostic object movement, bbox delta, consistency, and creation/deletion "
                        "features through the existing PairDeltaEncoder. Requires --pairdelta-input; the "
                        "spatial residual is zero-init and remains input evidence only.")
    g.add_argument("--rule-factor-hint", action="store_true",
                   help="Inject the independent core-prior extent/colour/movement/count factor vector "
                        "through the existing bounded input-evidence path.")
    g.add_argument("--object-pair-tokens", action="store_true",
                   help="Append colour-agnostic matched input/output object tokens to existing C2 memory; "
                        "uses shape, area, holes, and centroid correspondence evidence.")
    g.add_argument("--value-ctx-bind", action="store_true",
                   help="[V2 model] D11/codex: 20 EXPLICIT product columns -- change_value[10] = "
                        "P(change|src,ctx)*P(dst|src,ctx), copy_value[10] = P(copy|src,ctx)*one_hot(src). "
                        "The linear evidence proj cannot multiply the D7 gate by the value dist itself; "
                        "this hands it the finished recommendation. Zero-init, F7-safe.")
    g.add_argument("--kinematic-evidence", action="store_true",
                   help="[V2 model] E-5/A3: 7 per-cell kinematic columns -- mover mask + signed (dr,dc) "
                        "(live only under a cross-demo-consistent verified movement binding) + blocked "
                        "up/down/left/right bits (target-input geometry, always on). The rearrangement "
                        "family's per-cell substrate. Zero-init, F7-safe. NOT judged by frozen-core LODO.")
    g.add_argument("--color-mlp", type=int, default=0,
                   help="Interaction capacity for the colour head: adds a zero-init-output MLP residual of "
                        "this hidden dim on the SAME feature concat (c2_color_head_mlp_dim). A linear head "
                        "cannot express input-colour x evidence products ('IF colour a THEN b'); 128 suggested. "
                        "F7-safe: step-0 byte-identical to the warm-started linear head. 0 = off.")

    g = ap.add_argument_group("model lanes / structure levers")
    g.add_argument("--structure-alpha", type=float, default=0.0,
                   help="B2: c2_structure_fusion_alpha - fuse trained structure_head PAD/EOS into the "
                        "output logits. 0.0=aux-only (default); ~0.3 AFTER structure_head trains.")
    g.add_argument("--c2-relmap", action="store_true", help="Enable relational maps")
    g.add_argument("--token-gate-where", action="store_true",
                   help="Make the existing C2 per-token patch gate read target + support cross-attention "
                        "context and expose it to the LODO WHERE objective. Default off.")
    g.add_argument("--positive-where-gate", action="store_true",
                   help="Use sigmoid WHERE selection in [0,1], gate both patch/global C2 updates, and train "
                        "the selector with balanced per-task support supervision.")
    g.add_argument("--gate-selector-detach", action="store_true",
                   help="Repair A: let explicit WHERE supervision train the sigmoid selector while "
                        "blocking candidate colour/transport gradients from rewriting it.")
    g.add_argument("--isolated-relmap-query", action="store_true",
                   help="P3A Block 1: target relmap residual feeds ONLY the C2 query/gate lane "
                        "(x_query); the recurrence keeps the relmap-free x_base. Default off.")
    g.add_argument("--support-interaction-gate", action="store_true",
                   help="P3A Block 2: WHERE gate input = norm(x_query) * norm(patch_context) "
                        "(multiplicative target x support interaction) instead of the additive sum. "
                        "Requires --isolated-relmap-query --positive-where-gate --gate-selector-detach.")
    g.add_argument("--lodo-zero-support", action="store_true",
                   help="P3A Block 3: run a third aux forward per LODO batch with an all-false "
                        "context mask and export the zero-support WHERE gate for the three-arm "
                        "counterfactual (correct/shuffled/zero). Requires --token-gate-where.")
    g.add_argument("--ordered-evidence-flow", action="store_true",
                   help="Compute target/support relations and top-K WHERE masks before C2 so its target "
                        "queries and support memory are relation-aware.")
    g.add_argument("--bounded-evidence-fusion", action="store_true",
                   help="Norm-bound target relmap, C2, and post-C2 evidence residuals instead of adding "
                        "uncontrolled residual magnitudes.")
    g.add_argument("--target-relmap-rho", type=float, default=0.10,
                   help="Maximum target-relmap residual/input norm ratio under bounded fusion.")
    g.add_argument("--c2-update-rho", type=float, default=0.15,
                   help="Maximum C2-update/input norm ratio under bounded fusion.")
    g.add_argument("--post-hint-rho", type=float, default=0.10,
                   help="Maximum post-C2 evidence-hint/input norm ratio under bounded fusion.")
    g.add_argument("--extent-conditioned-structure", action="store_true",
                   help="Blend floor and existing candidate structure using the inferred same-extent "
                        "factor instead of globally forcing floor structure.")
    g.add_argument("--frame-hint", action="store_true",
                   help="Lane B: feed the deterministic solver's verified rearrange-FRAME family as a "
                        "zero-init input-side hint (the rule-hypothesis bus). Dataloader precomputes "
                        "frame_label; the TRM learns the binding. F7-safe (0 at init).")
    g.add_argument("--rule-hypothesis-hint", action="store_true",
                   help="Default-off A/B: run object_rule_bank.infer_rule_hypotheses inside the model, "
                        "embed the top operation family, and add it to grid_features. F7-safe zero-init; "
                        "evidence only, not a solver or output writer.")
    g.add_argument("--relmap-outside-grid", action="store_true",
                   help="§15.9.1: place PAD outside the PREDICTED output box (extent_pad_mask; extent from a "
                        "support-verified demo size-rule {identity,constant,ratio} -- same-shape is just the "
                        "identity rule, size-change is constant/ratio). Dedicated [1->3] proj scaled by conf "
                        "(0 -> floor untouched, no task hurt). Zero-init (F7-safe) unless "
                        "--structure-outside-warm-init.")
    g.add_argument("--structure-outside-warm-init", action="store_true",
                   help="Warm-init the outside_grid PAD row so pad is asserted on the padding from step 0 "
                        "(skips the ~1500-step climb-from-zero). Breaks step-0==floor; frozen-core is MAIN-safe. "
                        "Needs --relmap-outside-grid.")
    g.add_argument("--structure-outside-warm-init-value", type=float, default=1000.0,
                   help="Half the outside-grid pad-vs-valid swing (pad +V, eos/valid -V => swing 2V). The floor's "
                        "colour-over-pad gap on padding is mean~177/max~620 (scripts/verify_outside_grid_lever.py); "
                        "default V=1000 -> swing 2000 dominates it with margin, and the verifier asserts gap<2V.")
    g.add_argument("--relmap-eos-grid", action="store_true",
                   help="Thin-L EOS analogue of --relmap-outside-grid: build an EOS boundary mask from the "
                        "support-verified output extent and feed it to a separate structure_eos_proj. "
                        "Default off; does not touch color.")
    g.add_argument("--structure-eos-warm-init", action="store_true",
                   help="Warm-init the EOS boundary row so EOS is asserted on the predicted thin-L boundary "
                        "from step 0. Needs --relmap-eos-grid.")
    g.add_argument("--structure-eos-warm-init-value", type=float, default=1000.0,
                   help="Half the eos-vs-pad/valid swing for --structure-eos-warm-init (pad/valid -V, eos +V).")
    g.add_argument("--c2-relmap-demos", action="store_true",
                   help="section 15.2-A: feed SUPPORT-side relational maps into TestConditionedC2's demo features "
                        "before the cross-attention (separate zero-init proj => no-op at init; F7-safe). "
                        "Needs --c2-relmap. The cross-demo upgrade: A/B this vs map-only.")
    g.add_argument("--pairdelta-input", action="store_true",
                   help="section 15.2-B: add the PairDeltaEncoder cross-demo rule_vec as a zero-init INPUT-ONLY "
                        "hint (broadcast to grid_features). No output writer. Demoted PairDelta role.")
    g.add_argument("--v3-clean", action="store_true",
                   help="section 15 V3-CLEAN: route output through the factored structure/color head "
                        "(c2_dual_output_head=True) so the relational-map colour LOOKUP is actually READ at "
                        "output, and FORCE every section 15.3 output-side writer OFF (the factored head is the sole "
                        "writer; no fighting). The loaded yaml pins c2_dual_output_head=false, which silently "
                        "runs legacy lm_head + an inert relmap input residual -- this flag fixes that. PAIR "
                        "WITH --c2-relmap and --train-scope v3-color/v3-head/v3-adapter so the factored heads actually "
                        "train via the base CE; --train-scope delta will NOT train color_head/structure_head.")
    g.add_argument("--fresh-structure-head", action="store_true",
                   help="section 15.8 A/B: keep the FRESH 3-way structure_head as the dual-output structure "
                        "source instead of deriving PAD/EOS/VALID from lm_head's logsumexp. Default (off) = "
                        "structure-from-lm_head, which reproduces the floor's pad/eos/shape partition exactly and "
                        "fixes the factored LODO pad/shape regression. Pass this only to A/B the old fresh head.")
    g.add_argument("--c2-gate-init", type=float, default=0.0,
                   help="CHANGE 1 (break the C2 cold-start): initial value of C2's gate_patch/gate_global "
                        "(the tanh-gated demos->z_H channel). Default 0.0 => tanh(0)=0 => demos gated OFF. "
                        "Set >0 (try 0.5-1.0) so the channel is OPEN at init: cross_attn gets real gradient "
                        "and demos enter z_H immediately. Verify via --zh-check / --zh-every demo/input relL2 "
                        "(should jump off ~0) while MAIN holds. Locked sweet spot: 0.3.")
    g.add_argument("--cross-demo-shape", action="store_true",
                   help="CHANGE 3 (molded shape head): enable the supervised cross-demo output-shape head "
                        "(predict the target's output H,W) and UN-DETACH it so the shape CE backprops into "
                        "z_H -> the dense forcing function that makes Change 1's channel carry the demo->shape "
                        "rule (and is the size-change capability). Watch c2_shape_exact/h_acc rise AND "
                        "demo/input relL2 grow, while MAIN holds. ISOLATION: independent of --c2-gate-init / "
                        "the removed repair flags; kept for A/B provenance only.")

    g = ap.add_argument_group("train scope / learning rates")
    g.add_argument("--train-scope", choices=("delta", "v3-color", "v3-head", "v3-adapter",
                                             "v3-where-selector",
                                             "v3-where", "v3-value", "v3-transport", "quarantine",
                                             "v3-head+quarantine", "v3-adapter+quarantine", "all"),
                   default=None,
                   help=("DEFAULT: v3-head without --unified, FULL stack with --unified. Pass an explicit "
                         "scope only to deliberately override --unified's param set. "
                         "v3-head = train only V3 factored output heads + structure levers, frozen core; "
                         "v3-where-selector = P3A: ONLY the C2 selector surface (relmap projs, demo/pair/"
                         "cross-attn projs, gate_patch_token, where_gate_weights) -- update strengths "
                         "gate_patch/gate_global stay frozen (transport closed), everything else frozen; "
                         "v3-where = exact C2 patch/demo/relmap + colour/evidence scope, no structure/PID/core; "
                         "v3-value = exact colour/evidence-head scope, no C2/structure/PID/core; "
                         "v3-transport = C2/object/PairDelta transport + existing structure levers, "
                         "with colour writers/PID/core frozen; "
                         "v3-color = only V3 color_head + relmap input projection, freeze geometry; "
                         "v3-adapter = V3 heads plus C2/relmap/PairDelta adapters, frozen TRM core; "
                         "quarantine = ONLY the PID-quarantined candidate head (quarantine_*) -- z_H not in "
                         "the gradient path, MAIN risk structurally zero; "
                         "v3-head+quarantine = union of v3-head and quarantine (both frozen-core/MAIN-safe); "
                         "v3-adapter+quarantine = v3-adapter plus quarantine, frozen TRM core; "
                         "quarantine params get their own lr (--quarantine-lr) + wd=0 group; "
                         "delta = legacy rule/CTBank/repair modules (most were REMOVED -- near-empty param set); "
                         "all = previous dense optimizer"))
    g.add_argument("--lr", type=float, default=3e-5)
    g.add_argument("--struct-lr", type=float, default=None,
                   help="Dedicated lr for structure_relmap_proj (the §15.9 boundary lever), default = --lr. "
                        "The lever is ONE zero-init proj that must climb ~2-5 logits to beat the frozen lm_head's "
                        "(wrong) pad prediction; at the global lr (Adam ~ lr*steps) it moves ~0.01 in 300 steps = "
                        "far too slow frozen-core. Frozen-core is MAIN-safe, so push this high (e.g. 5e-3..1e-2) to "
                        "actually move pad/shape via the lever instead of the core.")
    g.add_argument("--evidence-lr", type=float, default=None,
                   help="FIX A: dedicated lr for color_evidence_proj (+ color_head_mlp_*), wd=0. The colour "
                        "evidence columns are fresh zero-init tensors that must climb O(1) weights to matter "
                        "against warm-started logits of 5-20; at the core lr 1e-5 they move ~3e-3 in 300 steps "
                        "= inert (measured, all 3 scopes). Default = --struct-lr if set, else 1e-3.")
    g.add_argument("--quarantine-lr", type=float, default=None,
                   help="Dedicated lr for quarantine_* params (wd always 0 for them). Default: --lr. "
                        "Needed under v3-head+quarantine where the warm color_head wants a gentle --lr "
                        "but the quarantine head trains from warm-init at ~3e-3.")
    g.add_argument("--hint-lr", type=float, default=None,
                   help="Dedicated lr for the zero-init input-side hint adapters (frame_embed, "
                        "rule_hyp_embed, c2_demo_relmap_proj, pairdelta_input_encoder/proj) when "
                        "trained under v3-head+quarantine. Default: --evidence-lr. wd always 0.")
    g.add_argument("--hint-lr-names", type=str, default="all",
                   help="Comma-separated hint-module name substrings that ride at --hint-lr "
                        "(choices: frame_embed, rule_hyp_embed, c2_demo_relmap_proj, "
                        "pairdelta_input_encoder, delta_rule_input_proj, rule_factor_proj; or 'all'). Hint modules "
                         "NOT listed stay in the scope but train at the core-safe --lr instead -- "
                         "the per-injector damage-control knob after the six-flag MAIN-ON break.")
    g.add_argument("--c2-lr", type=float, default=3e-4,
                   help="Dedicated LR for v3-where C2 attention/demo/relmap projections.")
    g.add_argument("--c2-gate-lr", type=float, default=1e-3,
                   help="Dedicated LR for the support-conditioned per-token gate; weight decay is always 0.")
    g.add_argument("--lr-warmup-steps", type=int, default=20,
                   help="linear LR warmup 0->lr over the first N steps (stabilizes adapters from random init)")
    g.add_argument("--unified", action="store_true",
                   help="TRAIN THE WHOLE FLOW from the TRM checkpoint as one chronological pipeline: "
                        "C2 conditioning -> structure(pad/eos/shape) -> delta/LODO -> colour, all training "
                        "together (forces --train-scope all + LODO pad/eos losses ON).")
    g.add_argument("--no-abort-on-main-drop", action="store_true", help="do not stop when MAIN color/eos/shape degrades")
    # (user request 2026-07-10) the four --abort-main-* threshold flags were REMOVED; the guard now
    # uses these fixed thresholds (the former defaults). --no-abort-on-main-drop still disables it.

    g = ap.add_argument_group("diagnostics / probes")
    g.add_argument("--per-family-eval", action="store_true",
                   help="MICROSCOPE (general-not-DSL): break the LODO panel by codex task family on "
                        "aug-0 at baseline + end. Measurement only -- NOT a per-family solver/dispatch.")
    g.add_argument("--per-family-collect", type=int, default=960,
                   help="aug-0 tasks to scan for the per-family eval (priority tasks are scattered across "
                        "the 960; each family is then capped to keep the eval fast)")
    g.add_argument("--zh-check", action="store_true",
                   help="DIAGNOSTIC ONLY (0 training steps, then exit): is the recursed state z_H/grid_z "
                        "actually DEMO-conditioned? Re-runs the REAL recurrence with the per-task demos "
                        "shuffled across the batch and reports how far grid_z + the colour prediction move "
                        "vs an input-shuffle reference. ratio~0 => the recursion ignores the demos (the rule "
                        "never reaches z_H). Run with the SAME model flags you train with so the head matches.")
    g.add_argument("--zh-trace", action="store_true",
                   help="Run the z_H conditioning report at BASELINE + END of a normal training run "
                        "(before/after), so you can see whether training actually conditions z_H on the demos. "
                        "Composes with --per-family-eval; unlike --zh-check it does NOT early-exit.")
    g.add_argument("--zh-every", type=int, default=0,
                   help="With --zh-trace, ALSO print the z_H conditioning report every N steps during "
                        "training (0=off). Use 50 so a run stopped early still shows the demo/input relL2 "
                        "TRAJECTORY (is C2 opening?), not just baseline/end.")
    g.add_argument("--zh-amp", type=str, default="",
                   help="FORCED-SIGNAL sweep (diagnostic, use with --zh-check): comma list of scales, e.g. "
                        "'1,4,10,50'. Re-runs the z_H report with the demo->z_H injections (C2 update, "
                        "PairDelta rule_vec, frame hint) multiplied by K, plus MAIN floor strict at each K. "
                        "Read: relL2 grows with K but flip stays 0 => path works, scale/optimization too weak; "
                        "nothing moves at 50x => path disconnected; MAIN breaks => signal uncontrolled.")
    g.add_argument("--shape-debug", action="store_true",
                   help="DIAGNOSTIC (0 steps, exit): run the shape head on the eval batches and print its "
                        "actual H/W predictions vs targets + value/offset histograms, to tell a structural "
                        "bug (collapsed/offset preds) from an untrained head. Use with --cross-demo-shape.")

    return ap


# ======================================================================================
# §3 CONFIG BUILD — pure function of args: yaml load + ALL overrides + ALL guards.
# Verbatim port of oracle lines 382-673 (prints preserved; DATASET -> args.data).
# ======================================================================================
def build_config(args) -> dict:
    # GATE (C-prime postmortem 2026-07-03): --train-scope's old default 'v3-head' silently OVERRODE
    # --unified's full-stack param set -- the branch chain checks train_scope before args.unified,
    # and argparse cannot tell default from explicit. The "unified" C-prime run trained 6 frozen-core
    # tensors for 500 steps: z_H bit-identical at every trace, MAIN pinned at 100 under 40%
    # pid-dropout, loss = irreducible noise. None-sentinel: only an EXPLICIT --train-scope
    # may override --unified. (Second occurrence of the V3-1 silent-default pattern.)
    args.train_scope_explicit = args.train_scope is not None
    if args.train_scope is None:
        args.train_scope = "all" if args.unified else "v3-head"
    try:
        args.save_checkpoint_step_set = {
            int(x.strip()) for x in args.save_checkpoint_steps.split(",") if x.strip()
        }
    except ValueError as exc:
        raise SystemExit("[stage1 GATE] --save-checkpoint-steps must be comma-separated integers") from exc
    if any(x <= 0 for x in args.save_checkpoint_step_set):
        raise SystemExit("[stage1 GATE] --save-checkpoint-steps values must be positive completed-step counts")
    if args.save_checkpoint_step_set and not args.save_checkpoint_dir:
        raise SystemExit("[stage1 GATE] --save-checkpoint-steps requires --save-checkpoint-dir")
    # NOTE: --unified trains the FULL stack (core + adapters) at lr<=1e-5 with LR warmup + NaN
    # guard -- the naive lr 3e-5 full-stack run diverged to NaN by step ~25.

    raw = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    raw["load_checkpoint"] = args.ckpt
    raw["data_paths"] = [args.data]
    raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0
    raw["global_batch_size"] = args.batch
    raw["run_name"] = "stage1_local_probe"
    raw["checkpoint_path"] = (str(Path(args.save_checkpoint_dir).resolve())
                              if args.save_checkpoint_dir else "reports/stage1_local_probe")
    arch = raw.setdefault("arch", {})
    # B2: fuse the trained structure_head PAD/EOS into the output logits (0.0 = aux-only = today's
    # behaviour). Turn on (~0.3) ONLY AFTER structure_head has trained, else it injects noise.
    arch["c2_structure_fusion_alpha"] = args.structure_alpha
    arch["c2_relmap"] = args.c2_relmap
    arch["c2_token_gate_where"] = bool(args.token_gate_where)
    arch["c2_positive_where_gate"] = bool(args.positive_where_gate)
    arch["c2_gate_selector_detach"] = bool(args.gate_selector_detach)
    arch["c2_ordered_evidence_flow"] = bool(args.ordered_evidence_flow)
    # P3A support-conditioned WHERE (Blocks 1-3; all default off).
    arch["c2_isolated_relmap_query"] = bool(args.isolated_relmap_query)
    arch["c2_support_interaction_gate"] = bool(args.support_interaction_gate)
    arch["c2_lodo_zero_support"] = bool(args.lodo_zero_support)
    arch["c2_bounded_evidence_fusion"] = bool(args.bounded_evidence_fusion)
    arch["c2_target_relmap_rho"] = float(args.target_relmap_rho)
    arch["c2_update_rho"] = float(args.c2_update_rho)
    arch["c2_post_hint_rho"] = float(args.post_hint_rho)
    if args.token_gate_where:
        arch["c2_per_token_gate"] = True
    arch["c2_frame_hint"] = args.frame_hint
    arch["c2_rule_hypothesis_hint"] = args.rule_hypothesis_hint
    arch["c2_relmap_outside_grid"] = args.relmap_outside_grid
    arch["c2_structure_outside_warm_init"] = args.structure_outside_warm_init
    arch["c2_structure_outside_warm_init_value"] = args.structure_outside_warm_init_value
    arch["c2_relmap_eos_grid"] = args.relmap_eos_grid
    arch["c2_structure_eos_warm_init"] = args.structure_eos_warm_init
    arch["c2_structure_eos_warm_init_value"] = args.structure_eos_warm_init_value
    # §15.2 per-component upgrades (zero-init, F7-safe, default OFF).
    arch["c2_relmap_demos"] = args.c2_relmap_demos          # A: support maps -> C2 demo features
    arch["c2_pairdelta_input_feature"] = args.pairdelta_input  # B: PairDelta rule_vec -> input hint
    arch["c2_pairdelta_include_identity"] = bool(args.pairdelta_include_identity)
    arch["c2_pairdelta_spatial"] = bool(args.pairdelta_spatial)
    arch["c2_rule_factor_hint"] = bool(args.rule_factor_hint)
    arch["c2_object_pair_tokens"] = bool(args.object_pair_tokens)
    # Canonical VALUE keeps palette only as a final output-validity constraint. The historical
    # --task-palette shorthand enables feature+bias; in this lane it deliberately becomes bias-only.
    arch["c2_task_palette_feature"] = bool(
        (args.task_palette or args.task_palette_feature) and not args.canonical_value_binder)
    arch["c2_task_palette_bias"] = bool(args.task_palette or args.task_palette_bias)
    arch["c2_task_palette_strength"] = float(args.task_palette_strength)
    arch["c2_task_palette_hard"] = bool(args.task_palette_hard)
    arch["c2_rel_where_hint"] = bool(args.rel_where_hint)
    arch["c2_rel_where_topk"] = max(1, int(args.rel_where_topk))
    arch["c2_algo_where_maps"] = bool(args.algo_where_maps)
    arch["c2_algo_where_touch"] = bool(args.algo_where_touch)   # D6 (B13): touch cols 14/15 evidence
    arch["c2_pairdelta_intent_hint"] = bool(args.pairdelta_intent_hint)
    arch["c2_transition_hint"] = bool(args.transition_hint)
    arch["c2_value_evidence_v2"] = bool(args.value_evidence_v2)
    arch["c2_value_v2_rich_ctx"] = bool(args.value_v2_rich_ctx)
    arch["c2_canonical_value_binder"] = bool(args.canonical_value_binder)
    arch["c2_value_backoff_tau"] = float(args.value_backoff_tau)
    arch["c2_extent_conditioned_structure"] = bool(args.extent_conditioned_structure)
    # V2-model evidence (trm_fvr_v2 only; the V2-GUARD below refuses them on a legacy config)
    arch["c2_verified_frame_evidence"] = bool(args.verified_frame_evidence)
    arch["c2_analogy_evidence"] = bool(args.analogy_evidence)
    arch["c2_value_ctx_gate"] = bool(args.value_ctx_gate)
    arch["c2_value_v2_backoff"] = bool(args.value_v2_backoff)
    arch["c2_pairdelta_color_evidence"] = bool(args.pairdelta_color_evidence)
    arch["c2_pairdelta_structure_evidence"] = bool(args.pairdelta_structure_evidence)
    arch["c2_pairdelta_bidi_evidence"] = bool(args.pairdelta_bidi_evidence)
    arch["c2_pairdelta_input_conf_gate"] = bool(args.pairdelta_input_conf_gate)
    arch["c2_value_ctx_bind"] = bool(args.value_ctx_bind)
    arch["c2_kinematic_evidence"] = bool(args.kinematic_evidence)   # E-5 (A3): kinematic per-cell facts
    # D11 guard (codex): backoff silently wins over rich-ctx inside the model -- both on means the
    # rich-ctx key is DEAD without a trace (the V3-1 silent-default disease). Refuse outright.
    if args.value_v2_backoff and args.value_v2_rich_ctx:
        raise SystemExit(
            "[stage1 GATE] --value-v2-backoff and --value-v2-rich-ctx are BOTH set: backoff takes "
            "precedence and rich-ctx would be silently dead. Pass exactly one (they are the A/B pair).")
    if args.bind_aux_weight > 0 and not (args.value_ctx_bind or args.canonical_value_binder):
        raise SystemExit(
            "[stage1 GATE] --bind-aux-weight requires --value-ctx-bind or "
            "--canonical-value-binder; otherwise no bind slice exists.")
    if args.bind_residual_aware and not args.canonical_value_binder:
        raise SystemExit(
            "[stage1 GATE] --bind-residual-aware requires --canonical-value-binder; the legacy bind "
            "path does not expose separated changed/copy support or base-without-bind logits.")
    if args.bind_residual_aware and abs(float(args.bind_aux_copy_weight)) > 1e-12:
        raise SystemExit(
            "[stage1 GATE] Repair B is changed-only: pass --bind-aux-copy-weight 0. Copy behaviour is "
            "measured but protected by the ordinary LODO copy/preservation losses.")
    if args.bind_residual_aware and args.quarantine_candidate:
        raise SystemExit(
            "[stage1 GATE] --bind-residual-aware cannot supervise --quarantine-candidate: quarantine "
            "does not deploy color_evidence_proj, so that loss would train a non-deployed writer.")
    if args.pairdelta_spatial and not args.pairdelta_input:
        raise SystemExit("[stage1 GATE] --pairdelta-spatial requires --pairdelta-input.")
    if (args.gate_where_weight > 0 or args.gate_support_contrast_weight > 0) and not args.token_gate_where:
        raise SystemExit(
            "[stage1 GATE] gate WHERE losses require --token-gate-where; otherwise no per-cell gate is emitted.")
    if args.paired_control_report and args.paired_bootstrap_samples < 1:
        raise SystemExit("[stage1 GATE] --paired-bootstrap-samples must be >= 1 "
                         f"(got {args.paired_bootstrap_samples}); fail now, not after the run.")
    if args.paired_control_report and not args.task_metrics_out:
        raise SystemExit("[stage1 GATE] --paired-control-report requires --task-metrics-out "
                         "(there is no treatment CSV to compare without it).")
    if args.positive_where_gate and not args.token_gate_where:
        raise SystemExit("[stage1 GATE] --positive-where-gate requires --token-gate-where.")
    if args.gate_selector_detach and not args.positive_where_gate:
        raise SystemExit("[stage1 GATE] --gate-selector-detach requires --positive-where-gate.")
    if args.positive_where_gate and not args.ordered_evidence_flow:
        raise SystemExit("[stage1 GATE] --positive-where-gate requires --ordered-evidence-flow so the "
                         "selector can read target relations before C2.")
    if args.positive_where_gate and not args.rel_where_hint:
        raise SystemExit("[stage1 GATE] --positive-where-gate requires --rel-where-hint so all top-K "
                         "support-fitted masks reach the selector.")
    if args.positive_where_gate and abs(float(args.c2_gate_init)) > 1e-12:
        raise SystemExit("[stage1 GATE] --positive-where-gate requires --c2-gate-init 0 for exact "
                         "step-0 identity; the selector learns before its update strengths open.")
    if args.isolated_relmap_query and not args.ordered_evidence_flow:
        raise SystemExit("[stage1 GATE] --isolated-relmap-query requires --ordered-evidence-flow "
                         "(x_query is built from the ordered target-relmap residual).")
    if args.support_interaction_gate:
        if not args.isolated_relmap_query:
            raise SystemExit("[stage1 GATE] --support-interaction-gate requires "
                             "--isolated-relmap-query (the gate reads x_query).")
        if not args.positive_where_gate:
            raise SystemExit("[stage1 GATE] --support-interaction-gate requires "
                             "--positive-where-gate (q = sigmoid in [0,1]).")
        if not args.gate_selector_detach:
            raise SystemExit("[stage1 GATE] --support-interaction-gate requires "
                             "--gate-selector-detach: WHERE supervision owns the selector during "
                             "P3A; transport losses must not rewrite it.")
    if args.lodo_zero_support and not args.token_gate_where:
        raise SystemExit("[stage1 GATE] --lodo-zero-support requires --token-gate-where; without "
                         "the per-cell gate there is no zero-support WHERE value to export.")
    if args.ordered_evidence_flow and not args.c2_relmap:
        raise SystemExit("[stage1 GATE] --ordered-evidence-flow requires --c2-relmap.")
    if args.extent_conditioned_structure and args.candidate_floor_structure:
        raise SystemExit("[stage1 GATE] --extent-conditioned-structure and --candidate-floor-structure "
                         "are mutually exclusive structure authorities.")
    if args.canonical_value_binder:
        _competing = {
            "transition-hint": args.transition_hint,
            "value-evidence-v2": args.value_evidence_v2,
            "value-ctx-gate": args.value_ctx_gate,
            "value-ctx-bind": args.value_ctx_bind,
            "algo-where-maps": args.algo_where_maps,
            "algo-where-touch": args.algo_where_touch,
            "pairdelta-intent-hint": args.pairdelta_intent_hint,
            "pairdelta-color-evidence": args.pairdelta_color_evidence,
            "pairdelta-bidi-evidence": args.pairdelta_bidi_evidence,
            "verified-frame-evidence": args.verified_frame_evidence,
            "analogy-evidence": args.analogy_evidence,
            "kinematic-evidence": args.kinematic_evidence,
            "color-mlp": args.color_mlp > 0,
            "task-palette-feature": bool(args.task_palette_feature),
        }
        _enabled_competing = [name for name, enabled in _competing.items() if enabled]
        if _enabled_competing:
            raise SystemExit("[stage1 GATE] --canonical-value-binder is the sole support-derived VALUE "
                             "authority; disable competing predictors: " + ", ".join(_enabled_competing))
    arch["c2_color_head_mlp_dim"] = int(args.color_mlp)
    arch["c2_quarantine_candidate"] = bool(args.quarantine_candidate)
    arch["c2_quarantine_hidden"] = int(args.quarantine_hidden)
    if args.quarantine_candidate:
        print(f"[V3 quarantine] PID-quarantined candidate head ON (hidden={args.quarantine_hidden}) | "
              f"candidate colour = MLP over PID-free evidence only (z_H never read) | canvas = floor "
              f"structure | warm-init copy-unless-consensus | train with --train-scope quarantine.")
        if not args.floor_candidate_split:
            print("[V3 quarantine WARNING] --quarantine-candidate needs --floor-candidate-split "
                  "(no candidate lane without the split; the flag will be INERT).")
        if args.train_scope == "v3-head+quarantine":
            print("[V3 quarantine NOTE] union scope: quarantine_* trains alongside the v3 heads "
                  "(frozen-core, floor-safe via abort guards, but not the pure-quarantine zero-MAIN-risk).")
        elif args.train_scope == "v3-adapter+quarantine":
            print("[V3 quarantine NOTE] adapter+quarantine scope: quarantine_* trains alongside V3/C2/PairDelta "
                  "adapters with the TRM core frozen; MAIN-strict abort guard is load-bearing.")
        elif args.train_scope != "quarantine":
            print(f"[V3 quarantine WARNING] --train-scope {args.train_scope}: the zero-MAIN-risk guarantee "
                  f"holds only under --train-scope quarantine (nothing but quarantine_* params train).")
    if args.pid_dropout > 0:
        print(f"[pid-dropout] necessity pressure ON: {args.pid_dropout:.0%} of TRAINING rows get the blank "
              f"PID(0) (eval batches untouched -> panel/abort stay honest). Watch [pid-null] rise.")
        if not args.unified and args.train_scope in ("v3-color", "v3-head", "v3-adapter",
                                                     "v3-where", "v3-value", "v3-transport", "quarantine",
                                                     "v3-head+quarantine", "v3-adapter+quarantine"):
            print("[pid-dropout WARNING] frozen-core scope: the recursion cannot re-route around a missing "
                  "PID, so pid-dropout mostly adds noise here. Intended for --unified (C-prime).")
    if args.color_mlp > 0:
        print(f"[V3 color-mlp] interaction residual ON (dim={args.color_mlp}) | zero-init output | "
              f"step-0 == linear head | adds input-colour x evidence products.")
    if args.transition_hint:
        print("[V3 transition-hint] VALUE binding ON | per-cell demo-consensus P(out|in) over changed "
               "support cells -> 10 zero-init color_head columns | LODO-safe | evidence only, no writer.")
    if args.token_gate_where:
        print(f"[V3 token-gate-WHERE] support-conditioned per-token C2 gate ON | "
              f"where_w={args.gate_where_weight} support_contrast_w={args.gate_support_contrast_weight} | "
              f"selector={'sigmoid' if args.positive_where_gate else 'legacy-tanh'} | "
              f"transport_grad={'detached' if args.gate_selector_detach else 'coupled'}.")
    if args.ordered_evidence_flow:
        print("[V3 ordered-flow] target/support relmaps and top-K WHERE evidence enter before C2.")
    if args.bounded_evidence_fusion:
        print(f"[V3 bounded-fusion] rho target={args.target_relmap_rho:g} "
              f"c2={args.c2_update_rho:g} post={args.post_hint_rho:g}.")
    if args.bind_residual_aware:
        print(f"[V3 bind-aux Repair B] residual-aware changed-only CE ON | weight={args.bind_aux_weight} "
              f"changed_w={args.bind_aux_changed_weight} | base colour logits detached | copy CE/flip "
              "diagnostic only.")
    elif args.bind_aux_weight > 0:
        print(f"[V3 bind-aux] support-masked CE on the canonical/legacy bind slice ON | weight={args.bind_aux_weight} "
              f"changed_w={args.bind_aux_changed_weight} copy_w={args.bind_aux_copy_weight}.")
    if args.canonical_value_binder:
        print(f"[V3 canonical-VALUE] collision-free hierarchical task-local binder ON | "
              f"tau={args.value_backoff_tau:g} | sole support-derived colour authority.")
    if args.rule_factor_hint or args.object_pair_tokens:
        print(f"[V3 composition] rule_factors={args.rule_factor_hint} object_pair_tokens={args.object_pair_tokens} "
              "| independent extent/colour/movement/count evidence.")
    if args.extent_conditioned_structure:
        print("[V3 extent-structure] existing candidate structure is blended with floor by same-extent confidence.")
    if args.value_evidence_v2:
        print("[V3 value-evidence-v2] copy/change + context-conditioned VALUE evidence ON | "
              "36 zero-init color_head columns | evidence only, no writer.")
        if args.value_v2_rich_ctx:
            print("[V3 value-v2-rich-ctx] ObjectBank cell_conditioning_signature is the VALUE-V2 context "
                  "key | richer neighbour/object/enclosure buckets | default-off probe.")
        if args.value_v2_aux_weight > 0:
            print(f"[V3 value-v2-aux] explicit V2-column LODO CE ON | weight={args.value_v2_aux_weight} "
                  f"changed_w={args.value_v2_aux_changed_weight} copy_w={args.value_v2_aux_copy_weight}.")
    elif args.value_v2_rich_ctx:
        print("[V3 value-v2-rich-ctx WARNING] requested but --value-evidence-v2 is off; flag is inert.")
    if args.rel_where_hint:
        print("[V3 rel-where] ObjectBank/relmap WHERE hint ON | evidence feature only | no VALUE writer.")
        if not args.c2_relmap:
            print("[V3 rel-where WARNING] --rel-where-hint needs --c2-relmap; hint will be zero/inert without maps.")
    if args.pairdelta_intent_hint:
        print("[V3 pairdelta-intent] PairDelta intent hint ON | evidence feature only | no logit writer.")
    if args.rule_hypothesis_hint:
        print("[V3 rule-hypothesis] object_rule_bank.infer_rule_hypotheses ON | top family embedding "
              "broadcast into grid_features | zero-init/F7-safe | evidence only, no proposal writer.")
    if args.task_palette or args.task_palette_feature or args.task_palette_bias:
        _palette_mode = []
        if arch["c2_task_palette_feature"]:
            _palette_mode.append("feature")
        if arch["c2_task_palette_bias"]:
            _palette_mode.append("hard-mask" if args.task_palette_hard else "soft-bias")
        print(f"[V3 palette] task palette ON ({'+'.join(_palette_mode)}) | "
              f"allowed=support inputs/outputs + target input | strength={args.task_palette_strength}")
    if args.c2_relmap_demos and not args.c2_relmap:
        print("[section 15.2-A WARNING] --c2-relmap-demos needs --c2-relmap (the dataloader supplies the support "
              "maps only when c2_relmap is on). The C2 demo-feed will be SKIPPED.")
    # --- §15 V3-CLEAN SWITCH ----------------------------------------------------------------
    # CLEANUP (2026-07-01): the 14 legacy inert CLI flags plus --color-force/--apply-canvas/
    # --copy-structure were DELETED with their 11 competing output writers. --palette-constrain kept.
    if args.v3_clean:
        # The factored head becomes the SOLE output writer; the relmap colour-lookup is read at output.
        arch["c2_dual_output_head"] = True
        arch["c2_structure_fusion_alpha"] = 0.0
        # §15.8: recover the factored pad/shape regression -- derive PAD/EOS/VALID from the trained
        # lm_head (floor-EXACT partition) instead of the fresh ~300-step structure_head.
        arch["c2_structure_from_lmhead"] = not args.fresh_structure_head
        print("[v3-clean section 15] c2_dual_output_head=True | factored structure/color head is the SOLE output "
              "writer | relmap READ at output | all section 15.3 writers forced OFF.")
        print("[v3-clean section 15.8] structure-from-lm_head=%s | %s." % (
            not args.fresh_structure_head,
            "PAD/EOS/VALID = log_softmax([lm_pad, lm_eos, logsumexp(lm_colour)]) -> floor-exact pad/shape"
            if not args.fresh_structure_head else "FRESH structure_head (A/B; expect the ~300-step pad/shape regression)"))
        if args.floor_candidate_split:
            print("[v3-clean selector] FLOOR/CANDIDATE split ON | MAIN uses lm_head floor | LODO trains V3 candidate "
                  "| support selector metrics tie to FLOOR.")
            if args.candidate_floor_structure:
                print("[v3-clean selector] Candidate uses FLOOR structure + V3 colour values on floor-valid cells.")
        if args.train_scope not in ("v3-color", "v3-head", "v3-adapter", "v3-where", "v3-value",
                                    "v3-transport", "quarantine",
                                    "v3-head+quarantine", "v3-adapter+quarantine", "all") and not args.unified:
            print("[v3-clean WARNING] --train-scope is 'delta': color_head/structure_head are NOT in the "
                  "delta param set, so the factored heads will NOT train. Re-run with --train-scope v3-color "
                  "for colour calibration, v3-head for structure+colour, or v3-adapter after the floor is stable.")
        if args.unified:
            print("[v3-clean WARNING] --unified trains the broad/full stack and can move the TRM core. "
                  "For floor-safe V3 calibration prefer --train-scope v3-head, then v3-adapter.")
        # CONFLICT GUARD: --unified intends a full-stack run, but an explicit frozen-core --train-scope
        # is checked FIRST in the param selector below, so it SILENTLY wins and freezes the core.
        if args.unified and args.train_scope_explicit and args.train_scope in (
            "v3-color", "v3-head", "v3-adapter", "v3-where", "v3-value", "v3-transport",
            "v3-head+quarantine", "v3-adapter+quarantine"
        ):
            print(f"[v3-clean WARNING] --unified + EXPLICIT --train-scope {args.train_scope}: the frozen-core "
                  f"scope OVERRIDES --unified's full-stack param set, so the TRM core stays FROZEN and "
                  f"LODO pad/shape CANNOT move off the floor. Drop --train-scope to let --unified train the core.")
        # The structure (pad/eos) gradient on the LODO canvas is weighted by --lodo-pad-weight/
        # --lodo-eos-weight (NOT auto-enabled by --unified). At 0.0 the boundary CE is x0 ->
        # structure_relmap_proj gets no gradient and pad/shape never trains, regardless of scope.
        if args.lodo_pad_weight <= 0.0 and args.lodo_eos_weight <= 0.0 and args.train_scope != "quarantine":
            print("[v3-clean WARNING] --lodo-pad-weight=0 and --lodo-eos-weight=0: the LODO boundary CE is OFF, "
                  "so structure_relmap_proj receives no gradient and pad/shape will not improve. Pass e.g. "
                  "--lodo-pad-weight 1.0 --lodo-eos-weight 3.0 to actually train structure. "
                  "(Under --train-scope quarantine this is EXPECTED: structure is frozen at the solved floor.)")
    if args.router_gate:
        print(f"[Phase3] ROUTER gate ON (thr={args.router_threshold}): colour head fires on CLEAN "
              f"recolours, shuts on relational/structural tasks (demo self-consistency)."
              + ("  + PALETTE constraint (no-invention prior on relational)." if args.palette_constrain else ""))
    # CHANGE 1: open the demos->z_H channel at init (config default is 0.0 = gated shut = the cold-start).
    arch["c2_gate_init"] = args.c2_gate_init
    # LODO plumbing
    arch["c2_lodo_blank_pid"] = True
    arch["c2_leave_one_demo_weight"] = 0.0
    arch["c2_lodo_force_build"] = True
    # The SHUFFLE build is a third full recurrence per forward (train AND eval) whose only
    # consumers are the contrast hinge + the anchor-polluted contrast metrics. With contrast off
    # it is pure waste -- MEASURED 81 s/step on the 400-step Q-run (~33% of every step).
    arch["c2_lodo_force_shuffle"] = bool(args.contrast_weight > 0 or args.token_gate_where)
    if not arch["c2_lodo_force_shuffle"]:
        print("[speed] contrast-weight=0 -> shuffle forward DISABLED (~33% faster steps; "
              "chgCE/gap print nan, read chgCEc/copyCEc instead).")
    arch["c2_lodo_contrast_weight"] = 0.0
    # Every eligible row trains LODO: the yaml pins max_samples=4, which silently DISCARDED half of
    # each batch-8's cross-demo signal -- the one gradient that teaches the rule was halved for free.
    arch["c2_lodo_max_samples"] = int(args.batch)
    arch["c2_floor_candidate_split"] = args.floor_candidate_split
    arch["c2_candidate_floor_structure"] = args.candidate_floor_structure
    arch["c2_delta_expose_base_logits"] = args.preserve > 0 or args.floor_candidate_split
    loss = arch.setdefault("loss", {})
    loss["c2_delta_lodo_weight"] = 0.0
    loss["c2_delta_changed_weight"] = args.lodo_changed_weight
    loss["c2_delta_color_weight"] = args.lodo_copy_weight
    # This probe defaults to colour application. PAD/EOS canvas CE is opt-in via CLI;
    # with --floor-candidate-split it trains only the candidate while MAIN stays floor-backed.
    loss["c2_delta_pad_weight"] = args.lodo_pad_weight
    loss["c2_delta_eos_weight"] = args.lodo_eos_weight
    # Contrast starts at 0 and is SCHEDULED with the LODO warmup/ramp in the training loop.
    loss["c2_delta_contrast_weight"] = 0.0
    loss["c2_delta_contrast_margin"] = 0.5
    loss["c2_delta_contrast_per_row"] = bool(args.contrast_per_row)
    loss["c2_delta_diag"] = True
    loss["c2_delta_preserve_weight"] = args.preserve
    loss["c2_value_v2_aux_weight"] = float(args.value_v2_aux_weight)
    loss["c2_value_v2_aux_changed_weight"] = float(args.value_v2_aux_changed_weight)
    loss["c2_value_v2_aux_copy_weight"] = float(args.value_v2_aux_copy_weight)
    loss["c2_value_ctx_bind_aux_weight"] = float(args.bind_aux_weight)
    loss["c2_value_ctx_bind_aux_changed_weight"] = float(args.bind_aux_changed_weight)
    loss["c2_value_ctx_bind_aux_copy_weight"] = float(args.bind_aux_copy_weight)
    loss["c2_bind_residual_aware"] = bool(args.bind_residual_aware)
    loss["c2_gate_where_weight"] = float(args.gate_where_weight)
    loss["c2_gate_support_contrast_weight"] = float(args.gate_support_contrast_weight)
    loss["c2_changed_valid_loss_weight"] = args.changed_valid
    # CHANGE 3: cross-demo output-shape head (predicts target_height/width; c2_shape_exact/h/w_acc).
    arch["c2_shape_head"] = args.cross_demo_shape        # H/W readout
    arch["c2_shape_pool"] = "zH_rowcol"                  # H/W-separable pool (mean pool is dimension-blind)
    loss["c2_shape_loss_weight"] = args.shape_weight if args.cross_demo_shape else 0.0

    if args.unified:
        # Full-stack training needs a safe lr: 3e-5 (no warmup) diverged to NaN by step 25.
        # Match the recipe that produced step_1631 (lr 1e-5) + the LR warmup + NaN guard below.
        args.lr = min(args.lr, 1.0e-5)
        # ===== THE UNIFIED FLOW (chronological, components complement) =====
        # L1 CONDITION (C2) -> L2 STRUCTURE (pad/eos) -> L3 RULE/LODO (delta) -> L4 COLOUR.
        # Structure must be explicit: PAD/EOS weights come from --lodo-pad/eos-weight so a command
        # line faithfully defines the run. Keep preservation so correct cells are not clobbered.
        if args.preserve <= 0:
            args.preserve = 1.0
            loss["c2_delta_preserve_weight"] = 1.0
        _pe_on = (args.lodo_pad_weight > 0.0) or (args.lodo_eos_weight > 0.0)
        # This banner must not lie about the param set: an explicit frozen-core scope wins over
        # --unified (the C-prime postmortem bug was this banner printing "FULL stack" over 6 tensors).
        if args.train_scope_explicit and args.train_scope != "all":
            _u_stack = f"scope OVERRIDDEN by explicit --train-scope {args.train_scope} (TRM core FROZEN)"
        else:
            _u_stack = "FULL stack (TRM core + adapters)"
        print(f"[unified] {_u_stack} @ lr<=1e-5 | "
              f"LODO pad/eos CE {'ON' if _pe_on else 'OFF (pass --lodo-pad-weight/--lodo-eos-weight)'} "
              f"(pad={args.lodo_pad_weight}, eos={args.lodo_eos_weight}) | C2+structure+delta+colour train together")

    # ===== V2-ONLY START-GATE (audit A6) =====
    # These arch keys exist ONLY in trm_fvr_v2's config; legacy trm_fvr_c2's config silently DROPS
    # unknown keys (V3-1 class: flag accepted, flag discarded, the run lies about what it tests).
    # ADD every future v2-only config key to this tuple when its driver flag lands.
    _V2_ONLY_KEYS = (
        "c2_verified_frame_evidence", "c2_analogy_evidence", "c2_value_ctx_gate",
        "c2_value_v2_backoff", "c2_pairdelta_color_evidence", "c2_pairdelta_structure_evidence",
        "c2_pairdelta_bidi_evidence", "c2_pairdelta_input_conf_gate", "c2_value_ctx_bind",
        "c2_token_gate_where", "c2_positive_where_gate", "c2_gate_selector_detach",
        "c2_ordered_evidence_flow",
        "c2_bounded_evidence_fusion", "c2_canonical_value_binder",
        "c2_pairdelta_include_identity", "c2_rule_factor_hint", "c2_object_pair_tokens",
        "c2_pairdelta_spatial",
        "c2_extent_conditioned_structure",
        "c2_algo_where_touch",                           # D6: wired (--algo-where-touch)
        "c2_kinematic_evidence",                         # E-5: wired (--kinematic-evidence)
        "c2_frame_hint_ranked",                          # D5: yaml-only, UNWIRED (model refuses)
    )
    _arch_name = str(arch.get("name", ""))
    _v2_set = sorted(k for k in _V2_ONLY_KEYS if arch.get(k))
    if _v2_set and "trm_fvr_v2" not in _arch_name:
        raise SystemExit(
            f"[V2-GUARD] {_v2_set} set, but arch.name={_arch_name or '<missing>'} is NOT the V2 model: "
            f"the legacy model silently drops these keys, so the run would test NOTHING you asked for. "
            f"Pass --config config/new_fvr_full_v2_base.yaml (arch recursive_reasoning.trm_fvr_v2).")
    return raw


# ======================================================================================
# §4 DISTRIBUTED — pretrain.py's exact torchrun pattern. No LOCAL_RANK => rank 0 / world 1
# (the byte-identical single-GPU path every equivalence gate runs on). Block 5 wiring:
#   R0 backend = nccl when CUDA+nccl exist (Linux/Kaggle), else gloo (the Windows 2-process
#      smoke lane); device = LOCAL_RANK % device_count so a 2-process test fits on 1 GPU.
#   R4 rank>0 stdout -> devnull (every print in this file is rank-0-only without touching
#      the ~100 call sites; stderr stays live so crashes on any rank are visible).
#   R5 agreed control flow: every per-rank skip/abort decision goes through _ddp_agree_max
#      on the gloo CPU group BEFORE acting — one rank skipping while others enter the next
#      collective is a deadlock, the multi-GPU analogue of the V3-1 silent kill.
# ======================================================================================
def init_distributed() -> SimpleNamespace:
    rank, world = 0, 1
    cpu_group = None
    if "LOCAL_RANK" in os.environ:
        import torch.distributed as dist
        backend = "nccl" if (torch.cuda.is_available() and dist.is_nccl_available()) else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        if torch.cuda.is_available():
            # modulo device_count: the 2-process single-GPU smoke (G5b) shares device 0;
            # on a real multi-GPU node this is exactly pretrain's LOCAL_RANK mapping.
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % max(1, torch.cuda.device_count()))
        cpu_group = dist.new_group(backend="gloo")
        if rank != 0:
            sys.stdout = open(os.devnull, "w")   # R4: rank-0-only printing, stderr untouched
    return SimpleNamespace(rank=rank, world=world, cpu_group=cpu_group, is_main=(rank == 0))


def _ddp_agree_max(ctx, value: float) -> float:
    """R5: agree a control-flow scalar across ranks (MAX over the gloo CPU group).
    1.0 from ANY rank => 1.0 on ALL ranks. No-op at world==1 (the gated single-GPU path)."""
    if ctx.world <= 1:
        return value
    import torch.distributed as dist
    t = torch.tensor([float(value)])
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=ctx.cpu_group)
    return float(t.item())


def _ddp_bcast_scalar(ctx, value: float) -> float:
    """R6: rank 0 decides (eval/panel/abort), everyone receives. No-op at world==1."""
    if ctx.world <= 1:
        return value
    import torch.distributed as dist
    t = torch.tensor([float(value)])
    dist.broadcast(t, src=0, group=ctx.cpu_group)
    return float(t.item())


def _ddp_allreduce_grads(ctx, params) -> None:
    """R2: average the trained-scope grads across ranks AFTER backward, BEFORE clip — so
    clip_grad_norm_ sees the true global-batch norm and returns the SAME value on every
    rank (the grad-guard decision then agrees by construction). Unscoped params never
    step, so reducing only the scope set is sufficient (pretrain's grads-that-exist rule)."""
    if ctx.world <= 1:
        return
    import torch.distributed as dist
    for p in params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(ctx.world)


# ======================================================================================
# §5 MODEL + SCOPE — the scope name tuples are MODULE-LEVEL DATA (gate G2 diffs them
# against the oracle text). Selection logic verbatim from oracle 1240-1307.
# ======================================================================================

# --- §10 LEGACY-GATED (census verdict): the pre-V3 delta branch scope. Most of its modules
#     were physically REMOVED from the models; the name tuple is kept so --train-scope delta
#     still resolves (and refuses with the 0-tensor SystemExit when truly empty). Never the
#     default; prints a [LEGACY-GATED] banner when selected.
LEGACY_DELTA_NAMES = (
    "delta_rule_encoder",
    "delta_rule_proj",
    "delta_rule_logit_fuse",
    "delta_rule_logit_head",
    "delta_rule_cell_gate",
    "delta_rule_struct_head",
    "delta_rule_color_head",
    "delta_rule_slot_attn",
    "delta_rule_prior_proj",
    "color_transition_bank",
    "color_repair_head",
    "c2_color_prior_gate",
    "c2_copy_structure_gate",
)
# V3 scopes keep the pretrained TRM floor intact. First calibrate the replacement
# factored heads; only then add C2/relmap/PairDelta adapters.
V3_COLOR_NAMES = (
    "color_head",
    "color_evidence_proj",
    # ".inner.relmap_proj" (not bare "relmap_proj"): substring matching also caught
    # structure_relmap_proj + c2_demo_relmap_proj, silently training STRUCTURE under a
    # scope labelled "frozen-structure".
    "inner.relmap_proj",
)
V3_VALUE_NAMES = (
    "color_head",            # includes optional color_head_mlp_* residuals
    "color_evidence_proj",
)
V3_WHERE_C2_NAMES = (
    "inner.relmap_proj",
    "c2_demo_relmap_proj",
    "c2.demo_proj",
    "c2.demo_scalar_proj",
    "c2.demo_mix",
    "c2.pair_proj",
    "c2.pair_mix",
    "c2.cross_attn",
    "c2.patch_proj",
    "c2.global_proj",
)
V3_WHERE_GATE_NAMES = (
    "c2.gate_patch_token",
    "c2.where_gate_weights",
    "c2.gate_patch",
    "c2.gate_global",
)
V3_WHERE_NAMES = V3_VALUE_NAMES + V3_WHERE_C2_NAMES + V3_WHERE_GATE_NAMES + ("rule_factor_proj",)
# P3A Block 5: the EXACT selector-training surface. V3_WHERE_C2_NAMES plus the selector head and
# hint weights -- and NOTHING else: gate_patch/gate_global (update strengths) stay frozen at zero
# (transport intentionally closed during P3A), colour/structure/PID/binder/PairDelta all frozen.
V3_WHERE_SELECTOR_NAMES = V3_WHERE_C2_NAMES + (
    "c2.gate_patch_token",
    "c2.where_gate_weights",
)
# Forbidden SUBSTRINGS for the v3-where-selector scope; gate_patch/gate_global need exact-suffix
# checks because "c2.gate_patch" is a substring of the allowed "c2.gate_patch_token".
_WHERE_SELECTOR_FORBIDDEN = (
    "pid_task", "puzzle_emb", "lm_head", ".L_level.", "structure_", "color_head",
    "color_evidence_proj", "quarantine_", "pairdelta", "delta_rule", "rule_factor_proj",
    "frame_embed", "rule_hyp_embed", "canonical",
)


def select_where_selector_params(named_params) -> tuple[list, list, list]:
    """(name, param) pairs -> (params, selected_names, forbidden_names). Single source for the
    v3-where-selector scope AND its startup forbidden-tensor gate, unit-testable without a model."""
    sel = [(n, p) for n, p in named_params if any(k in n for k in V3_WHERE_SELECTOR_NAMES)]
    bad = [n for n, _p in sel
           if any(k in n for k in _WHERE_SELECTOR_FORBIDDEN)
           or n.endswith("c2.gate_patch") or n.endswith("c2.gate_global")]
    return [p for _n, p in sel], [n for n, _p in sel], bad
V3_TRANSPORT_NAMES = V3_WHERE_C2_NAMES + V3_WHERE_GATE_NAMES + (
    "pairdelta_input_encoder",
    "delta_rule_input_proj",
    "rule_factor_proj",
    "structure_head",
    "structure_relmap_proj",
    "structure_outside_proj",
    "structure_eos_proj",
    "structure_pairdelta_proj",
)
V3_HEAD_NAMES = (
    "color_head",
    "color_evidence_proj",
    "structure_head",
    "structure_relmap_proj",
    "structure_outside_proj",
    "structure_eos_proj",
    "structure_pairdelta_proj",   # D9 (File #5): exists only under --pairdelta-structure-evidence
)
V3_ADAPTER_NAMES = V3_HEAD_NAMES + (
    "frame_embed",
    "rule_hyp_embed",
    "c2.",
    "c2_demo_relmap_proj",
    "pairdelta_input_encoder",
    "delta_rule_input_proj",
    "rule_factor_proj",
    "pid_task_modulator",
    "pid_task_gate",
)
# §10 LEGACY-GATED: the old dense-adapter scope (superseded by the v3-* family).
LEGACY_ADAPTER_NAMES = LEGACY_DELTA_NAMES + ("c2.", "structure_head", "pid_task_modulator")
# Zero-init input-side hint adapters (each exists only when its flag is on, so the name match
# is naturally gated): --frame-hint, --rule-hypothesis-hint, --c2-relmap-demos, --pairdelta-input.
HINT_ADAPTER_NAMES = (
    "frame_embed",
    "rule_hyp_embed",
    "c2_demo_relmap_proj",
    "pairdelta_input_encoder",
    "delta_rule_input_proj",
    "rule_factor_proj",
)


def build_data_and_model(args, raw, dist_ctx):
    """PretrainConfig -> sharded dataloader -> seeded model build. Verbatim oracle 675-698,
    with rank/world threaded (0/1 today: identical path; sharding is free when DDP lands)."""
    config = pretrain.PretrainConfig(**raw)
    loader, meta = pretrain.create_dataloader(
        config, "train", dist_ctx.rank, dist_ctx.world, test_set_mode=False,
        epochs_per_iter=1, global_batch_size=config.global_batch_size,
    )
    if args.epochs is not None:
        epoch_steps = int(args.epochs * meta.total_groups * meta.mean_puzzle_examples / args.batch)
        if epoch_steps <= 0:
            raise ValueError(f"--epochs produced a non-positive step count: {epoch_steps}")
        args.steps = epoch_steps
        print(f"[epochs] epochs={args.epochs:g} total_groups={meta.total_groups} "
              f"mean_puzzle_examples={meta.mean_puzzle_examples:.4f} batch={args.batch} "
              f"-> steps={args.steps}")
    # REPRODUCIBLE MODEL INIT: the newly-initialised keys (C2 cross_attn, colour/repair heads) are
    # drawn from the global RNG at construction. With c2_gate_init>0 the step-0 baseline DEPENDS on
    # the random cross_attn draw, so without this seed an A/B that adds a module shifts the draw and
    # the baselines drift (observed: LODO changed base 33 vs 19 across runs).
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    loss_head, _, _ = pretrain.create_model(config, meta, rank=dist_ctx.rank, world_size=dist_ctx.world)
    loss_head.train()
    device = torch.device("cuda")
    return config, loader, meta, loss_head, device


def build_eval_loader(config):
    """Create a rank-independent loader so probes never advance the training iterator."""
    loader, _ = pretrain.create_dataloader(
        config, "train", 0, 1, test_set_mode=False,
        epochs_per_iter=1, global_batch_size=config.global_batch_size,
    )
    return loader


def _freeze_lodo_contract(batch: dict, seed: int, max_samples: int) -> dict:
    """Attach an immutable, local-RNG LODO fold contract without mutating ``batch``."""
    if "context_mask" not in batch:
        raise ValueError("cannot freeze LODO contract without context_mask")
    context_mask = batch["context_mask"]
    if not torch.is_tensor(context_mask) or context_mask.ndim != 2:
        raise ValueError("context_mask must be a Bool tensor with shape [B,M]")

    mask_cpu = context_mask.detach().to(device="cpu", dtype=torch.bool)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    scores = torch.rand(mask_cpu.shape, generator=generator)
    scores.masked_fill_(~mask_cpu, -1.0)
    holdout_idx = scores.argmax(dim=-1)
    aux_valid = mask_cpu.sum(dim=-1) >= 2

    if max_samples > 0:
        valid_indices = torch.nonzero(aux_valid, as_tuple=False).flatten()
        if valid_indices.numel() > max_samples:
            selected = valid_indices[
                torch.randperm(valid_indices.numel(), generator=generator)[:max_samples]]
            limited = torch.zeros_like(aux_valid)
            limited[selected] = True
            aux_valid = limited

    frozen = dict(batch)
    frozen["_force_lodo_eval"] = torch.ones(
        context_mask.shape[0], device=context_mask.device, dtype=torch.bool)
    frozen["_lodo_holdout_idx"] = holdout_idx.to(context_mask.device)
    frozen["_lodo_aux_valid"] = aux_valid.to(context_mask.device)
    return frozen


def collect_eval_batches(args, loader, device) -> list:
    """FIXED LODO eval: the SAME held-out demos scored every log step. Verbatim oracle 700-723.
    CROSS-RUN DETERMINISM: model creation consumes a config-dependent amount of RNG, which shifted
    which batches the loader yields here -- seed right before collecting so the frozen eval set is
    byte-identical across runs regardless of which model flags are on."""
    import numpy as _np
    import random as _random
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = _np.random.get_state()
    python_state = _random.getstate()
    try:
        torch.manual_seed(args.eval_seed)
        _np.random.seed(args.eval_seed)
        _random.seed(args.eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.eval_seed)
        eval_batches = []
        for _set, _cb, _g in loader:
            if "context_inputs" not in _cb or _cb["context_inputs"].shape[1] < 2:
                continue
            frozen = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in _cb.items()}
            frozen = _freeze_lodo_contract(
                frozen, seed=args.eval_seed + len(eval_batches), max_samples=args.batch)
            eval_batches.append(frozen)
            if len(eval_batches) >= args.eval_batches:
                break
        print(f"[fixed-eval] {len(eval_batches)} frozen batches, seed={args.eval_seed} "
              f"(same held-out demos scored every {args.log_every} steps)")
        return eval_batches
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        _np.random.set_state(numpy_state)
        _random.setstate(python_state)


def build_family_eval_sets(args, config, device):
    """PER-FAMILY EVAL setup (general-not-DSL MICROSCOPE). Verbatim oracle 757-814 EXCEPT the
    DRV-4 fix: the whole setup runs under a SAVED/RESTORED global RNG state (torch+numpy+random),
    so this MEASUREMENT flag no longer changes the TRAINING batch stream (the oracle re-seeded
    and left the state mutated -> runs with/without --per-family-eval were not trajectory-comparable)."""
    if not args.per_family_eval:
        return None
    import numpy as _npf
    import random as _rndf
    # DRV-4: save EVERY global RNG this setup touches; restore on all paths.
    _rng = torch.random.get_rng_state()
    _crng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    _nst = _npf.random.get_state()
    _rst = _rndf.getstate()
    family_eval_sets = None
    try:
        import sys as _psys
        _psys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.oracle_eval import _load_family_map, PROBE_DATASETS
        _aug0 = PROBE_DATASETS["aug0"]
        _id2hash, _hash2fam = _load_family_map(_aug0)
        if _id2hash is None:
            print("[per-family] skipped (no aug-0 identifiers.json / atlas CSV).")
        else:
            torch.manual_seed(args.eval_seed); _npf.random.seed(args.eval_seed); _rndf.seed(args.eval_seed)
            _fam_config = config.model_copy(update={"data_paths": [_aug0], "data_paths_test": []})
            _fam_loader, _ = pretrain.create_dataloader(
                _fam_config, "train", 0, 1, test_set_mode=False,
                epochs_per_iter=1, global_batch_size=config.global_batch_size)
            _rows = []                                              # (family, per-task row dict)
            for _s2, _cb2, _g2 in _fam_loader:
                if "context_inputs" not in _cb2 or _cb2["context_inputs"].shape[1] < 2:
                    continue
                _B = _cb2["inputs"].shape[0]
                _pid = _cb2.get("puzzle_identifiers")
                for _b in range(_B):
                    _idx = int(_pid[_b]) if _pid is not None else -1
                    _h = _id2hash[_idx] if 0 <= _idx < len(_id2hash) else None
                    _fam = _hash2fam.get(_h, "other")
                    _row = {k: (v[_b] if torch.is_tensor(v) and v.shape and v.shape[0] == _B else v)
                            for k, v in _cb2.items()}
                    _rows.append((_fam, _row))
                if len(_rows) >= args.per_family_collect:
                    break
            from collections import defaultdict as _ddf
            _by_fam = _ddf(list)
            for _fam, _row in _rows:
                _by_fam[_fam].append(_row)
            bs = config.global_batch_size
            family_eval_sets = {}
            for _fam, _frows in _by_fam.items():
                _bs_list = []
                for _i in range(0, len(_frows) - bs + 1, bs):
                    _grp = _frows[_i:_i + bs]
                    _batch = {}
                    for _k in _grp[0]:
                        if torch.is_tensor(_grp[0][_k]):
                            _batch[_k] = torch.stack([_r[_k] for _r in _grp]).to(device)
                        else:
                            _batch[_k] = _grp[0][_k]
                    _batch = _freeze_lodo_contract(
                        _batch,
                        seed=args.eval_seed + sum(len(v) for v in family_eval_sets.values()) + len(_bs_list),
                        max_samples=args.batch,
                    )
                    _bs_list.append(_batch)
                    if len(_bs_list) >= 15:      # cap per family (keeps baseline+end eval fast)
                        break
                if _bs_list:
                    family_eval_sets[_fam] = _bs_list
            print("[per-family] eval sets: " +
                  ", ".join(f"{f}={len(b) * bs}t" for f, b in family_eval_sets.items()))
    except Exception as _e:
        print(f"[per-family] setup failed ({type(_e).__name__}: {_e}); per-family eval disabled.")
        family_eval_sets = None
    finally:
        torch.random.set_rng_state(_rng)
        if _crng is not None:
            torch.cuda.set_rng_state_all(_crng)
        _npf.random.set_state(_nst)
        _rndf.setstate(_rst)
    return family_eval_sets


def select_scope(args, loss_head):
    """Trainable-param selection. Verbatim oracle branch chain 1240-1307 (tuples above)."""
    if args.train_scope == "v3-where-selector":
        # P3A Block 5: startup assertion is part of the scope itself -- a forbidden tensor in the
        # selection refuses to start (the V3-1 lesson: silent scope drift kills runs invisibly).
        params, _selector_names, _selector_bad = select_where_selector_params(
            (n, p) for n, p in loss_head.named_parameters() if p.requires_grad)
        if _selector_bad:
            raise SystemExit(
                f"[stage1 GATE] scope v3-where-selector selected forbidden tensors: {_selector_bad}")
        scope_label = "v3-where-selector(frozen-core;update-strengths-frozen)"
    elif args.train_scope == "v3-where":
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and any(k in n for k in V3_WHERE_NAMES)]
        scope_label = "v3-where(frozen-core,structure,pid;C2-selector-trainable)"
    elif args.train_scope == "v3-value":
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and any(k in n for k in V3_VALUE_NAMES)]
        scope_label = "v3-value(frozen-core,c2,structure,pid)"
    elif args.train_scope == "v3-transport":
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and any(k in n for k in V3_TRANSPORT_NAMES)]
        scope_label = "v3-transport(frozen-core,color,pid)"
    elif args.train_scope == "v3-color":
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and any(k in n for k in V3_COLOR_NAMES)]
        scope_label = "v3-color(frozen-structure,core)"
    elif args.train_scope == "v3-head":
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and any(k in n for k in V3_HEAD_NAMES)]
        scope_label = "v3-head(frozen-core)"
    elif args.train_scope == "v3-adapter":
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and any(k in n for k in V3_ADAPTER_NAMES)]
        scope_label = "v3-adapter(frozen-core)"
    elif args.train_scope == "quarantine":
        # ONLY the PID-quarantined candidate head. z_H, lm_head, structure levers, C2 -- all frozen:
        # the LODO gradient flows through candidate colour -> quarantine_* and stops there, so MAIN
        # cannot move and the abort guard is a formality. Full-strength lodo weight is safe here.
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and "quarantine_" in n]
        scope_label = "quarantine(everything-else-frozen)"
    elif args.train_scope == "v3-head+quarantine":
        # Union scope: V3 factored heads + structure levers + evidence proj AND the quarantined
        # candidate head, PLUS the zero-init INPUT-SIDE hint adapters when their flags are on --
        # previously these existed but were excluded here, so --frame-hint etc. sat at norm=0.0000
        # forever (measured, 3 runs). They perturb grid_features feeding the FROZEN core, so once
        # nonzero MAIN is no longer byte-invariant -- the MAIN-strict abort guard is load-bearing.
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and (any(k in n for k in V3_HEAD_NAMES)
                                          or any(k in n for k in HINT_ADAPTER_NAMES)
                                          or "quarantine_" in n)]
        scope_label = "v3-head+quarantine(frozen-core,+hint-adapters)"
    elif args.train_scope == "v3-adapter+quarantine":
        # TRM-focused union scope: the full V3 adapter stack + the quarantined candidate, with the
        # pretrained recurrence/lm floor frozen. Hot LR is opt-in per hint module via --hint-lr-names.
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and (any(k in n for k in V3_ADAPTER_NAMES)
                                          or "quarantine_" in n)]
        scope_label = "v3-adapter+quarantine(frozen-core,+adapters,+quarantine)"
    elif args.unified or args.train_scope == "all":
        # Train the FULL stack (TRM core + all adapters) so STRUCTURE can adapt: the output canvas
        # is lm_head(z_H), so a FROZEN lm_head/recursion leaves LODO pad/shape flat (proven: pad
        # stuck at ~5% for 100 steps adapter-only). Stability = lr 1e-5 + LR warmup + NaN guard.
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and "puzzle_emb" not in n]
        scope_label = "unified-full(core+adapters)" if args.unified else "all"
    else:
        print("[LEGACY-GATED] --train-scope delta: the pre-V3 delta-branch scope (most modules "
              "physically removed; kept runnable for provenance, never the default).")
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and any(k in n for k in LEGACY_DELTA_NAMES)]
        scope_label = "delta"
    if not params:
        raise SystemExit(
            f"[stage1] train scope '{scope_label}' selected 0 trainable tensors -- nothing would train. "
            f"The 'delta' scope names mostly-REMOVED legacy modules; use --train-scope v3-head (default), "
            f"v3-color, v3-where, v3-value, v3-transport, v3-adapter, or --unified.")
    if args.train_scope in ("v3-where", "v3-value", "v3-transport"):
        selected = {id(p) for p in params}
        selected_names = [n for n, p in loss_head.named_parameters() if id(p) in selected]
        forbidden = ("pid_task", "puzzle_emb", "lm_head", ".L_level.")
        if args.train_scope != "v3-transport":
            forbidden = forbidden + ("structure_",)
        if args.train_scope == "v3-transport":
            forbidden = forbidden + ("color_head", "color_evidence_proj")
        bad = [n for n in selected_names if any(k in n for k in forbidden)]
        if args.train_scope == "v3-value":
            bad.extend(n for n in selected_names
                       if ".c2.gate_global" in n or ".c2.gate_patch" in n)
        if bad:
            raise SystemExit(f"[stage1 GATE] scope {args.train_scope} selected forbidden tensors: {bad}")
    # HARD GATE (C-prime postmortem): --unified without an explicit scope MUST select the full stack.
    # The bug this catches: a scope default winning the branch chain above -> "unified" run that
    # silently trains a handful of frozen-core tensors (measured: 6 tensors, 500 wasted steps).
    _total_trainable = sum(1 for _n, _p in loss_head.named_parameters() if _p.requires_grad)
    if args.unified and not args.train_scope_explicit and len(params) < int(0.9 * _total_trainable):
        raise SystemExit(
            f"[stage1 GATE] --unified resolved to scope '{scope_label}' with only "
            f"{len(params)}/{_total_trainable} trainable tensors -- a silent frozen-core override. "
            f"Refusing to start; the full-stack param selection is miswired.")
    return params, scope_label


# ======================================================================================
# §6 OPTIMIZER — lr groups (struct/evidence/quarantine/hint), all wd=0 (lever-erosion rule).
# DRV-5 FIX (pre-registered): the split is UNCONDITIONAL. The oracle skipped it when
# struct_lr == evidence_lr == --lr (only reachable at --lr 1e-3), silently parking
# color_evidence_proj in the wd=0.01 default group -- lever-erosion reappearance #4.
# On every ledger command the oracle took the split branch, so groups are identical (G2/G3).
# ======================================================================================
def build_optimizer(args, loss_head, params, scope_label):
    # §15.9 boundary lever needs its OWN lr: a single zero-init proj must climb ~2-5 logits to beat
    # the frozen lm_head; at the global lr it trains far too slowly frozen-core (proven: pad flat).
    struct_lr = args.struct_lr if args.struct_lr is not None else args.lr
    # FIX A: the colour EVIDENCE projection gets its own group -- fresh zero-init tensors cannot
    # train at the core-safe lr (measured: 6e-6 -> 6.7e-3 weight in 300 steps at lr 1e-5 = inert).
    evidence_lr = args.evidence_lr if args.evidence_lr is not None else (
        args.struct_lr if args.struct_lr is not None else 1e-3)
    quarantine_lr = args.quarantine_lr if args.quarantine_lr is not None else args.lr
    c2_lr = float(args.c2_lr)
    c2_gate_lr = float(args.c2_gate_lr)
    # Hint adapters are zero-init like the evidence proj; default to the evidence lr.
    hint_lr = args.hint_lr if args.hint_lr is not None else evidence_lr
    _struct_ids = {id(p) for n, p in loss_head.named_parameters()
                   if ("structure_relmap_proj" in n
                       or "structure_outside_proj" in n
                       or "structure_eos_proj" in n
                       or "structure_pairdelta_proj" in n
                       or (args.train_scope == "v3-transport" and "structure_head." in n))}
                   # D9 and M4: zero-init/frozen-core structure tensors share the no-decay lever rules.
    _evidence_ids = {id(p) for n, p in loss_head.named_parameters()
                     if ("color_evidence_proj" in n or "color_head_mlp_" in n)}
    # quarantine_* ALWAYS rides its own wd=0 group: the warm-init +4/+8 columns are held values
    # (lever-erosion rule). Before this split, a quarantine-scope run that tripped the old
    # conditional silently put them in the wd=0.01 default group -- the third lever-erosion
    # reappearance.
    _quar_ids = {id(p) for n, p in loss_head.named_parameters() if "quarantine_" in n}
    _c2_gate_ids = {id(p) for n, p in loss_head.named_parameters()
                    if any(k in n for k in V3_WHERE_GATE_NAMES)}
    _c2_ids = {id(p) for n, p in loss_head.named_parameters()
               if any(k in n for k in V3_WHERE_C2_NAMES) and id(p) not in _c2_gate_ids}
    # Hint adapters: own wd=0 group at hint_lr. --hint-lr-names restricts WHICH hint modules get
    # the hot lr; the rest ride at --lr.
    if args.hint_lr_names.strip().lower() == "all":
        _hot_hint_names = HINT_ADAPTER_NAMES
    else:
        _requested = tuple(s.strip() for s in args.hint_lr_names.split(",") if s.strip())
        _unknown = [s for s in _requested if s not in HINT_ADAPTER_NAMES]
        if _unknown:
            raise SystemExit(f"[stage1] --hint-lr-names unknown module(s) {_unknown}; "
                             f"valid: {list(HINT_ADAPTER_NAMES)} or 'all'")
        _hot_hint_names = _requested
    _hint_ids = {id(p) for n, p in loss_head.named_parameters()
                 if any(k in n for k in _hot_hint_names)}
    struct_params = [p for p in params if id(p) in _struct_ids]
    evidence_params = [p for p in params if id(p) in _evidence_ids]
    quar_params = [p for p in params if id(p) in _quar_ids]
    c2_params = [p for p in params if id(p) in _c2_ids]
    c2_gate_params = [p for p in params if id(p) in _c2_gate_ids]
    hint_params = [p for p in params if id(p) in _hint_ids
                   and id(p) not in _struct_ids and id(p) not in _evidence_ids
                   and id(p) not in _quar_ids and id(p) not in _c2_ids
                   and id(p) not in _c2_gate_ids]
    other_params = [p for p in params
                     if id(p) not in _struct_ids and id(p) not in _evidence_ids
                     and id(p) not in _quar_ids and id(p) not in _hint_ids
                     and id(p) not in _c2_ids and id(p) not in _c2_gate_ids]
    # weight_decay=0 on the lever groups: AdamW decay is lr*wd*theta, and at struct_lr=1e-2 that is
    # 0.17/step on the +-1000 warm-init levers -- MEASURED leak 1732.04 -> 1717.85 over 90 steps.
    # Levers must climb from zero or HOLD a warm value; decay toward zero is the opposite of both.
    groups = [{"params": other_params, "lr": args.lr}]
    _base_group_name = (
        "head" if args.train_scope in ("v3-where", "v3-value")
        else "transport" if args.train_scope == "v3-transport"
        else "core"
    )
    _gnames = [_base_group_name]
    if struct_params:
        groups.append({"params": struct_params, "lr": struct_lr, "weight_decay": 0.0})
        _gnames.append("struct")
    if evidence_params:
        groups.append({"params": evidence_params, "lr": evidence_lr, "weight_decay": 0.0})
        _gnames.append("evidence")
    if quar_params:
        groups.append({"params": quar_params, "lr": quarantine_lr, "weight_decay": 0.0})
        _gnames.append("quarantine")
    if c2_params:
        groups.append({"params": c2_params, "lr": c2_lr})
        _gnames.append("c2")
    if c2_gate_params:
        groups.append({"params": c2_gate_params, "lr": c2_gate_lr, "weight_decay": 0.0})
        _gnames.append("c2-gate")
    if hint_params:
        groups.append({"params": hint_params, "lr": hint_lr, "weight_decay": 0.0})
        _gnames.append("hint")
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
    opt._trv2_group_names = _gnames   # M-B/M-D/M-F read the group labels from here
    print(f"[struct-lr] structure levers ({len(struct_params)} tensors) lr={struct_lr}, wd=0; "
          f"[evidence-lr] color_evidence_proj/mlp ({len(evidence_params)} tensors) lr={evidence_lr}, wd=0; "
          f"[quarantine-lr] quarantine_* ({len(quar_params)} tensors) lr={quarantine_lr}, wd=0; "
          f"[c2-lr] WHERE attention/demo/relmap ({len(c2_params)} tensors) lr={c2_lr}; "
          f"[c2-gate-lr] token gate ({len(c2_gate_params)} tensors) lr={c2_gate_lr}, wd=0; "
          f"[hint-lr] hint adapters ({len(hint_params)} tensors: {','.join(_hot_hint_names) if hint_params else 'none'}) "
          f"lr={hint_lr}, wd=0; rest at lr={args.lr}.")
    print(f"[stage1-probe] steps={args.steps} lr={args.lr} "
          f"batch={args.batch} scope={scope_label} lodo_target={args.lodo_weight} "
          f"lodo_pad={args.lodo_pad_weight} lodo_eos={args.lodo_eos_weight} "
          f"lodo_copy={args.lodo_copy_weight} lodo_changed={args.lodo_changed_weight} "
          f"lodo_warmup={args.lodo_warmup_steps} lodo_ramp={args.lodo_ramp_steps} | "
          f"trainable tensors={len(params)}")
    return opt


def scheduled_lodo_weight(args, step: int) -> float:
    if args.lodo_weight <= 0:
        return 0.0
    if step < args.lodo_warmup_steps:
        return 0.0
    if args.lodo_ramp_steps <= 0:
        return float(args.lodo_weight)
    frac = min(1.0, (step - args.lodo_warmup_steps + 1) / max(1, args.lodo_ramp_steps))
    return float(args.lodo_weight) * frac


# ======================================================================================
# §7 EVAL + PROBES — the oracle's nested closures as real functions over the run state S
# (SimpleNamespace: args, config, loss_head, device, eval_batches, family_eval_sets, ...).
# Bodies verbatim; only the capture mechanism changed.
# ======================================================================================
@contextmanager
def _evaluation_mode(module):
    """Temporarily evaluate ``module`` without changing its caller-owned mode."""
    was_training = module.training
    module.eval()
    try:
        yield module
    finally:
        module.train(was_training)


@contextmanager
def _seeded_evaluation(module, seed: int):
    """Run a deterministic probe without consuming caller-owned RNG or mode state."""
    import random

    import numpy as np

    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        random.seed(int(seed))
        with _evaluation_mode(module):
            yield module
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _fixed_eval_batch_identity(batch: dict, batch_index: int, seed: int) -> str:
    """Failure-only identity string for reproducing one frozen evaluation batch."""
    puzzle_identifiers = batch.get("puzzle_identifiers")
    if torch.is_tensor(puzzle_identifiers):
        pid_text = str([int(x) for x in puzzle_identifiers.detach().cpu().reshape(-1).tolist()])
    elif puzzle_identifiers is None:
        pid_text = "<missing>"
    else:
        pid_text = str(puzzle_identifiers)
    return (
        f"fixed_eval_batch={int(batch_index)} seed={int(seed)} "
        f"puzzle_identifiers={pid_text}"
    )


@torch.no_grad()
def eval_fixed(S) -> dict:
    """Macro-average the health panel over the frozen eval batches (seeded holdout).
    Metrics are stored as mean*count, so each batch is normalized by its OWN count FIRST
    (recovering the true per-batch value) and then averaged. This avoids >100% inflation when
    the halt-count differs from the metric denominator (e.g. a trained q_head changes halting).
    DRV-2 detector: warns ONCE if a key is present in only SOME batches (the macro-average
    silently deflates such keys by the missing batches). Detection only; math unchanged."""
    args = S.args
    agg: dict = {}
    seen: dict = {}
    nb = 0
    for i, eb in enumerate(S.eval_batches):
        batch_seed = args.eval_seed + i
        with _seeded_evaluation(S.loss_head, batch_seed):
            with torch.device(S.device):
                c0 = S.loss_head.initial_carry(eb)
            try:
                _c, _l, m, _, _ = S.loss_head(carry=c0, batch=eb, return_keys=[])
            except FloatingPointError as exc:
                raise FloatingPointError(
                    f"{_fixed_eval_batch_identity(eb, i, batch_seed)}; {exc}") from exc
        cnt = float(m["count"].item()) if "count" in m else 0.0
        denom = cnt if cnt > 0 else 1.0
        nb += 1
        for k, v in m.items():
            if torch.is_tensor(v) and v.numel() == 1:
                agg[k] = agg.get(k, 0.0) + v.detach().float() / denom   # per-batch true value
                seen[k] = seen.get(k, 0) + 1
    if not getattr(S, "drv2_warned", False):
        _partial = sorted(k for k, c in seen.items() if c != nb)
        if _partial:
            print(f"[eval-fixed WARNING DRV-2] {len(_partial)} metric key(s) present in only SOME "
                  f"frozen batches -> the macro-average DEFLATES them: "
                  f"{_partial[:8]}{' ...' if len(_partial) > 8 else ''}")
        S.drv2_warned = True
    # panel computes g(k)=agg[k]/agg["count"]; set count=nb so g = mean of per-batch values.
    agg["count"] = torch.tensor(float(nb))
    return agg


@torch.no_grad()
def eval_per_family(S, tag: str) -> None:
    args = S.args
    if not S.family_eval_sets:
        return
    print(f"\n[PER-FAMILY EVAL @ {tag}]  LODO changed/unchg/color_exact/strict + WHERE/VALUE split")
    print(f"  (judge VALUE binding on conditional_recolor ONLY -- it is the family with proven "
          f"extractable evidence; rearrangement is Lane-B, size_change is structure-solved)")
    order = ["conditional_recolor", "size_change", "rearrangement", "other"]
    for fam in order + [f for f in S.family_eval_sets if f not in order]:
        bset = S.family_eval_sets.get(fam)
        if not bset:
            continue
        agg: dict = {}; nb = 0
        for i, eb in enumerate(bset):
            with _seeded_evaluation(S.loss_head, args.eval_seed + i):
                with torch.device(S.device):
                    c0 = S.loss_head.initial_carry(eb)
                _c, _l, m, _, _ = S.loss_head(carry=c0, batch=eb, return_keys=[])
            cnt = float(m["count"].item()) if "count" in m else 0.0
            denom = cnt if cnt > 0 else 1.0
            nb += 1
            for k, v in m.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    agg[k] = agg.get(k, 0.0) + v.detach().float() / denom
        gg = lambda k: (float(agg[k]) / nb) if k in agg else float("nan")
        print(f"  {fam:20s} n={len(bset) * S.config.global_batch_size:3d}  "
              f"changed={gg('lodo_changed_color_acc_pct'):5.1f} "
              f"unchg={gg('lodo_unchanged_color_acc_pct'):5.1f} "
              f"exact={gg('lodo_color_exact_pct'):5.1f} "
              f"strict={gg('lodo_strict_exact_pct'):5.1f} | "
              f"where_f1={gg('lodo_where_f1_pct'):5.1f} "
              f"value_pred={gg('lodo_value_on_pred_changed_pct'):5.1f} "
              f"chgCEc={gg('lodo_changed_color_ce'):5.2f} "
              f"copyCEc={gg('lodo_copy_color_ce'):5.2f}")


@torch.no_grad()
def eval_selector(S, tag: str) -> None:
    """VERIFY-AND-SELECT (composition-C, Take #1): per task, score the two MODEL candidates -- the
    lm_head FLOOR vs the factored V3 CANDIDATE -- by LODO reconstruction of the held-out demo,
    commit the best (tie->floor). ONE forward per batch: under --floor-candidate-split the loss
    panel already emits per-example floor AND candidate views from the same aux pass."""
    args = S.args
    if not args.selector_eval:
        return
    if not args.floor_candidate_split:
        print(f"\n[SELECTOR @ {tag}] SKIPPED: needs --floor-candidate-split (the split exposes the "
              f"per-example floor vs candidate views the selector compares).")
        return
    def _per_example(batches):
        ce, cs, fe, fs = [], [], [], []
        for i, eb in enumerate(batches):
            with _seeded_evaluation(S.loss_head, args.eval_seed + i):
                with torch.device(S.device):
                    c0 = S.loss_head.initial_carry(eb)
                _c, _l, m, _, _ = S.loss_head(carry=c0, batch=eb, return_keys=[])
            pe = m.get("lodo_color_exact_per_ex"); fpe = m.get("lodo_floor_color_exact_per_ex")
            if pe is None or fpe is None:
                continue
            ce.append(pe.detach().float().cpu())
            fe.append(fpe.detach().float().cpu())
            _cs = m.get("lodo_color_cellsim_per_ex"); _fs = m.get("lodo_floor_color_cellsim_per_ex")
            cs.append((_cs if _cs is not None else torch.zeros_like(pe)).detach().float().cpu())
            fs.append((_fs if _fs is not None else torch.zeros_like(pe)).detach().float().cpu())
        if not ce:
            return None
        return torch.cat(ce), torch.cat(cs), torch.cat(fe), torch.cat(fs)

    # Run on ALL frozen eval + EACH per-family set: the floor-safe RECOVERY only SHOWS where the
    # floor solves >0 (eval_batches alone can be floor=0).
    sets = [("ALL", S.eval_batches)]
    if S.family_eval_sets:
        order = ["conditional_recolor", "size_change", "rearrangement", "other"]
        sets += [(f, S.family_eval_sets[f]) for f in order if f in S.family_eval_sets]
    print(f"\n[SELECTOR @ {tag}]  VERIFY-AND-SELECT (composition-C) | floor vs V3 candidate | tie->FLOOR")
    print(f"  {'set':20s}  floor%  cand%  SELECT%  sel>=floor  cand_chosen")
    for name, batches in sets:
        res = _per_example(batches)
        if res is None:
            continue
        eh, sh, ef, sf = res
        n = ef.numel()
        # per-task choice: higher (exact, then graded cellsim); TIE -> FLOOR (never net-negative)
        head_better = (eh > ef) | ((eh == ef) & (sh > sf + 1e-6))
        sel = torch.where(head_better, eh, ef)
        fp = ef.mean().item() * 100; hp = eh.mean().item() * 100; sp = sel.mean().item() * 100
        guar = (sel >= ef - 1e-9).float().mean().item() * 100
        print(f"  {name:20s} {fp:6.1f} {hp:6.1f} {sp:7.1f} {guar:9.1f}%  {int(head_better.sum()):3d}/{n}")
    print(f"  (sel>=floor MUST be 100 = the R13 kill; the floor-safe RECOVERY shows where floor>0, e.g.")
    print(f"   rearrangement. 1 fold/forward -> SELECT = headroom; deployable non-peeking = multi-fold")
    print(f"   LOOCV verify_and_select_candidates.py, PROVEN 2.7>=1.5, 100%.)")


# ---- Z_H CONDITIONING CHECK (general-not-DSL diagnostic #1): the colour head reads
# grid_z = z_H[grid positions]. If the RULE never reaches the recursion, grid_z is just an
# encoding of the target INPUT and the head can only fall back to the deterministic lookup.
# MEASUREMENT, not a new lever. ----
@torch.no_grad()
def zh_conditioning_report(S, tag: str = "") -> None:
    args = S.args
    loss_head = S.loss_head
    _CO = 2  # COLOR_OFFSET: tokens >=2 are colours (0=PAD,1=EOS)
    from models.losses_fvr import IGNORE_LABEL_ID as _IGN
    inner = loss_head.model.inner
    was_training = loss_head.training
    loss_head.eval()  # deterministic forward so any movement is demo signal, not dropout noise
    # DEMO roll = EVERY demo-derived tensor, prefix-driven so a new context feature can never be
    # forgotten here again (the old static tuple missed context_output_rel_maps + frame_label ->
    # the probe measured a mixed contract no training path ever sees).
    _INP = ("inputs", "input_visual_features", "rel_maps")

    def _demo_keys(batch):
        return [k for k in batch if k.startswith("context_")] + ["frame_label"]

    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)

    def _pass(batch):
        bs = batch["inputs"].shape[0]
        carry = inner.fresh_carry(bs)
        ie, _, rm, _ = inner._input_embeddings(batch)
        z_H, _ = inner._run_recurrence(carry, ie, seq_info)
        grid_z = z_H[:, inner.puzzle_emb_len:].float()           # [B,P,D] -- what the head reads
        logits, extras = inner._output_logits(z_H, batch, rel_maps=rm)
        # Under --floor-candidate-split `logits` IS the frozen floor: stablemax-saturated, so its
        # argmax flipping ~0 under demo swap is EXPECTED. The CANDIDATE head is what LODO trains --
        # its flip is the number that must move.
        cand = extras.get("c2_candidate_logits", logits)
        return grid_z, logits, cand

    def _roll(batch, keys):
        bs = batch["inputs"].shape[0]
        perm = torch.roll(torch.arange(bs, device=batch["inputs"].device), 1)
        out = dict(batch)
        for k in keys:
            if k in batch and torch.is_tensor(batch[k]) and batch[k].shape[:1] == (bs,):
                out[k] = batch[k][perm]
        return out

    def _cmp(a, b, mask):
        # Restrict to the target's COLOUR cells (mask) + drop degenerate positions. PAD positions
        # have near-constant low-norm z_H where cosine is pure noise; the colour cells are the
        # meaningful, well-conditioned ones.
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)        # [B,P]
        rel = (a - b).norm(dim=-1) / (a.norm(dim=-1) + 1e-6)             # [B,P]
        good = mask & (a.norm(dim=-1) > 1e-3) & (b.norm(dim=-1) > 1e-3)
        good = good & torch.isfinite(cos) & torch.isfinite(rel)
        if int(good.sum()) == 0:
            return None
        cosv, relv = cos[good], rel[good]
        return cosv.mean().item(), relv.mean().item(), (cosv < 0.99).float().mean().item()

    def _flip(lr, ls, mask):
        ar, as_ = lr.argmax(-1), ls.argmax(-1)
        colour = (ar >= _CO) & mask                            # changed colour PREDICTIONS at real cells
        denom = colour.sum().clamp_min(1)
        return (((ar != as_) & colour).sum().float() / denom).item()

    def _eval_set(batches):
        acc = {k: 0.0 for k in ("cs", "cd", "rd", "md", "fd", "cfd", "ci", "ri", "fi", "cfi", "ms")}
        n = 0
        for b in batches:
            if b["inputs"].shape[0] < 2:
                continue
            mask = b["inputs"] >= _CO                           # real target colour cells [B,P]
            gz_r, lg_r, cd_r = _pass(b)
            gz_r2, _, _ = _pass(b)                               # determinism sanity
            gz_d, lg_d, cd_d = _pass(_roll(b, _demo_keys(b)))   # demos shuffled across tasks
            gz_i, lg_i, cd_i = _pass(_roll(b, _INP))            # target inputs shuffled (reference)
            rs, rd_, ri_ = _cmp(gz_r, gz_r2, mask), _cmp(gz_r, gz_d, mask), _cmp(gz_r, gz_i, mask)
            if rs is None or rd_ is None or ri_ is None:
                continue
            acc["cs"] += rs[0]
            acc["cd"] += rd_[0]; acc["rd"] += rd_[1]; acc["md"] += rd_[2]
            acc["ci"] += ri_[0]; acc["ri"] += ri_[1]
            acc["fd"] += _flip(lg_r, lg_d, mask)
            acc["fi"] += _flip(lg_r, lg_i, mask)
            acc["cfd"] += _flip(cd_r, cd_d, mask)               # CANDIDATE flip under demo swap
            acc["cfi"] += _flip(cd_r, cd_i, mask)
            if "labels" in b:                                   # MAIN floor safety at this scale
                t = b["labels"].long()
                keep = t != _IGN
                acc["ms"] += ((lg_r.argmax(-1) == t) | ~keep).all(-1).float().mean().item()
            n += 1
        if n == 0:
            return None
        return {k: v / n for k, v in acc.items()}, n

    print(f"\n[Z_H CONDITIONING CHECK{(' @ ' + tag) if tag else ''}]  does the recursion's z_H carry the demo-derived rule?")
    print("  grid_z = z_H[grid positions] = exactly what the colour head reads.")
    print("  DEMO-shuffle = same target, OTHER task's demos. INPUT-shuffle(ref) = other target, same demos.")
    print("  read: DEMO cos~1.000 / relL2~0 / flip~0  AND  demo/input relL2 ~0%  =>  z_H IGNORES the demos.\n")
    header_done = False
    sets = [("ALL", S.eval_batches)]
    if S.family_eval_sets:
        order = ["conditional_recolor", "size_change", "rearrangement", "other"]
        sets += [(f, S.family_eval_sets[f]) for f in order if f in S.family_eval_sets]
    for name, batches in sets:
        res = _eval_set(batches)
        if res is None:
            continue
        a, n = res
        if not header_done:
            print(f"  sanity real-vs-real cos={a['cs']:.4f} (want 1.0000 -> deterministic, movement below is real)")
            header_done = True
        ratio = a["rd"] / (a["ri"] + 1e-6)
        print(f"  {name:20s} nb={n:2d} | DEMO cos={a['cd']:.3f} relL2={a['rd']:.3f} moved={a['md']*100:4.1f}% "
              f"flip={a['fd']*100:4.1f}% candflip={a['cfd']*100:4.1f}% | INPUT(ref) cos={a['ci']:.3f} "
              f"relL2={a['ri']:.3f} flip={a['fi']*100:4.1f}% | demo/input relL2={ratio*100:5.1f}%")
    print("\n  -> flip = FLOOR argmax (saturated; ~0 is expected). candflip = the trained candidate head:")
    print("     if candflip AND demo/input relL2 stay ~0, the rule is NOT reaching the recursion/head.")

    # ---- FORCED-SIGNAL SWEEP (--zh-amp): amplify the demo->z_H injections and watch the same
    # numbers. Restores scale=1.0 afterwards. ----
    amp_list = [float(x) for x in args.zh_amp.split(",") if x.strip()] if args.zh_amp else []
    if amp_list:
        def _set_amp(k: float) -> None:
            inner._demo_injection_scale = k
            if getattr(inner, "c2", None) is not None:
                inner.c2._demo_injection_scale = k
        print("\n  [FORCED-SIGNAL SWEEP] demo->z_H injections (C2 update, PairDelta rule_vec, frame hint) x K")
        print(f"  {'K':>6}  {'demo_relL2':>10}  {'demo/input%':>11}  {'flip%':>6}  {'candflip%':>9}  {'MAIN_strict%':>12}")
        for k in amp_list:
            _set_amp(k)
            res = _eval_set(S.eval_batches)
            if res is None:
                continue
            a, _n = res
            ratio = a["rd"] / (a["ri"] + 1e-6)
            print(f"  {k:6.1f}  {a['rd']:10.4f}  {ratio*100:11.1f}  {a['fd']*100:6.1f}  "
                  f"{a['cfd']*100:9.1f}  {a['ms']*100:12.1f}")
        _set_amp(1.0)
        print("  read: relL2 grows with K + flips ~0 -> path ALIVE, signal too weak (scale/lr/loss);")
        print("        flat at the largest K -> path disconnected; MAIN_strict drops -> uncontrolled.")
    if was_training:
        loss_head.train()


@torch.no_grad()
def main_recovery_report(S, tag: str) -> None:
    """FLOOR RECOVERABILITY with optional input evidence ON versus fully scaled OFF.

    Scale zero removes target/support relmaps, C2 updates, broadcast hints, visual-rule
    residuals, and C2-derived PID modulation. OFF ~100 means the recurrent floor is intact;
    OFF <100 means parameters outside those evidence residuals caused a real regression.
    """
    from models.losses_fvr import IGNORE_LABEL_ID as _IGN
    loss_head = S.loss_head
    inner = loss_head.model.inner
    was_training = loss_head.training
    loss_head.eval()
    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)

    def _set_amp(k: float) -> None:
        inner._demo_injection_scale = k
        if getattr(inner, "c2", None) is not None:
            inner.c2._demo_injection_scale = k

    def _strict() -> float:
        tot, n = 0.0, 0
        for b in S.eval_batches:
            ie, _, rm, _ = inner._input_embeddings(b)
            z_H, _ = inner._run_recurrence(inner.fresh_carry(b["inputs"].shape[0]), ie, seq_info)
            lg, _ = inner._output_logits(z_H, b, rel_maps=rm)
            t = b["labels"].long()
            keep = t != _IGN
            tot += ((lg.argmax(-1) == t) | ~keep).all(-1).float().mean().item()
            n += 1
        return tot / max(n, 1)

    _set_amp(1.0); on = _strict()
    _set_amp(0.0); off = _strict()
    _set_amp(1.0)
    verdict = ("floor INTACT -- deploy from the injection-off forward; ON-drift is not a deployment loss"
               if off >= 0.999 else
               "floor DAMAGED even with injections off -- real regression, investigate")
    print(f"\n[floor-recovery @ {tag}] MAIN strict: input evidence ON={on * 100:.1f}%  "
          f"OFF={off * 100:.1f}%  ({verdict})")
    if was_training:
        loss_head.train()


@torch.no_grad()
def pid_null_report(S, tag: str) -> None:
    """NECESSITY-PRESSURE READ: MAIN strict on the frozen eval batches with the true PID vs the
    blank PID(0). null-pid strict IS demo-only performance -- under --pid-dropout this number
    RISING is the model reconstructing the mapping from the demos (the C-prime success metric)."""
    args = S.args
    if not (args.pid_null_eval or args.pid_dropout > 0):
        return
    from models.losses_fvr import IGNORE_LABEL_ID as _IGN
    loss_head = S.loss_head
    inner = loss_head.model.inner
    was_training = loss_head.training
    loss_head.eval()
    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)

    def _strict(null_pid: bool) -> float:
        tot, n = 0.0, 0
        for b in S.eval_batches:
            bb = dict(b)
            if null_pid:
                bb["puzzle_identifiers"] = torch.zeros_like(b["puzzle_identifiers"])
            ie, _, rm, _ = inner._input_embeddings(bb)
            z_H, _ = inner._run_recurrence(inner.fresh_carry(bb["inputs"].shape[0]), ie, seq_info)
            lg, _ = inner._output_logits(z_H, bb, rel_maps=rm)
            t = b["labels"].long()
            keep = t != _IGN
            tot += ((lg.argmax(-1) == t) | ~keep).all(-1).float().mean().item()
            n += 1
        return tot / max(n, 1)

    with_pid = _strict(False)
    null_pid = _strict(True)
    print(f"\n[pid-null @ {tag}] MAIN strict: pid={with_pid * 100:.1f}%  null-pid={null_pid * 100:.1f}%  "
          f"(null-pid = demo-only performance; rising under --pid-dropout = necessity pressure working)")
    if was_training:
        loss_head.train()


@torch.no_grad()
def shape_debug(S) -> None:
    import collections
    loss_head = S.loss_head
    inner = loss_head.model.inner
    if not getattr(inner.config, "c2_shape_head", False):
        print("[shape-debug] c2_shape_head OFF; pass --cross-demo-shape."); return
    was_training = loss_head.training
    loss_head.eval()
    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    tot = hc = wc = 0
    hoff = collections.Counter(); hpv = collections.Counter(); wpv = collections.Counter()
    rows = []
    for eb in S.eval_batches:
        bs = eb["inputs"].shape[0]
        carry = inner.fresh_carry(bs)
        ie, _, _, _ = inner._input_embeddings(eb)
        z_H, _ = inner._run_recurrence(carry, ie, seq_info)
        h_logits, w_logits = inner._shape_logits(z_H, eb)
        h_pred = h_logits.argmax(-1); w_pred = w_logits.argmax(-1)
        h_tgt = eb["target_height"].long() - 1; w_tgt = eb["target_width"].long() - 1
        for b in range(bs):
            ht, wt, hp, wp = int(h_tgt[b]), int(w_tgt[b]), int(h_pred[b]), int(w_pred[b])
            tot += 1; hc += int(hp == ht); wc += int(wp == wt)
            hoff[hp - ht] += 1; hpv[hp] += 1; wpv[wp] += 1
            if len(rows) < 14: rows.append((ht + 1, wt + 1, hp + 1, wp + 1))
    print(f"\n[SHAPE-DEBUG] n={tot}  h_acc={100*hc/max(tot,1):.1f}%  w_acc={100*wc/max(tot,1):.1f}%  (untrained head)")
    print("  target(h,w) -> pred(h,w):", rows)
    print("  h_pred value histogram (top):", hpv.most_common(6))
    print("  w_pred value histogram (top):", wpv.most_common(6))
    print("  (h_pred - h_target) offset histogram (top):", hoff.most_common(6))
    if was_training:
        loss_head.train()


# ======================================================================================
# §8 PANEL — the health readout, verbatim oracle 1404-1595. Returns False => abort.
# DRV-1 FIX inside V2TAIL: offset/width from trm_fvr_v2.evidence_slice (schema-owned);
# the oracle's hand-summed widths (imported from the OLD modules) survive as the fallback.
# ======================================================================================
def _fusion_compression(raw: float, applied: float) -> float:
    """Block 3: how hard the bounded-fusion rho clamp squashed an evidence residual this eval.
    compression = raw/applied norm ratio; raw < 1e-6 (dead or disabled lane) or any non-finite
    input reads as 1.0 (no signal, never a warning)."""
    import math
    if not (math.isfinite(raw) and math.isfinite(applied)) or raw < 1e-6:
        return 1.0
    return raw / max(applied, 1e-6)


def _fusion_compression_update(streaks: dict, comp: dict, warn: float, patience: int) -> list:
    """Block 3 streak bookkeeping (pure, testable): +1 per lane per eval above threshold, reset to 0
    below. Returns the lanes at/over patience THIS eval -- so the warning first fires on the
    patience-th consecutive exceed and keeps firing while the condition persists. Never aborts."""
    patience = max(1, int(patience))
    fired = []
    for lane, cval in comp.items():
        if warn > 0 and cval > warn:
            streaks[lane] = streaks.get(lane, 0) + 1
            if streaks[lane] >= patience:
                fired.append(lane)
        else:
            streaks[lane] = 0
    return fired


def panel(S, step: int, loss_val: float, m: dict, lodo_w: float,
          run_stats: dict | None = None) -> bool:
    args = S.args
    loss_head = S.loss_head
    c = float(m["count"].item()) if "count" in m else 1.0
    c = c if c > 0 else 1.0
    g = lambda k: (float(m[k].item()) / c) if k in m else float("nan")

    def pct_s(k: str, den_k: str | None = None) -> str:
        if den_k is not None and (den_k not in m or g(den_k) <= 0):
            return "NA"
        return f"{g(k):.1f}"

    print(f"\n[step {step:>3}] loss={loss_val:8.2f}  lodo_w={lodo_w:.4f} "
          f"kl={g('d_kl_keep'):.2f} keep%={g('d_preserved_correct_frac') * 100:.0f}")
    print(f"  MAIN  %: color={pct_s('main_color_acc_pct')} changed={pct_s('main_changed_color_acc_pct')} "
          f"pad={pct_s('main_pad_acc_pct', 'n_pad_cells')} eos={pct_s('main_eos_acc_pct', 'n_eos_cells')} "
          f"shape={pct_s('main_shape_exact_pct')} strict={pct_s('main_strict_exact_pct')}")
    print(f"  LODO  %: changed={pct_s('lodo_changed_color_acc_pct')} unchg={pct_s('lodo_unchanged_color_acc_pct')} "
          f"color={pct_s('lodo_color_acc_pct')} color_exact={pct_s('lodo_color_exact_pct')} "
          f"pad={pct_s('lodo_pad_acc_pct', 'n_lodo_pad_cells')} v2pad={pct_s('lodo_valid_to_pad_pct', 'n_lodo_color_cells')} "
          f"v2eos={pct_s('lodo_valid_to_eos_pct', 'n_lodo_color_cells')} "
          f"eos={pct_s('lodo_eos_acc_pct', 'n_lodo_eos_cells')} "
          f"shape_task={pct_s('lodo_task_shape_exact_pct')} strict_task={pct_s('lodo_task_strict_exact_pct')} "
          f"close={pct_s('lodo_close_pct')} strict_grid={pct_s('lodo_strict_exact_pct')}")
    # WHERE vs VALUE attribution: where_f1 = does the head change the RIGHT cells; value_pred =
    # of the cells it changed, correct-colour rate; value_true = changed acc on TRUE transform
    # cells. where HIGH + value LOW = binding failure; where LOW = selection failure.
    # chgCE/gap are the GRADED trend (visible long before color_exact moves).
    if "lodo_where_f1_pct" in m:
        # chgCEc/copyCEc = COLOUR-CHOICE CE (log_softmax over the 10 colour channels -> the
        # candidate's floor anchor cancels exactly). lodo_raw = the actual training objective
        # on the fixed eval. Read: chgCEc FALLING with copyCEc flat = binding is being learned;
        # copyCEc RISING = calibration eroding the copy basin.
        print(f"  WV    %: where_f1={pct_s('lodo_where_f1_pct')} "
              f"value_pred={pct_s('lodo_value_on_pred_changed_pct')} "
              f"value_true={pct_s('lodo_changed_color_acc_pct')} "
              f"trans_cover={g('c2_transition_hint_coverage') * 100:.1f} "
              f"chgCEc={g('lodo_changed_color_ce'):.3f} "
              f"copyCEc={g('lodo_copy_color_ce'):.3f} "
              f"lodo_raw={g('c2_delta_lodo_raw'):.2f} "
              f"chgCE={g('c2_delta_contrast_real_changed'):.2f} "
              f"gap={g('c2_delta_contrast_gap'):+.3f}")
        if "c2_value_v2_support_coverage" in m:
            print(f"  VALV2 : chg_rate[Tchg]={g('c2_value_v2_change_rate_on_changed'):.3f} "
                  f"chg_rate[Tcopy]={g('c2_value_v2_change_rate_on_copy'):.3f} "
                  f"copy_rate[Tcopy]={g('c2_value_v2_copy_rate_on_copy'):.3f} "
                  f"coverage={g('c2_value_v2_support_coverage'):.3f} "
                  f"entropy_conf={g('c2_value_v2_entropy_conf'):.3f} "
                  f"margin={g('c2_value_v2_margin'):.3f} "
                  f"where_mass={g('c2_value_v2_where_mass'):.3f}")
            if "c2_value_v2_aux_raw" in m:
                print(f"  V2AUX : raw={g('c2_value_v2_aux_raw'):.3f} "
                      f"chgCE={g('c2_value_v2_aux_changed_ce'):.3f} "
                      f"copyCE={g('c2_value_v2_aux_copy_ce'):.3f} "
                      f"chgAcc={g('c2_value_v2_aux_changed_acc') * 100:.1f} "
                      f"copyAcc={g('c2_value_v2_aux_copy_acc') * 100:.1f} "
                      f"weighted={g('c2_value_v2_aux_weighted_loss'):.3f}")
                _inner = getattr(getattr(loss_head, "model", None), "inner", None)
                # FIX A: V2 columns live in the dedicated color_evidence_proj (own lr group).
                _ch = getattr(_inner, "color_evidence_proj", None)
                if _ch is not None and getattr(args, "value_evidence_v2", False):
                    _cfg = getattr(_inner, "config", None)
                    # DRV-1 FIX: offset/width are SCHEMA-OWNED -- trm_fvr_v2.evidence_slice reads
                    # the same cfg flags for either model. The oracle hand-summed widths imported
                    # from the OLD modules (object_bank/trm_fvr_c2); constants agree today
                    # (G3-verified) but any v2 schema change would silently desync the probe.
                    _sl = None
                    try:
                        from models.recursive_reasoning.trm_fvr_v2 import evidence_slice as _ev_slice
                        _sl = _ev_slice(_cfg, "value_v2")
                    except Exception:
                        _sl = None
                    if _sl is not None:
                        _off, _vdim = int(_sl[0]), int(_sl[1])
                    else:
                        # fallback oracle: the old hand-summed arithmetic, verbatim
                        from models.recursive_reasoning.object_bank import REL_MAP_CHANNELS
                        from models.recursive_reasoning.trm_fvr_c2 import VALUE_EVIDENCE_V2_DIM
                        _off = 0
                        _off += REL_MAP_CHANNELS if bool(getattr(_cfg, "c2_relmap", False)) else 0
                        _off += 10 if bool(getattr(_cfg, "c2_task_palette_feature", False)) else 0
                        _off += (max(1, int(getattr(_cfg, "c2_rel_where_topk", 1)))
                                 if bool(getattr(_cfg, "c2_rel_where_hint", False)) else 0)
                        _off += 1 if bool(getattr(_cfg, "c2_pairdelta_intent_hint", False)) else 0
                        _off += 10 if bool(getattr(_cfg, "c2_transition_hint", False)) else 0
                        _vdim = VALUE_EVIDENCE_V2_DIM
                    _w = _ch.weight[:, _off:_off + _vdim].detach().float()
                    # color_evidence_proj.weight.grad must be non-zero here; else V2AUX cannot train the tail.
                    _cg = _ch.weight.grad
                    _gw = (_cg[:, _off:_off + _vdim].detach().float()
                           if _cg is not None else None)
                    _gn = float(_gw.norm()) if _gw is not None else float("nan")
                    _gm = float(_gw.abs().max()) if _gw is not None else float("nan")
                    print(f"  V2TAIL: off={_off} w_norm={float(_w.norm()):.4e} "
                          f"grad_norm={_gn:.4e} grad_max={_gm:.4e} "
                          f"logit_std={g('c2_value_v2_aux_logit_std'):.4e} "
                          f"logit_abs={g('c2_value_v2_aux_logit_abs_mean'):.4e}")
    if "c2_bind_aux_raw" in m:
        print(f"  BIND  : raw={g('c2_bind_aux_raw'):.3f} "
              f"chgCE={g('c2_bind_aux_changed_ce'):.3f} "
              f"copyCE={g('c2_bind_aux_copy_ce'):.3f} "
              f"chgAcc={g('c2_bind_aux_changed_acc') * 100:.1f} "
              f"copyAcc={g('c2_bind_aux_copy_acc') * 100:.1f} "
              f"support={g('c2_bind_aux_support_coverage') * 100:.1f} "
              f"weighted={g('c2_bind_aux_weighted_loss'):.3f}")
        if "c2_bind_aux_changed_support_coverage" in m:
            print(f"  BINDR : chg_support={g('c2_bind_aux_changed_support_coverage') * 100:.1f} "
                  f"copy_support={g('c2_bind_aux_copy_support_coverage') * 100:.1f} "
                  f"base_wrong_margin={g('c2_bind_aux_base_wrong_margin'):.3f} "
                  f"residual_norm={g('c2_bind_aux_residual_norm'):.6f} "
                  f"corrected={g('c2_bind_aux_corrected_changed_frac') * 100:.1f} "
                  f"copy_flips={g('c2_bind_aux_caused_copy_flip_frac') * 100:.1f}")
    if "c2_gate_where_f1" in m:
        print(f"  GATE  : f1={g('c2_gate_where_f1') * 100:.1f} "
              f"macro={g('c2_where_target_macro_f1') * 100:.1f} "
              f"micro={g('c2_where_target_micro_f1') * 100:.1f} "
              f"fpr={g('c2_gate_where_fpr') * 100:.1f} "
              f"macro_fpr={g('c2_where_target_macro_fpr') * 100:.1f} "
              f"chg={g('c2_gate_where_changed_mean'):.3f} "
              f"copy={g('c2_gate_where_copy_mean'):.3f} "
              f"chgMSE={g('c2_gate_where_changed_mse'):.3f} "
              f"copyMSE={g('c2_gate_where_copy_mse'):.3f} "
              f"shuffle_f1={g('c2_gate_shuffle_f1') * 100:.1f} "
              f"shuffle_macro={g('c2_where_shuffle_macro_f1') * 100:.1f} "
              f"f1_gap={g('c2_where_support_gap') * 100:+.1f} "
              f"support_hinge={g('c2_gate_support_contrast_raw'):.3f}")
    if "c2_where_support_fit_f1" in m:
        print(f"  WHERE : support_fit_f1={g('c2_where_support_fit_f1') * 100:.1f} "
              f"support_fit_fpr={g('c2_where_support_fit_fpr') * 100:.1f} "
              f"target_macro={g('c2_where_target_macro_f1') * 100:.1f} "
              f"target_micro={g('c2_where_target_micro_f1') * 100:.1f} "
              f"shuffle_macro={g('c2_where_shuffle_macro_f1') * 100:.1f} "
              f"support_gap={g('c2_where_support_gap') * 100:+.1f}")
    # LEVER growth: is structure_relmap_proj actually moving? Static norms across steps => the
    # lever is not learning; rising norms => it is training.
    _inner = getattr(getattr(loss_head, "model", None), "inner", None)
    _srp = getattr(_inner, "structure_relmap_proj", None)
    if _srp is not None:
        _wn = float(_srp.weight.detach().float().norm())
        _bn = float(_srp.bias.detach().float().norm()) if _srp.bias is not None else float("nan")
        _sop = getattr(_inner, "structure_outside_proj", None)
        _on = float(_sop.weight.detach().float().norm()) if _sop is not None else float("nan")
        _sep = getattr(_inner, "structure_eos_proj", None)
        _en = float(_sep.weight.detach().float().norm()) if _sep is not None else float("nan")
        _ssc = g("c2_outside_grid_extent_conf")
        _esc = g("c2_eos_grid_extent_conf")
        # extent/eos conf come from the MAIN forward's extras (the aux forward's extras are
        # discarded); statistically close to the LODO conf (one more demo) but label it honestly.
        print(f"  LEVER %: struct_w_norm={_wn:.4f} struct_b_norm={_bn:.4f} "
              f"outside_w_norm={_on:.4f} eos_w_norm={_en:.4f} "
              f"extent_conf(main)={_ssc:.2f} eos_conf(main)={_esc:.2f}")
    _ql = getattr(_inner, "quarantine_lin", None)
    if _ql is not None:
        # Is the quarantine head actually training? Norms must MOVE off warm-init (lin ~12.65);
        # lin_grad is THIS step's post-backward grad norm (panel runs after opt.step, before the
        # next zero_grad) -- ~0/nan = the LODO gradient is not reaching the head (wiring bug).
        _qmo = getattr(_inner, "quarantine_mlp_out", None)
        _qg = _ql.weight.grad
        _qgn = float(_qg.detach().float().norm()) if _qg is not None else float("nan")
        print(f"  QUAR  : lin_norm={float(_ql.weight.detach().float().norm()):.4f} "
              f"mlp_out_norm={float(_qmo.weight.detach().float().norm()) if _qmo is not None else float('nan'):.4f} "
              f"lin_grad={_qgn:.3e}")
    if "lodo_select_color_exact_pct" in m:
        print(f"  SELECT%: floor_color_exact={pct_s('lodo_floor_color_exact_pct')} "
              f"cand_color_exact={pct_s('lodo_color_exact_pct')} "
              f"select_color_exact={pct_s('lodo_select_color_exact_pct')} "
              f"select_cellsim={pct_s('lodo_select_color_cellsim_pct')} "
              f"safe={pct_s('lodo_select_ge_floor_pct')} "
              f"cand_chosen={pct_s('lodo_candidate_chosen_pct')} "
              f"floor_copy_flip={pct_s('lodo_floor_correct_copy_flip_pct')}")
    if "c2_task_palette_allowed_count" in m or "c2_palette_target_coverage_pct" in m:
        print(f"  PALETTE%: colors={g('c2_task_palette_allowed_count'):.1f}/10 "
              f"target_cover={g('c2_palette_target_coverage_pct'):.1f} "
              f"pred_allowed={g('c2_palette_pred_allowed_pct'):.1f} "
              f"pred_blocked={g('c2_palette_pred_disallowed_pct'):.1f}")
    if "c2_rel_where_hint_mean" in m or "c2_pairdelta_conditional_score" in m:
        print(f"  RELHINT: mean={g('c2_rel_where_hint_mean'):.3f} "
              f"chg={g('c2_rel_where_hint_on_changed'):.3f} "
              f"copy={g('c2_rel_where_hint_on_unchanged'):.3f} "
              f"gap={g('c2_rel_where_gap'):+.3f} "
              f"conf={g('c2_rel_where_confidence'):.3f} "
              f"f1={g('c2_rel_where_f1'):.3f} fpr={g('c2_rel_where_fpr'):.3f}")
    if "c2_awm_enclosed_on_changed" in m:
        print(f"  AWM   : encl_chg={g('c2_awm_enclosed_on_changed'):.3f} "
              f"encl_copy={g('c2_awm_enclosed_on_copy'):.3f} "
              f"seed_chg={g('c2_awm_seed_on_changed'):.3f} "
              f"seed_copy={g('c2_awm_seed_on_copy'):.3f}")
    if "c2_canonical_bind_support_coverage" in m:
        print(f"  CANON : coverage={g('c2_canonical_bind_support_coverage') * 100:.1f} "
              f"route={g('c2_canonical_bind_same_position_route'):.3f} "
              f"same_extent={g('c2_canonical_bind_same_extent'):.3f} "
              f"hist_change={g('c2_canonical_bind_histogram_change'):.3f} "
              f"movement={g('c2_canonical_bind_movement'):.3f} "
              f"collisions={g('c2_canonical_bind_key_collisions'):.0f}")
    if "c2_update_raw_norm_ratio" in m or "c2_target_relmap_raw_norm_ratio" in m:
        print(f"  FUSION: rel_raw={g('c2_target_relmap_raw_norm_ratio'):.3f} "
              f"rel_applied={g('c2_target_relmap_applied_norm_ratio'):.3f} "
              f"sup_in={g('c2_support_input_relmap_applied_norm_ratio'):.3f} "
              f"sup_out={g('c2_support_output_relmap_applied_norm_ratio'):.3f} "
              f"c2_raw={g('c2_update_raw_norm_ratio'):.3f} "
              f"c2_applied={g('c2_update_applied_norm_ratio'):.3f}")
        # Block 3: compression per lane + patience-gated warning. Behind --no-mextra (the
        # oracle-comparison escape hatch) like every V2-only panel line; diagnostic only --
        # the CONTINUE/ABORT verdict below is untouched, and rho/weight-decay stay as-is.
        if not args.no_mextra:
            _comp = {
                "rel": _fusion_compression(g("c2_target_relmap_raw_norm_ratio"),
                                           g("c2_target_relmap_applied_norm_ratio")),
                "c2": _fusion_compression(g("c2_update_raw_norm_ratio"),
                                          g("c2_update_applied_norm_ratio")),
            }
            print(f"  FUSION compression: rel={_comp['rel']:.2f} c2={_comp['c2']:.2f}")
            _streaks = getattr(S, "fusion_compression_streaks", None)
            if _streaks is None:
                _streaks = {}
                S.fusion_compression_streaks = _streaks
            for _lane in _fusion_compression_update(_streaks, _comp, args.fusion_compression_warn,
                                                    args.fusion_compression_patience):
                print(f"  [FUSION-WARN] {_lane} compression {_comp[_lane]:.2f} > "
                      f"{args.fusion_compression_warn:g} for {_streaks[_lane]} consecutive evals -- "
                      f"the rho clamp is discarding most of this lane's evidence residual "
                      f"(recalibrate the extractor, not rho; diagnostic only, run continues).")
    if "c2_rule_factor_same_extent" in m or "c2_object_pair_count" in m:
        print(f"  FACTOR: same_extent={g('c2_rule_factor_same_extent'):.3f} "
              f"recolour={g('c2_rule_factor_recolour'):.3f} "
              f"movement={g('c2_rule_factor_move'):.3f} "
              f"dir_cons={g('c2_rule_factor_direction_consistency'):.3f} "
              f"match_cov={g('c2_rule_factor_match_coverage'):.3f} "
              f"obj_tokens={g('c2_object_pair_count'):.1f} "
              f"obj_cov={g('c2_object_match_coverage'):.3f} "
              f"obj_precision={g('c2_object_match_precision_proxy'):.3f}")
    if "c2_rel_where_hint_mean" in m or "c2_pairdelta_conditional_score" in m:
        print(f"  PDINTENT: conditional={g('c2_pairdelta_conditional_score'):.3f} "
              f"global={g('c2_pairdelta_global_score'):.3f} "
              f"shape_pres={g('c2_pairdelta_shape_preserved'):.3f} "
              f"changed_rate={g('c2_pairdelta_changed_rate'):.3f}")
    # PDEV: the D-block evidence coverage line. The 500-step D8/D9 A/B ran BLIND -- none of the
    # pd/verified-frame/analogy coverage scalars were printed; one line fixes that.
    if any(k in m for k in ("c2_pd_color_consensus_mass", "c2_pd_struct_conf",
                            "c2_pd_bidi_invertibility", "c2_verified_frame_coverage",
                            "c2_value_ctx_bind_mass", "c2_algo_touch_mass", "c2_kin_conf")):
        print(f"  PDEV  : consensus={g('c2_pd_color_consensus_mass'):.3f} "
              f"minchg={g('c2_pd_color_min_change'):.3f} "
              f"pos={g('c2_pd_color_pos_prior'):.3f} "
              f"struct_conf={g('c2_pd_struct_conf'):.3f} "
              f"invert={g('c2_pd_bidi_invertibility'):.3f} "
              f"del={g('c2_pd_bidi_del_rate'):.3f} "
              f"vframe_cov={g('c2_verified_frame_coverage'):.3f} "
              f"analogy_cov={g('c2_analogy_coverage'):.3f} "
              f"touch={g('c2_algo_touch_mass'):.3f} "
              f"kin_conf={g('c2_kin_conf'):.3f} "
              f"kin_mover={g('c2_kin_mover_mass'):.3f} "
              f"in_conf={g('c2_pairdelta_input_conf'):.3f} "
              f"bind={g('c2_value_ctx_bind_mass'):.3f}")
    if "c2_rule_hyp_norm" in m:
        print(f"  RULEHYP: norm={g('c2_rule_hyp_norm'):.4e} "
              f"nonzero={g('c2_rule_hyp_nonzero_frac'):.3f}")
    # HINTS attribution line: ALL broadcast-injector norms side by side. Any injector norm >>0
    # here while MAIN-ON drops names the culprit without an ablation run.
    if "c2_frame_hint_norm" in m or "c2_pairdelta_input_norm" in m:
        print(f"  HINTS : frame_norm={g('c2_frame_hint_norm'):.4e} "
              f"frame_nonzero={g('c2_frame_hint_nonzero_frac'):.3f} "
              f"pairdelta_norm={g('c2_pairdelta_input_norm'):.4e} "
              f"spatial_norm={g('c2_pairdelta_spatial_feature_norm'):.3f} "
              f"spatial_valid={g('c2_pairdelta_spatial_valid_rate'):.3f}")
    if args.cross_demo_shape:
        print(f"  SHAPE %: exact={g('c2_shape_exact') * 100:.1f} h_acc={g('c2_shape_h_acc') * 100:.1f} "
              f"w_acc={g('c2_shape_w_acc') * 100:.1f} loss={g('c2_shape_loss'):.2f}")
    print(f"  BLOCK %: shape={g('lodo_block_shape_pct'):.0f} "
          f"color={g('lodo_block_color_pct'):.0f} pad={g('lodo_block_pad_pct'):.0f} "
          f"eos={g('lodo_block_eos_pct'):.0f}")
    print(f"  COUNTS : lodo_color={g('n_lodo_color_cells'):.0f} changed={g('n_lodo_changed_cells'):.0f} "
          f"unchanged={g('n_lodo_unchanged_cells'):.0f} pad={g('n_lodo_pad_cells'):.0f} "
          f"eos={g('n_lodo_eos_cells'):.0f} rows={g('n_lodo_rows'):.0f}")
    # Block 6 M-lines append BELOW the verbatim panel (plan decision 5); --no-mextra kills them.
    mextra_tail(S, step, loss_val, m, run_stats)
    if not args.no_abort_on_main_drop and step > 0:
        broken = (
            g("main_color_acc_pct") < 80.0
            or g("main_eos_acc_pct") < 80.0
            or g("main_shape_exact_pct") < 80.0
            or g("main_strict_exact_pct") < 75.0
        )
        if broken:
            print("[abort] MAIN degraded below safety threshold. Stop this run; lower --lodo-weight, "
                  "increase warmup, or keep calibration mode.")
            return False
    return True


# ======================================================================================
# §8b METRICS UPGRADE (Block 6): M-A FEED, M-B PRESS, M-C TREND, M-D RUN, M-E PERF,
# M-F START, M-G SCORE, M-H MOVERS + the panel CONTRACT check.
# Every line carries a bracketed tag ([START]/[FEED]/[PRESS]/[TREND]/[RUN ]/[PERF]/
# [SCORE]/[MOVERS]/[CONTRACT]) so equivalence gates can filter or kill them; --no-mextra
# disables ALL of them (the oracle-comparison escape hatch). Pure helpers (_trend_update,
# _movers_compute, _contract_missing, _feed_classify, _score_static_rows) are model-free
# and unit-gated in G6. Everything here is rank-0 by construction (prints are gated at
# init; the callers additionally guard the compute).
# ======================================================================================
def _mvals(m: dict) -> dict:
    """Metric dict -> count-normalised floats (the panel's g(k) for every key at once)."""
    c = float(m["count"].item()) if "count" in m and torch.is_tensor(m["count"]) else \
        float(m.get("count", 1.0))
    c = c if c > 0 else 1.0
    out = {}
    for k, v in m.items():
        if torch.is_tensor(v) and v.numel() == 1:
            out[k] = float(v.item()) / (1.0 if k == "count" else c)
        elif isinstance(v, (int, float)):
            out[k] = float(v) / (1.0 if k == "count" else c)
    return out


# --- M-C TREND -------------------------------------------------------------------------
def _trend_update(state: dict, vals: dict) -> list:
    """Track per-key history; warn when a key FALLS across 3 consecutive panels (4 points).
    Falling is 'bad' for every key the caller passes (the caller excludes loss from warns)."""
    hist = state.setdefault("hist", [])
    state.setdefault("base", dict(vals))
    hist.append(dict(vals))
    if len(hist) > 6:
        hist.pop(0)
    warns = []
    for key in vals:
        seq = [h[key] for h in hist if key in h]
        if len(seq) >= 4 and all(seq[i + 1] < seq[i] for i in range(len(seq) - 4, len(seq) - 1)):
            warns.append(f"{key} FALLINGx3 ({seq[-4]:.2f}->{seq[-1]:.2f})")
    return warns


# --- M-H MOVERS ------------------------------------------------------------------------
def _movers_compute(prev: dict, base: dict, cur: dict, k: int = 3):
    """Top-k |Δ since last panel| risers and fallers: (key, prev, cur, Δ-since-base, Δ-panel)."""
    ups, downs = [], []
    for key, v in cur.items():
        if key not in prev or key.startswith("n_") or key == "count":
            continue
        d = v - prev[key]
        if d != d or d == 0:
            continue
        (ups if d > 0 else downs).append((key, prev[key], v, v - base.get(key, v), d))
    ups.sort(key=lambda t: -abs(t[4]))
    downs.sort(key=lambda t: -abs(t[4]))
    return ups[:k], downs[:k]


# --- CONTRACT check (PDEV's origin story, made structural) -------------------------------
_CONTRACT_TABLE = (
    ("floor_candidate_split", "floor_candidate_split", ("lodo_select_color_exact_pct",)),
    ("value_evidence_v2", "value_evidence_v2", ("c2_value_v2_support_coverage",)),
    ("verified_frame", "verified_frame_evidence", ("c2_verified_frame_coverage",)),
    ("analogy", "analogy_evidence", ("c2_analogy_coverage",)),
    ("kinematic", "kinematic_evidence", ("c2_kin_conf",)),
    ("algo_touch", "algo_where_touch", ("c2_algo_touch_mass",)),
    ("pd_color", "pairdelta_color_evidence", ("c2_pd_color_consensus_mass",)),
    ("pd_struct", "pairdelta_structure_evidence", ("c2_pd_struct_conf",)),
    ("pd_bidi", "pairdelta_bidi_evidence", ("c2_pd_bidi_invertibility",)),
    ("value_ctx_bind", "value_ctx_bind", ("c2_value_ctx_bind_mass",)),
    ("bind_aux", "bind_aux_weight", ("c2_bind_aux_changed_acc", "c2_bind_aux_support_coverage")),
    ("bind_residual", "bind_residual_aware", ("c2_bind_aux_changed_support_coverage",
                                                  "c2_bind_aux_caused_copy_flip_frac")),
    ("token_gate_where", "token_gate_where", ("c2_gate_where_f1", "c2_gate_shuffle_f1")),
    ("positive_where_gate", "positive_where_gate", ("c2_where_target_macro_f1", "c2_where_support_gap")),
    ("canonical_value", "canonical_value_binder", ("c2_canonical_bind_support_coverage",
                                                      "c2_canonical_bind_key_collisions")),
    ("rule_factors", "rule_factor_hint", ("c2_rule_factor_same_extent", "c2_rule_factor_move")),
    ("object_pair_tokens", "object_pair_tokens", ("c2_object_pair_count", "c2_object_match_coverage")),
    ("bounded_fusion", "bounded_evidence_fusion", ("c2_update_raw_norm_ratio",
                                                     "c2_update_applied_norm_ratio")),
    ("transition", "transition_hint", ("c2_transition_hint_coverage",)),
    ("task_palette", "task_palette", ("c2_task_palette_allowed_count",)),
    ("rel_where", "rel_where_hint", ("c2_rel_where_hint_mean",)),
    ("pairdelta_intent", "pairdelta_intent_hint", ("c2_pairdelta_conditional_score",)),
    ("cross_demo_shape", "cross_demo_shape", ("c2_shape_exact",)),
    ("frame_hint", "frame_hint", ("c2_frame_hint_norm",)),
    ("rule_hyp_hint", "rule_hypothesis_hint", ("c2_rule_hyp_norm",)),
    ("pairdelta_input", "pairdelta_input", ("c2_pairdelta_input_norm",)),
    ("pairdelta_spatial", "pairdelta_spatial", ("c2_pairdelta_spatial_feature_norm",
                                                   "c2_pairdelta_spatial_valid_rate")),
)


def _contract_missing(args, m: dict) -> list:
    """[(flag_label, missing_metric_key)] for every ON flag whose panel keys are absent."""
    out = []
    for label, attr, keys in _CONTRACT_TABLE:
        if getattr(args, attr, False):
            for k in keys:
                if k not in m:
                    out.append((label, k))
    return out


# --- M-A FEED ---------------------------------------------------------------------------
_FEED_TABLE = (
    ("vframe", "verified_frame_evidence", "c2_verified_frame_coverage"),
    ("analogy", "analogy_evidence", "c2_analogy_coverage"),
    ("kin", "kinematic_evidence", "c2_kin_conf"),
    ("touch", "algo_where_touch", "c2_algo_touch_mass"),
    ("pd_color", "pairdelta_color_evidence", "c2_pd_color_consensus_mass"),
    ("pd_struct", "pairdelta_structure_evidence", "c2_pd_struct_conf"),
    ("pd_bidi", "pairdelta_bidi_evidence", "c2_pd_bidi_invertibility"),
    ("bind", "value_ctx_bind", "c2_value_ctx_bind_mass"),
    ("bind_aux", "bind_aux_weight", "c2_bind_aux_support_coverage"),
    ("bind_residual", "bind_residual_aware", "c2_bind_aux_changed_support_coverage"),
    ("gate_where", "token_gate_where", "c2_gate_where_changed_mean"),
    ("value_v2", "value_evidence_v2", "c2_value_v2_support_coverage"),
    ("transition", "transition_hint", "c2_transition_hint_coverage"),
    ("frame_hint", "frame_hint", "c2_frame_hint_nonzero_frac"),
    ("rule_hyp", "rule_hypothesis_hint", "c2_rule_hyp_nonzero_frac"),
    ("pd_input", "pairdelta_input", "c2_pairdelta_input_norm"),
)


def _feed_classify(args, m: dict) -> list:
    """Per ACTIVE block: LIVE (metric > 0) / ZERO (fires but empty -- starved builder) /
    NO-METRIC (flag on, nothing measured -- wiring gap). The direct answer to 'is each
    component getting its input', per block, from what the eval already computed."""
    vals = _mvals(m)
    out = []
    for name, attr, key in _FEED_TABLE:
        if not getattr(args, attr, False):
            continue
        if key not in m:
            out.append((name, ("NO-METRIC", float("nan"))))
        else:
            v = vals.get(key, 0.0)
            out.append((name, ("LIVE", v) if v > 0 else ("ZERO", v)))
    return out


# --- M-B PRESS ---------------------------------------------------------------------------
def press_report(S) -> None:
    """Per-evidence-block grad norm of color_evidence_proj (schema slices via
    trm_fvr_v2.evidence_layout -- the DRV-1 fix generalised from value_v2 to EVERY block)
    + per-lr-group grad norms. Nonzero FEED + zero PRESS = unpressured evidence (the D8/D9
    500-step lesson as a per-panel readout). Runs post-opt.step, pre-zero_grad (grads live)."""
    inner = getattr(getattr(S.loss_head, "model", None), "inner", None)
    proj = getattr(inner, "color_evidence_proj", None)
    parts = []
    if proj is not None and proj.weight.grad is not None:
        try:
            from models.recursive_reasoning.trm_fvr_v2 import evidence_layout as _lay
            for name, w, off in _lay(inner.config)[0]:
                gn = float(proj.weight.grad[:, off:off + w].detach().float().norm())
                parts.append(f"{name}={gn:.4e}")
        except Exception:
            parts.append(f"total={float(proj.weight.grad.detach().float().norm()):.4e}")
    gparts = []
    for nm, grp in zip(getattr(S.opt, "_trv2_group_names", []), S.opt.param_groups):
        tot = 0.0
        for p in grp["params"]:
            if p.grad is not None:
                tot += float(p.grad.detach().float().norm()) ** 2
        gparts.append(f"{nm}={tot ** 0.5:.4e}")
    print(f"[PRESS] ev: {' '.join(parts) if parts else 'n/a'} | groups: {' '.join(gparts)}")


# --- M-D RUN + M-E PERF -------------------------------------------------------------------
def run_perf_report(S, step: int, run_stats: dict | None) -> None:
    import time as _t
    rs = run_stats or {}
    names = getattr(S.opt, "_trv2_group_names", [])
    lrs = " ".join(f"{nm}={grp['lr']:.2e}" for nm, grp in zip(names, S.opt.param_groups))
    mw = []
    for nm, grp in zip(names, S.opt.param_groups):
        mx = 0.0
        for p in grp["params"]:
            mx = max(mx, float(p.detach().abs().max()))
        mw.append(f"{nm}={mx:.3g}")
    print(f"[RUN ] nan_skips={rs.get('nan', 0)} grad_norm={rs.get('grad_norm', float('nan')):.4f} "
          f"lr: {lrs} | max|w|: {' '.join(mw)}")
    now = _t.time()
    pf = getattr(S, "_perf", None)
    if pf is not None:
        ds, dt = step - pf["step"], now - pf["t"]
        if ds > 0 and dt > 0:
            sps = ds / dt
            eta = (S.args.steps - step) / sps if sps > 0 else float("nan")
            mem = ""
            if torch.cuda.is_available():
                dev = torch.cuda.current_device()
                mem = (f" cuda={torch.cuda.memory_allocated(dev) / 1e9:.2f}"
                       f"/{torch.cuda.get_device_properties(dev).total_memory / 1e9:.1f}GB")
            print(f"[PERF] {sps:.3f} steps/s eta={eta / 60:.1f}m{mem}")
    S._perf = {"t": now, "step": step}


# --- M-G SCORE ----------------------------------------------------------------------------
def _score_static_rows(S, batches=None, collect: dict | None = None) -> list:
    """Component scorecard over the frozen eval: the driver calls each ACTIVE component's OWN
    builder (the same §10b/SS5 functions the model wires) and scores the EVIDENCE ITSELF
    against the held-out truth. Rows: (name, cov%, acc@changed, acc@copy, Δfloor, n_cells);
    the kin row reads (name, mover-recall, mover-precision, nan, nan, n_changed).
    Evidence is deterministic of the (frozen) batch => constant for the run: computed ONCE.
    `collect[name] = [(covered [B,L] bool, pred_tok [B,L] long), ...]` feeds SCORE-dynamic."""
    args = S.args
    batches = batches if batches is not None else S.eval_batches
    active = [n for n, attr in (("vframe", "verified_frame_evidence"),
                                ("analogy", "analogy_evidence"),
                                ("kin", "kinematic_evidence"),
                                ("pd_color", "pairdelta_color_evidence"))
              if getattr(args, attr, False)]
    if not active or not batches:
        return []
    from models.recursive_reasoning import core_prior as CP
    agg = {n: dict(cov_n=0, col_n=0, chg_ok=0, chg_n=0, cp_ok=0, cp_n=0, ok=0, fl_ok=0, all_n=0)
           for n in active}
    rows_used = 0
    cap = int(getattr(args, "score_rows", 48) or 0)
    with torch.no_grad():
        for eb in batches:
            ci = eb["context_inputs"].detach().to("cpu", torch.long)
            co = eb.get("context_labels", eb.get("context_outputs"))
            co = co.detach().to("cpu", torch.long)
            cmk = eb.get("context_mask")
            cmk = (cmk.detach().cpu().bool() if cmk is not None
                   else torch.ones(ci.shape[:2], dtype=torch.bool))
            ti = eb["inputs"].detach().to("cpu", torch.long)
            tl = eb["labels"].detach().to("cpu", torch.long)
            B, L = ti.shape
            side = int(L ** 0.5)
            pd_feats = None
            if "pd_color" in active:
                from models.recursive_reasoning.pair_delta_v2 import pd_color_evidence
                pd_feats, _ = pd_color_evidence(ci, co, cmk, ti)
            bcov = {n: torch.zeros(B, L, dtype=torch.bool) for n in active if n != "kin"}
            btok = {n: torch.zeros(B, L, dtype=torch.long) for n in active if n != "kin"}
            for b in range(B):
                if cap and rows_used >= cap:
                    break
                rows_used += 1
                lab, inp = tl[b], ti[b]
                colour = lab >= 2
                if int(colour.sum()) == 0:
                    continue
                changed = colour & (lab != inp)
                copy = colour & (lab == inp)
                keep = cmk[b].nonzero(as_tuple=True)[0]
                preds = {}
                if "vframe" in active and keep.numel():
                    grid, conf, _p = CP.evidence_verified_frame_grid(ci[b][keep], co[b][keep], ti[b], side)
                    preds["vframe"] = grid if conf > 0 else torch.zeros_like(grid)
                if "analogy" in active and keep.numel():
                    grid, _cf, _p = CP.evidence_analogy(ci[b][keep], co[b][keep], ti[b], side)
                    preds["analogy"] = grid
                if "pd_color" in active:
                    preds["pd_color"] = pd_feats[b, :, :10]
                for name, pr in preds.items():
                    a = agg[name]
                    covered = (pr.sum(-1) > 0) & colour
                    a["col_n"] += int(colour.sum())
                    a["cov_n"] += int(covered.sum())
                    if bool(covered.any()):
                        hit = (pr.argmax(-1) + 2) == lab
                        a["chg_ok"] += int((hit & covered & changed).sum())
                        a["chg_n"] += int((covered & changed).sum())
                        a["cp_ok"] += int((hit & covered & copy).sum())
                        a["cp_n"] += int((covered & copy).sum())
                        a["ok"] += int((hit & covered).sum())
                        a["fl_ok"] += int(((inp == lab) & covered).sum())
                        a["all_n"] += int(covered.sum())
                        if name in bcov:
                            bcov[name][b] = covered
                            btok[name][b] = torch.where(covered, pr.argmax(-1) + 2,
                                                        torch.zeros_like(lab))
                if "kin" in active and keep.numel():
                    feats, _cf, _p = CP.evidence_kinematics(ci[b][keep], co[b][keep], ti[b], side)
                    mask = feats[:, 0] > 0
                    a = agg["kin"]
                    a["col_n"] += int(changed.sum())
                    a["cov_n"] += int((mask & changed).sum())        # recall numerator
                    a["chg_ok"] += int((mask & changed).sum())       # precision numerator
                    a["chg_n"] += int(mask.sum())                    # precision denominator
            if collect is not None:
                for n in bcov:
                    collect.setdefault(n, []).append((bcov[n], btok[n]))
            if cap and rows_used >= cap:
                break
    pct = lambda n, d: (100.0 * n / d) if d else float("nan")
    out = []
    for name in active:
        a = agg[name]
        if name == "kin":
            out.append((name, pct(a["cov_n"], a["col_n"]), pct(a["chg_ok"], a["chg_n"]),
                        float("nan"), float("nan"), a["col_n"]))
        else:
            dfl = (pct(a["ok"], a["all_n"]) - pct(a["fl_ok"], a["all_n"])) if a["all_n"] else float("nan")
            out.append((name, pct(a["cov_n"], a["col_n"]), pct(a["chg_ok"], a["chg_n"]),
                        pct(a["cp_ok"], a["cp_n"]), dfl, a["col_n"]))
    return out


def score_static(S) -> None:
    """M-G static half: printed ONCE (evidence is deterministic of the frozen batches).
    The chgAcc=21.7 measurement that closed the value-v2 family, generalised to every
    D-block: 'is component X's recommendation RIGHT where it fires', model-independent."""
    if getattr(S.args, "no_mextra", False) or not getattr(S.dist, "is_main", True):
        return
    if not int(getattr(S.args, "score_rows", 48) or 0):
        return
    import time as _t
    t0 = _t.time()
    cache: dict = {}
    rows = _score_static_rows(S, collect=cache)
    if not rows:
        return
    S._score_cache = cache
    print(f"[SCORE] static component scorecard (frozen eval, once, row-cap "
          f"{S.args.score_rows}, {_t.time() - t0:.1f}s) -- evidence vs held-out truth")
    print(f"[SCORE] {'component':<10} {'cov%':>6} {'acc@chg':>8} {'acc@copy':>9} {'Dfloor':>7} {'n':>8}")
    for name, cov, ac, ap, df, n in rows:
        note = "  (recall/precision of mover mask vs changed cells)" if name == "kin" else ""
        print(f"[SCORE] {name:<10} {cov:6.1f} {ac:8.1f} {ap:9.1f} {df:7.1f} {n:8d}{note}")
    if S.family_eval_sets:
        for fam, bset in S.family_eval_sets.items():
            for name, cov, ac, ap, df, n in _score_static_rows(S, batches=bset):
                print(f"[SCORE] {fam[:12]:>12}/{name:<9} cov={cov:5.1f} acc@chg={ac:5.1f} "
                      f"acc@copy={ap:5.1f} n={n}")


def score_dynamic(S, step: int) -> None:
    """M-G dynamic half: per panel, the MODEL's accuracy on each component's covered vs
    uncovered cells + model-vs-component agreement (against the CACHED step-0 masks; only
    model predictions are recomputed). Rising agreement with a high-intrinsic-acc component
    = the model is learning to exploit it; with a low one = learning a bad habit."""
    cache = getattr(S, "_score_cache", None)
    if not cache or not S.eval_batches:
        return
    inner = S.loss_head.model.inner
    was = S.loss_head.training
    S.loss_head.eval()
    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    stats: dict = {}
    with torch.no_grad():
        for bi, eb in enumerate(S.eval_batches):
            ie, _, rm, _ = inner._input_embeddings(eb)
            z_H, _ = inner._run_recurrence(inner.fresh_carry(eb["inputs"].shape[0]), ie, seq_info)
            lg, _ = inner._output_logits(z_H, eb, rel_maps=rm)
            pred = lg.argmax(-1).detach().cpu()
            lab = eb["labels"].detach().cpu().long()
            colour = lab >= 2
            okm = pred == lab
            for name, per_batch in cache.items():
                if bi >= len(per_batch):
                    continue
                cov, ctok = per_batch[bi]
                st = stats.setdefault(name, dict(c_ok=0, c_n=0, u_ok=0, u_n=0, a_ok=0))
                st["c_ok"] += int((okm & cov).sum())
                st["c_n"] += int(cov.sum())
                unc = colour & ~cov
                st["u_ok"] += int((okm & unc).sum())
                st["u_n"] += int(unc.sum())
                st["a_ok"] += int(((pred == ctok) & cov).sum())
    if was:
        S.loss_head.train()
    pct = lambda n, d: (100.0 * n / d) if d else float("nan")
    base = getattr(S, "_score_dyn0", None)
    if base is None:
        base = S._score_dyn0 = {}
    parts = []
    for name, st in stats.items():
        cur = (pct(st["c_ok"], st["c_n"]), pct(st["u_ok"], st["u_n"]), pct(st["a_ok"], st["c_n"]))
        b = base.setdefault(name, cur)
        parts.append(f"{name}: cov={cur[0]:.1f}({cur[0] - b[0]:+.1f}) "
                     f"uncov={cur[1]:.1f}({cur[1] - b[1]:+.1f}) agree={cur[2]:.1f}({cur[2] - b[2]:+.1f})")
    if parts:
        print("[SCORE] dyn " + " | ".join(parts))


# --- M-F START banner ----------------------------------------------------------------------
def start_banner(S) -> None:
    """The first-minute gate as printed contract: scope, trainable set, world, batch split,
    active evidence layout, lr groups. Extends the memory-mandated 'trainable tensors in the
    HUNDREDS for unified' check into something a log always carries."""
    if getattr(S.args, "no_mextra", False) or not getattr(S.dist, "is_main", True):
        return
    n_p = sum(p.numel() for p in S.params)
    lay = "none-active"
    try:
        from models.recursive_reasoning.trm_fvr_v2 import evidence_layout as _lay
        rows = _lay(S.loss_head.model.inner.config)[0]
        if rows:
            lay = " ".join(f"{n}[{o}:{o + w})" for n, w, o in rows)
    except Exception:
        lay = "<no v2 schema (legacy model)>"
    gd = " ".join(f"{nm}={len(g['params'])}@{g['lr']:g}"
                  for nm, g in zip(getattr(S.opt, "_trv2_group_names", []), S.opt.param_groups))
    print(f"[START] scope={S.scope_label} trainable={len(S.params)}t/{n_p:,}p "
          f"world={S.dist.world} batch={S.args.batch}(per-rank "
          f"{S.args.batch // max(1, S.dist.world)}) steps={S.args.steps}")
    print(f"[START] groups: {gd}")
    print(f"[START] evidence: {lay}")


# --- panel tail orchestrator -----------------------------------------------------------------
def mextra_tail(S, step: int, loss_val: float, m: dict, run_stats: dict | None) -> None:
    if getattr(S.args, "no_mextra", False):
        return
    vals = _mvals(m)
    # CONTRACT: a flag that is ON with its panel keys absent = a line that silently cannot
    # print (the V3-1/PDEV blind-run class). Report each miss once.
    seen = getattr(S, "_contract_seen", None)
    if seen is None:
        seen = S._contract_seen = set()
    for lbl, key in _contract_missing(S.args, m):
        if (lbl, key) not in seen:
            seen.add((lbl, key))
            print(f"[CONTRACT] MISSING: flag '{lbl}' is ON but metric '{key}' is absent from "
                  f"the eval dict -- its panel line cannot print (V3-1/PDEV class).")
    # FEED: print at step 0 and whenever a block's classification flips.
    cls = _feed_classify(S.args, m)
    fs = getattr(S, "_feed_state", None)
    if fs is None:
        fs = S._feed_state = {}
    if step == 0 or any(fs.get(n) != c[0] for n, c in cls):
        if cls:
            print("[FEED] " + " ".join(f"{n}={c[0]}({c[1]:.3f})" for n, c in cls))
        for n, c in cls:
            fs[n] = c[0]
    press_report(S)
    # TREND over the abort-guard keys + loss (loss excluded from falling-warns: falling loss
    # is the goal).
    tvals = {k: vals[k] for k in ("main_color_acc_pct", "main_eos_acc_pct",
                                  "main_shape_exact_pct", "main_strict_exact_pct") if k in vals}
    tvals["loss"] = float(loss_val)
    st = getattr(S, "_trend", None)
    if st is None:
        st = S._trend = {}
    warns = [w for w in _trend_update(st, tvals) if not w.startswith("loss ")]
    base = st.get("base", {})
    print("[TREND] " + " ".join(
        f"{k.replace('main_', '').replace('_acc_pct', '').replace('_exact_pct', '')}"
        f"={v:.1f}({v - base.get(k, v):+.1f})" for k, v in tvals.items())
        + ((" | WARN: " + "; ".join(warns)) if warns else ""))
    run_perf_report(S, step, run_stats)
    score_dynamic(S, step)
    mv = getattr(S, "_movers", None)
    if mv is None:
        mv = S._movers = {}
    if "base" not in mv:
        mv["base"] = dict(vals)
        mv["prev"] = dict(vals)
    else:
        ups, downs = _movers_compute(mv["prev"], mv["base"], vals, k=3)
        fmt = lambda t: f"{t[0]} {t[1]:.2f}->{t[2]:.2f}(d0{t[3]:+.2f})"
        print("[MOVERS] UP: " + ("; ".join(fmt(t) for t in ups) if ups else "none")
              + " | DOWN: " + ("; ".join(fmt(t) for t in downs) if downs else "none"))
        mv["prev"] = dict(vals)


# ======================================================================================
# §9 TRAIN LOOP + main() — verbatim oracle 1597-1710 over the run state S.
# ======================================================================================
def _build_source_manifest(root: Path, relative_paths=PROVENANCE_SOURCE_FILES) -> dict:
    """Content-address the exact source files that define a run."""
    import hashlib
    import json
    import subprocess

    root = Path(root).resolve()
    entries = []
    for item in sorted({str(p).replace("\\", "/") for p in relative_paths}):
        candidate = Path(item)
        path = candidate if candidate.is_absolute() else root / candidate
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"provenance source file is missing: {path}")
        data = path.read_bytes()
        try:
            display = path.relative_to(root).as_posix()
        except ValueError:
            display = str(path)
        entries.append({
            "path": display,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        })
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest = {
        "schema": 1,
        "fingerprint": hashlib.sha256(canonical).hexdigest(),
        "files": entries,
        "git_head": None,
        "git_status_porcelain": None,
    }
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, encoding="utf-8").stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
        manifest["git_head"] = head
        manifest["git_status_porcelain"] = status
    except (OSError, subprocess.SubprocessError):
        pass
    return manifest


def _build_lodo_contract_manifest(eval_batches: list, eval_seed: int) -> dict:
    """Serialize the immutable held-out fold identities used by every fixed evaluation."""
    import hashlib
    import json

    batches = []
    for index, batch in enumerate(eval_batches):
        case_hash = hashlib.sha256()
        for key in ("inputs", "context_inputs", "context_outputs", "context_mask"):
            value = batch.get(key)
            if torch.is_tensor(value):
                case_hash.update(key.encode("utf-8"))
                case_hash.update(value.detach().to("cpu").contiguous().numpy().tobytes())

        def _tolist(key, default):
            value = batch.get(key)
            return value.detach().to("cpu").reshape(-1).tolist() if torch.is_tensor(value) else default

        batches.append({
            "batch_index": index,
            "seed": int(eval_seed) + index,
            "case_sha256": case_hash.hexdigest(),
            "puzzle_identifiers": [int(x) for x in _tolist("puzzle_identifiers", [])],
            "holdout_idx": [int(x) for x in _tolist("_lodo_holdout_idx", [])],
            "aux_valid": [bool(x) for x in _tolist("_lodo_aux_valid", [])],
        })
    body = {"schema": 1, "eval_seed": int(eval_seed), "batches": batches}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return body


def _run_artifact_root(S) -> Path:
    if S.args.save_checkpoint_dir:
        return Path(S.args.save_checkpoint_dir)
    if getattr(S.args, "task_metrics_out", ""):
        return Path(S.args.task_metrics_out).resolve().parent
    return Path(S.raw.get("checkpoint_path", "reports/test_run_v2_probe"))


def _write_run_manifests(S) -> None:
    """Write source and fold contracts before training so failed runs remain attributable."""
    import json

    if not S.dist.is_main:
        return
    source_paths = list(PROVENANCE_SOURCE_FILES)
    configured = Path(S.args.config)
    if configured.is_absolute():
        source_paths.append(str(configured))
    elif configured.as_posix() not in source_paths:
        source_paths.append(configured.as_posix())
    source = _build_source_manifest(ROOT, source_paths)
    lodo = _build_lodo_contract_manifest(S.eval_batches, S.args.eval_seed)
    out_dir = _run_artifact_root(S)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("source_manifest.json", source), ("lodo_contract.json", lodo)):
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    S.source_manifest = source
    S.lodo_contract_manifest = lodo
    print(f"[provenance] source={source['fingerprint'][:12]} "
          f"lodo={lodo['fingerprint'][:12]} -> {out_dir}")


def _write_failure_artifact(S, batch: dict, step: int, stage: str, detail: str = "") -> Path | None:
    """Persist the exact offending rank batch and RNG state without aborting the guard."""
    import json
    import random
    import re

    import numpy as np

    try:
        failure_dir = _run_artifact_root(S) / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage))
        path = failure_dir / f"step_{int(step):06d}_rank_{int(S.dist.rank):03d}_{safe_stage}.pt"
        batch_cpu = {
            key: (value.detach().to("cpu") if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        payload = {
            "schema": 1,
            "step": int(step),
            "rank": int(S.dist.rank),
            "stage": str(stage),
            "detail": str(detail),
            "batch": batch_cpu,
            "resolved_config": S.raw,
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        pid_value = batch_cpu.get("puzzle_identifiers")
        summary = {
            "step": int(step), "rank": int(S.dist.rank), "stage": str(stage),
            "detail": str(detail), "artifact": str(path),
            "puzzle_identifiers": pid_value.reshape(-1).tolist() if torch.is_tensor(pid_value) else [],
        }
        path.with_suffix(".json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[failure-artifact] {stage} rank={S.dist.rank} step={step} -> {path}")
        return path
    except Exception as exc:  # diagnostics must never defeat the numerical safety guard
        print(f"[failure-artifact WARNING] could not persist {stage}: {type(exc).__name__}: {exc}")
        return None


def _has_nonfinite_grad(params) -> bool:
    for parameter in params:
        grad = parameter.grad
        if grad is None:
            continue
        if grad.is_sparse:
            grad = grad.coalesce().values()
        elif grad.layout != torch.strided:
            grad = grad.values()
        if not bool(torch.isfinite(grad).all()):
            return True
    return False


def save_checkpoint(S, completed_step: int, final: bool = False) -> None:
    """Rank-0 checkpoint writer keyed by completed optimizer updates, without duplicate saves."""
    if not S.args.save_checkpoint_dir or not S.dist.is_main:
        return
    if completed_step in S.saved_checkpoint_steps:
        return
    out_dir = Path(S.args.save_checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"step_{completed_step}"
    torch.save(S.loss_head.state_dict(), ckpt_path)
    config_path = out_dir / "all_config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(S.raw, f, sort_keys=False)
    S.saved_checkpoint_steps.add(completed_step)
    tag = "final " if final else "milestone "
    print(f"[checkpoint] saved {tag}model -> {ckpt_path}")
    print(f"[checkpoint] saved resolved config -> {config_path}")


def _gradient_probe_core(step_fn, params, steps: int, target_ratio: float) -> dict:
    """P3A Block 4 core: per-step gradient norms of L_where and the RAW support contrast on the
    SAME shared parameter set, then lambda_support = target_ratio * G_where / (G_support + 1e-12).

    torch.autograd.grad only -- .grad fields are never populated and no optimizer exists here, so
    parameters cannot move (the caller and the unit test fingerprint them anyway). The achieved
    ratio is re-evaluated PER STEP with the single recommended lambda: if the two gradient scales
    are unstable across batches, the mean achieved ratio drifts out of [0.25, 1.0] and the probe
    fails instead of blessing a constant that only worked on its own averaging window.

    step_fn(i) must return (l_where, l_support) as ATTACHED scalar tensors."""
    import math

    g_where, g_support = [], []
    for i in range(int(steps)):
        l_where, l_support = step_fn(i)

        def _norm(loss):
            # The support contrast's empty-set contract returns a DETACHED zero (no matched rows
            # in this batch). That is a true zero gradient, not an error.
            if not loss.requires_grad:
                return 0.0
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            total = torch.zeros((), dtype=torch.float64)
            for g in grads:
                if g is not None:
                    total = total + g.detach().double().pow(2).sum().cpu()
            return float(total.sqrt())

        g_where.append(_norm(l_where))
        g_support.append(_norm(l_support))
        del l_where, l_support   # release the graph before the next forward
    mean_w = sum(g_where) / max(1, len(g_where))
    mean_s = sum(g_support) / max(1, len(g_support))
    recommended = float(target_ratio) * mean_w / (mean_s + 1e-12)
    achieved_per_step = [recommended * s / (w + 1e-12) for w, s in zip(g_where, g_support)]
    achieved = sum(achieved_per_step) / max(1, len(achieved_per_step))
    ok = (all(math.isfinite(a) for a in achieved_per_step)
          and math.isfinite(recommended)
          and 0.25 <= achieved <= 1.0)
    return {
        "probe_steps": int(steps),
        "g_where_per_step": g_where,
        "g_support_per_step": g_support,
        "g_where_mean": mean_w,
        "g_support_mean": mean_s,
        "target_ratio": float(target_ratio),
        "recommended_lambda_support": recommended,
        "achieved_weighted_ratio_per_step": achieved_per_step,
        "achieved_weighted_ratio": achieved,
        "ratio_window": [0.25, 1.0],
        "pass": ok,
    }


def where_gradient_probe(S) -> None:
    """P3A Block 4 runner: forward --where-gradient-probe-steps training batches through the loss
    head with the probe stash armed, feed the attached L_where / L_support_raw pair to
    _gradient_probe_core over the v3-where-selector parameter surface, verify no parameter moved,
    print the report, and write the calibration JSON."""
    import json

    args = S.args
    device = S.device
    probe_params, probe_names, probe_bad = select_where_selector_params(
        (n, p) for n, p in S.loss_head.named_parameters() if p.requires_grad)
    if probe_bad:
        raise SystemExit(f"[where-probe GATE] forbidden tensors in the probe surface: {probe_bad}")
    if not probe_params:
        raise SystemExit("[where-probe GATE] no v3-where-selector parameters exist -- build the C2 "
                         "WHERE stack first (--token-gate-where --positive-where-gate "
                         "--ordered-evidence-flow --rel-where-hint ...).")
    fingerprint = [p.detach().clone() for p in probe_params]

    batches = []
    for _set, cb, _g in S.loader:
        if "context_inputs" not in cb or cb["context_inputs"].shape[1] < 2:
            continue
        batches.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()})
        if len(batches) >= int(args.where_gradient_probe_steps):
            break
    if len(batches) < int(args.where_gradient_probe_steps):
        raise SystemExit(f"[where-probe GATE] only {len(batches)} usable batches; "
                         f"--where-gradient-probe-steps {args.where_gradient_probe_steps} needs more data.")

    S.loss_head.train()   # measure the arms exactly as training builds them (LODO + shuffle)

    def step_fn(i):
        S.loss_head._where_grad_probe_stash = {}
        try:
            with torch.device(device):
                carry = S.loss_head.initial_carry(batches[i])
            with torch.enable_grad():
                _c, _loss, _m, _, _ = S.loss_head(carry=carry, batch=batches[i], return_keys=[])
            stash = S.loss_head._where_grad_probe_stash
        finally:
            S.loss_head._where_grad_probe_stash = None
        missing = [k for k in ("l_where_raw", "l_support_raw") if k not in stash]
        if missing:
            raise SystemExit(
                f"[where-probe GATE] forward produced no {missing} -- the probe needs the WHERE "
                "gate (--token-gate-where --positive-where-gate) and the shuffle arm (forced by "
                "--token-gate-where) plus a LODO build (--lodo-weight > 0).")
        return stash["l_where_raw"], stash["l_support_raw"]

    result = _gradient_probe_core(step_fn, probe_params,
                                  int(args.where_gradient_probe_steps),
                                  float(args.where_gradient_target_ratio))
    for p, before in zip(probe_params, fingerprint):
        if not torch.equal(p.detach(), before):
            raise AssertionError("where-gradient probe mutated a parameter -- this must never happen")
    result["parameters_unchanged"] = True
    result["n_params"] = len(probe_params)
    result["scope"] = "v3-where-selector"

    out_path = args.where_gradient_probe_out or (
        os.path.splitext(args.task_metrics_out)[0] + "_where_probe.json"
        if args.task_metrics_out else "where_gradient_probe.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[P3A GRADIENT PROBE]  (no optimizer; {len(probe_params)} selector tensors, "
          f"fingerprint-verified unchanged)")
    print(f"  G_where={result['g_where_mean']:.6g}  G_support={result['g_support_mean']:.6g}  "
          f"per-step W={['%.4g' % v for v in result['g_where_per_step']]} "
          f"S={['%.4g' % v for v in result['g_support_per_step']]}")
    print(f"  recommended_lambda_support={result['recommended_lambda_support']:.6g}  "
          f"(target_ratio={result['target_ratio']:g})")
    print(f"  achieved_weighted_ratio={result['achieved_weighted_ratio']:.4f}  "
          f"window=[0.25,1.00]  -> {'PASS' if result['pass'] else 'FAIL'}")
    print(f"  report -> {out_path}")


def train(S) -> None:
    args = S.args
    loss_head = S.loss_head
    opt = S.opt
    params = S.params
    device = S.device

    if args.where_gradient_probe_only:
        # P3A Block 4: calibration-only entry -- never reaches the optimizer loop.
        if S.dist.is_main:
            where_gradient_probe(S)
        return

    step = 0
    nan_count = 0
    if S.dist.is_main:   # R6: probes are rank-0 work; other ranks proceed to the loop
        eval_per_family(S, "baseline (step 0, pre-train)")    # general-not-DSL microscope (no-op unless flag)
        eval_selector(S, "baseline (step 0, pre-train)")      # floor-safe verify-and-select (no-op unless --selector-eval)
        pid_null_report(S, "baseline (step 0, pre-train)")    # demo-only MAIN strict (no-op unless pid flags)
        if args.zh_trace:
            zh_conditioning_report(S, "baseline (step 0, pre-train)")
    # Phase 1: dedicated generator so colour-perm doesn't perturb the fixed-eval RNG (eval_fixed
    # saves/restores its own state). TRAINING batches only -> the fixed-eval stays canonical.
    # R8: + rank*100003 decorrelates the per-rank augmentation streams (rank 0 => +0, so the
    # single-GPU seed and every gate comparison are unchanged).
    color_perm_gen = (torch.Generator().manual_seed(args.eval_seed + 777 + S.dist.rank * 100003)
                      if args.color_perm else None)
    # C-prime necessity pressure: dedicated CPU generator, same isolation rationale.
    pid_drop_gen = (torch.Generator().manual_seed(args.eval_seed + 888 + S.dist.rank * 100003)
                    if args.pid_dropout > 0 else None)
    while step < args.steps:
        for _set, cb, _g in S.loader:
            # R5: the shard filter must AGREE — ranks hold different shards, so one rank's bad
            # batch would otherwise desync the step counters and hang the next collective.
            _bad = 1.0 if ("context_inputs" not in cb or cb["context_inputs"].shape[1] < 2) else 0.0
            if _ddp_agree_max(S.dist, _bad) > 0:
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
            if args.color_perm:
                apply_color_perm(batch, color_perm_gen)          # Phase 1: anti-memorisation aug
            if args.pid_dropout > 0:
                # Per-ROW blank-PID on TRAINING batches only (frozen eval batches are never touched,
                # so the panel/abort guard read an un-dropped MAIN). On dropped rows the MAIN CE can
                # only fall by reading the demos -- the necessity pressure of run C-prime.
                _pd = (torch.rand(batch["puzzle_identifiers"].shape[0], generator=pid_drop_gen)
                       < args.pid_dropout).to(device)
                batch["puzzle_identifiers"] = torch.where(
                    _pd, torch.zeros_like(batch["puzzle_identifiers"]), batch["puzzle_identifiers"])
            lodo_w = scheduled_lodo_weight(args, step)
            loss_head.c2_delta_lodo_weight = lodo_w
            # Contrast rides the SAME warmup/ramp (as a fraction of the final lodo weight): the
            # discriminative pressure and the canvas CE arrive together, never full-strength on
            # random-init adapters (the measured MAIN-strict decay 100->70 before the ramp ended).
            _frac = (lodo_w / args.lodo_weight) if args.lodo_weight > 0 else 0.0
            loss_head.c2_delta_contrast_weight = float(args.contrast_weight) * _frac
            # LR warmup: ramp 0 -> lr over the first steps so random adapters don't spike -> NaN.
            # Scale each group's OWN base lr (setdefault captures it on step 0) so the dedicated
            # structure_relmap_proj lr survives the warmup instead of being clobbered to args.lr.
            warm = min(1.0, (step + 1) / float(max(1, args.lr_warmup_steps)))
            for grp in opt.param_groups:
                grp["lr"] = grp.setdefault("base_lr", grp["lr"]) * warm
            forward_error = None
            try:
                with torch.device(device):
                    carry = loss_head.initial_carry(batch)
                carry, loss_val, metrics, _, _ = loss_head(
                    carry=carry, batch=batch, return_keys=[])
            except FloatingPointError as exc:
                # Every rank must still reach the agreement collective. Reify the exception as
                # a non-finite local loss, then preserve the exact failing batch below.
                forward_error = exc
                loss_val = torch.full((), float("nan"), device=device)
                metrics = {}
            # G5c TEST HOOK (env-gated, inert in every real run): TRV2_TEST_NAN="rank:step"
            # forces a non-finite loss on exactly that rank+step so the 2-process gate can
            # prove the AGREED skip (rank 0 must skip on rank 1's NaN). Unset => dead branch.
            _tn = os.environ.get("TRV2_TEST_NAN", "")
            if _tn:
                _tr, _ts = _tn.split(":")
                if S.dist.rank == int(_tr) and step == int(_ts):
                    loss_val = loss_val + float("nan")
            # NaN GUARD: a single non-finite loss must NOT corrupt the weights -> skip the step.
            # R5: the decision is AGREED — a NaN on ANY rank skips the step on ALL ranks (else the
            # skipping rank misses the grad all-reduce and the others hang). nan_count therefore
            # advances identically on every rank, so the >=10 abort also agrees.
            _loss_bad = 0.0 if torch.isfinite(loss_val) else 1.0
            if _loss_bad > 0:
                _write_failure_artifact(
                    S, batch, step,
                    "forward_exception" if forward_error is not None else "nonfinite_loss",
                    detail=str(forward_error) if forward_error is not None else f"loss={loss_val}",
                )
            if _ddp_agree_max(S.dist, _loss_bad) > 0:
                nan_count += 1
                loss_head.zero_grad(set_to_none=True)
                print(f"[nan-guard] step {step}: non-finite loss skipped ({nan_count} total)")
                if nan_count >= 10:
                    print("[abort] too many non-finite losses; lower --lr or inspect the data/losses.")
                    return
                step += 1
                continue
            loss_head.zero_grad(set_to_none=True)   # zero ALL grads (incl. unused puzzle_emb)
            loss_val.backward()
            # Detect the originating rank BEFORE all-reduce. Otherwise one poisoned rank makes
            # every rank's reduced gradient non-finite and destroys provenance.
            _local_grad_bad = 1.0 if _has_nonfinite_grad(params) else 0.0
            if _local_grad_bad > 0:
                _write_failure_artifact(
                    S, batch, step, "nonfinite_local_grad",
                    detail="one or more scoped gradients were non-finite before all-reduce",
                )
            if _ddp_agree_max(S.dist, _local_grad_bad) > 0:
                nan_count += 1
                loss_head.zero_grad(set_to_none=True)
                print(f"[grad-guard] step {step}: pre-allreduce non-finite gradient "
                      f"-> step skipped ({nan_count} total)")
                if nan_count >= 10:
                    print("[abort] too many non-finite grads; lower --lr or inspect the failure artifacts.")
                    return
                step += 1
                continue
            # R2: average the scope grads across ranks BEFORE the clip, so the clip norm below is
            # the true global-batch norm and is bitwise-identical on every rank.
            _ddp_allreduce_grads(S.dist, params)
            # GRAD GUARD: the loss can be FINITE while the backward produces a non-finite GRADIENT.
            # clip_grad_norm_ does NOT sanitise this -- a NaN/inf total norm scales every grad to
            # NaN, and opt.step() then corrupts ALL weights. Skip the step whenever the PRE-clip
            # grad norm is non-finite -> weights stay finite, run continues.
            # R5: post-reduce the norm already agrees across ranks; the explicit agreement is
            # belt-and-braces against a backend returning rank-divergent reductions.
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            _grad_bad = 0.0 if torch.isfinite(grad_norm) else 1.0
            if _grad_bad > 0:
                _write_failure_artifact(
                    S, batch, step, "nonfinite_reduced_grad_norm",
                    detail=f"grad_norm={float(grad_norm):.9g}",
                )
            if _ddp_agree_max(S.dist, _grad_bad) > 0:
                nan_count += 1
                loss_head.zero_grad(set_to_none=True)   # discard the poisoned grads, do NOT step
                print(f"[grad-guard] step {step}: non-finite grad-norm {float(grad_norm):.3g} "
                      f"-> step skipped ({nan_count} total)")
                if nan_count >= 10:
                    print("[abort] too many non-finite grads; lower --lr or inspect the data/losses.")
                    return
                step += 1
                continue
            opt.step()
            completed_step = step + 1
            if completed_step in args.save_checkpoint_step_set:
                save_checkpoint(S, completed_step)
            if step % args.log_every == 0 or step == args.steps - 1:
                # R6: the fixed eval + panel are rank-0 work; the CONTINUE/ABORT verdict is
                # broadcast so every rank leaves the loop together (a lone rank returning while
                # the others enter the next all-reduce is the deadlock class).
                _cont = 1.0
                if S.dist.is_main:
                    # Read the FIXED eval (comparable), not this step's random training batch.
                    eval_m = eval_fixed(S)
                    loss_scalar = float(loss_val.item())
                    _rs = dict(nan=nan_count, grad_norm=float(grad_norm))
                    _cont = 1.0 if panel(S, step, loss_scalar, eval_m, lodo_w, run_stats=_rs) else 0.0
                if _ddp_bcast_scalar(S.dist, _cont) <= 0:
                    if S.dist.is_main:
                        pid_null_report(S, f"abort (step {step})")
                        main_recovery_report(S, f"abort (step {step})")
                        # Block 1: aborted runs are exactly the ones worth diagnosing; the NaN-grad
                        # abort above stays excluded (per-ex metrics from broken numerics mislead).
                        _task_metrics_finalize(S)
                    return
            if S.dist.is_main and args.zh_trace and args.zh_every > 0 and step > 0 and step % args.zh_every == 0:
                zh_conditioning_report(S, f"step {step}")   # TRAJECTORY: is C2 opening yet?
            step += 1
            if step >= args.steps:
                break

    if not S.dist.is_main:   # R6: end probes + checkpoint are rank-0 work; others wait at main()'s barrier
        return
    eval_per_family(S, f"end (step {step}, trained)")     # before/after vs the baseline call above
    eval_selector(S, f"end (step {step}, trained)")       # floor-safe verify-and-select
    pid_null_report(S, f"end (step {step}, trained)")     # did demo-only MAIN strict move?
    if args.zh_trace:
        zh_conditioning_report(S, f"end (step {step}, trained)")
    main_recovery_report(S, f"end (step {step})")         # is the floor deployable regardless of ON-drift?

    if args.preserve > 0:
        print("[done] Stage-2 GATE: keep (preserved_correct_frac) stays high (>0.9) while chg_col holds. "
              "Too-strong KL -> chg_col stalls (lower --preserve).")
    if args.changed_valid > 0:
        print("[done] Stage-3 GATE: mchg (main changed-colour acc) rises vs baseline while pad/eos hold.")
    if args.lodo_weight <= 0:
        print("[done] CALIBRATION MODE: final gated LODO CE stayed OFF. Judge MAIN stability.")
    elif args.lodo_warmup_steps > 0:
        print("[done] STAGED INTEGRATION: final gated LODO CE was warmed up/ramped. MAIN safety thresholds "
              "must hold before treating LODO gains as usable.")
    save_checkpoint(S, step, final=True)
    # Block 1+2 diagnostics run AFTER the final checkpoint (and nonfatally): a CSV/pairing failure
    # must never discard the artifact of an otherwise successful run.
    _task_metrics_finalize(S)


def _sha_case_id(pid, target_in, ctx_in, ctx_out, ctx_mask) -> str:
    """Stable per-example id: SHA-256 over pid + the exact tensors that define the case, so the same
    PID under different augmentations pairs UNAMBIGUOUSLY across a control run and a treatment run."""
    import hashlib
    h = hashlib.sha256()
    h.update(str(int(pid)).encode())
    for t in (target_in, ctx_in, ctx_out, ctx_mask):
        if t is None:
            h.update(b"\x00none")
        else:
            h.update(t.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


# Ordered (csv_column, loss-metrics key). WHERE cols need the positive gate BUILT and bind cols the
# binder BUILT (loss weights may be 0 -- the M0 measurement control); candidate cols are present
# whenever the aux/LODO pass runs. Absent key OR an empty per-row subset -> NaN (undefined, never 0).
_TASK_METRIC_KEYS = [
    ("where_f1", "c2_where_f1_per_ex"),
    ("where_fpr", "c2_where_fpr_per_ex"),
    ("where_has_changed", "c2_where_has_changed_per_ex"),
    ("where_mask_exact", "c2_where_mask_exact_per_ex"),
    ("bind_changed_acc", "c2_bind_changed_acc_per_ex"),
    ("bind_copy_acc", "c2_bind_copy_acc_per_ex"),
    ("bind_coverage", "c2_bind_support_coverage_per_ex"),
    ("bind_changed_exact", "c2_bind_changed_exact_per_ex"),
    ("bind_copy_exact", "c2_bind_copy_exact_per_ex"),
    ("candidate_changed_acc", "candidate_changed_acc_per_ex"),
    ("candidate_copy_acc", "candidate_copy_acc_per_ex"),
    ("candidate_structure_exact", "candidate_structure_exact_per_ex"),
    ("candidate_strict_exact", "candidate_strict_exact_per_ex"),
    # P1 VALUE extraction (training-free binder audit): the eight CSV metrics plus in-memory
    # contract-failure and supported-cell counts consumed only by the P1 panel.
    ("raw_bind_changed_acc", "p1_raw_bind_changed_acc_per_ex"),
    ("raw_bind_copy_acc", "p1_raw_bind_copy_acc_per_ex"),
    ("effective_changed_coverage", "p1_effective_changed_coverage_per_ex"),
    ("effective_copy_coverage", "p1_effective_copy_coverage_per_ex"),
    ("raw_bind_margin", "p1_raw_bind_margin_per_ex"),
    ("marginal_bind_changed_acc", "p1_marginal_bind_changed_acc_per_ex"),
    ("fixed_replacement_gain", "p1_fixed_replacement_gain_per_ex"),
    ("fixed_replacement_copy_loss", "p1_fixed_replacement_copy_loss_per_ex"),
    ("p1_norm_fail", "p1_norm_fail_per_ex"),
    ("p1_finite_fail", "p1_finite_fail_per_ex"),
    ("p1_changed_supported_cells", "p1_changed_supported_cells_per_ex"),
    # P3A Block 3: three-arm WHERE counterfactual on identical held-out rows. NaN outside the
    # shared row set / where the metric is undefined; diffs are computed per row downstream.
    ("where_correct_f1", "where_correct_f1_per_ex"),
    ("where_shuffle_f1", "where_shuffle_f1_per_ex"),
    ("where_zero_f1", "where_zero_f1_per_ex"),
    ("where_correct_fpr", "where_correct_fpr_per_ex"),
]

# The paired CSV schema (Block 1 contract). where_has_changed / where_mask_exact stay in-memory only
# (they drive the Block 2 mechanism panel), so the CSV columns match the spec exactly.
_TASK_CSV_COLS = [
    "eval_case_id", "puzzle_identifier", "family",
    "where_f1", "where_fpr",
    "bind_changed_acc", "bind_copy_acc", "bind_coverage", "bind_changed_exact", "bind_copy_exact",
    "candidate_changed_acc", "candidate_copy_acc", "candidate_structure_exact", "candidate_strict_exact",
    "raw_bind_changed_acc", "raw_bind_copy_acc",
    "effective_changed_coverage", "effective_copy_coverage",
    "raw_bind_margin", "marginal_bind_changed_acc",
    "fixed_replacement_gain", "fixed_replacement_copy_loss",
    "where_correct_f1", "where_shuffle_f1", "where_zero_f1", "where_correct_fpr",
]


@torch.no_grad()
def _collect_task_rows(S) -> list:
    """Run the frozen eval once and emit ONE row per held-out example: case id, pid, family, and every
    per-example WHERE/bind/candidate metric the enabled losses exposed. Uses the family-tagged per-family
    sets when available, else the fixed-eval set (family='all'). No optimizer step is ever taken."""
    args = S.args
    sources = list(S.family_eval_sets.items()) if S.family_eval_sets else [("all", S.eval_batches)]
    rows = []
    _collision_total = 0.0
    for fam, bset in sources:
        for i, eb in enumerate(bset):
            with _seeded_evaluation(S.loss_head, args.eval_seed + i):
                with torch.device(S.device):
                    c0 = S.loss_head.initial_carry(eb)
                _c, _l, m, _, _ = S.loss_head(carry=c0, batch=eb, return_keys=[])
            if "c2_canonical_bind_key_collisions" in m:
                _collision_total += float(m["c2_canonical_bind_key_collisions"])
            B = int(eb["inputs"].shape[0])
            pid = eb.get("puzzle_identifiers")
            ci = eb.get("context_inputs"); co = eb.get("context_outputs"); cm = eb.get("context_mask")
            for b in range(B):
                row = {
                    "eval_case_id": _sha_case_id(
                        int(pid[b]) if pid is not None else -1,
                        eb["inputs"][b],
                        ci[b] if ci is not None else None,
                        co[b] if co is not None else None,
                        cm[b] if cm is not None else None),
                    "puzzle_identifier": int(pid[b]) if pid is not None else -1,
                    "family": fam,
                }
                for col, key in _TASK_METRIC_KEYS:
                    v = m.get(key)
                    row[col] = float(v[b]) if (torch.is_tensor(v) and v.numel() > b) else float("nan")
                rows.append(row)
    S.task_metrics_agg = {"canonical_key_collisions": _collision_total}
    return rows


def dump_task_metrics(S, path: str, rows=None) -> list:
    """Write the per-example paired CSV (Block 1). Returns the in-memory rows (richer than the CSV) so
    the Block 2 mechanism panel can reuse them without a second eval pass."""
    import csv, math, os
    if rows is None:
        rows = _collect_task_rows(S)
    _parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(_parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TASK_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_where = sum(1 for r in rows if not math.isnan(r["where_f1"]))
    n_bind = sum(1 for r in rows if not math.isnan(r["bind_changed_acc"]))
    print(f"[task-metrics] wrote {len(rows)} per-example rows -> {path} "
          f"(WHERE populated={n_where}, bind populated={n_bind})")
    return rows


def paired_control_report(treatment_csv: str, control_csv: str, n_boot: int, seed: int) -> None:
    """Paired per-case treatment-minus-control deltas with a FAMILY-STRATIFIED bootstrap CI. Treatment
    is this run's --task-metrics-out; control is a prior run's CSV. Fails on duplicate or mismatched
    eval_case_id so the pairing is exact and task difficulty cancels as common-mode."""
    import csv, math, random
    from collections import defaultdict

    if int(n_boot) < 1:
        raise ValueError(f"--paired-bootstrap-samples must be >= 1, got {n_boot}")
    n_boot = int(n_boot)

    def _load(p):
        d = {}
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                cid = r["eval_case_id"]
                if cid in d:
                    raise ValueError(f"duplicate eval_case_id in {p}: {cid}")
                d[cid] = r
        return d

    treat = _load(treatment_csv); ctrl = _load(control_csv)
    ts, cs = set(treat), set(ctrl)
    if ts != cs:
        raise ValueError(
            f"eval_case_id mismatch: {len(ts - cs)} only in treatment, {len(cs - ts)} only in "
            f"control (paired comparison requires identical case sets -- same eval seed/data/batch).")
    ids = sorted(ts)
    _fam_mm = [cid for cid in ids
               if treat[cid].get("family", "all") != ctrl[cid].get("family", "all")]
    if _fam_mm:
        raise ValueError(
            f"family label mismatch on {len(_fam_mm)} paired case(s) (first: {_fam_mm[0]} -- "
            f"treatment='{treat[_fam_mm[0]].get('family')}' vs control='{ctrl[_fam_mm[0]].get('family')}'); "
            f"the stratified bootstrap requires identical --per-family-eval settings in both runs.")
    metrics = [c for c in _TASK_CSV_COLS if c not in ("eval_case_id", "puzzle_identifier", "family")]
    rnd = random.Random(seed)
    print(f"\n[PAIRED CONTROL REPORT]  treatment={treatment_csv}")
    print(f"                         control  ={control_csv}  ({len(ids)} paired cases, "
          f"{n_boot} family-stratified bootstraps)")
    print(f"  {'metric':26s} {'ctrl':>7s} {'treat':>7s} {'dmean':>8s} {'dmed':>8s} "
          f"{'ci_lo':>8s} {'ci_hi':>8s} {'+':>5s} {'0':>5s} {'-':>5s} {'n':>5s}")
    for mk in metrics:
        fam_deltas = defaultdict(list); deltas = []; tvals = []; cvals = []
        for cid in ids:
            try:
                tv = float(treat[cid][mk]); cv = float(ctrl[cid][mk])
            except (KeyError, ValueError):
                continue
            if math.isnan(tv) or math.isnan(cv):
                continue
            fam_deltas[treat[cid].get("family", "all")].append(tv - cv)
            deltas.append(tv - cv); tvals.append(tv); cvals.append(cv)
        n = len(deltas)
        if n == 0:
            print(f"  {mk:26s}   (no finite paired cases -- is the metric's loss enabled in BOTH runs?)")
            continue
        dmean = sum(deltas) / n
        dmed = sorted(deltas)[n // 2]
        pos = sum(1 for d in deltas if d > 1e-9)
        neg = sum(1 for d in deltas if d < -1e-9)
        boots = []
        for _ in range(n_boot):
            acc = 0.0; cnt = 0
            for ds in fam_deltas.values():
                k = len(ds)
                for _ in range(k):
                    acc += ds[rnd.randrange(k)]
                cnt += k
            boots.append(acc / max(cnt, 1))
        boots.sort()
        lo = boots[min(int(0.025 * n_boot), n_boot - 1)]
        hi = boots[min(int(0.975 * n_boot), n_boot - 1)]
        print(f"  {mk:26s} {sum(cvals)/n:7.3f} {sum(tvals)/n:7.3f} {dmean:8.3f} {dmed:8.3f} "
              f"{lo:8.3f} {hi:8.3f} {pos:5d} {n-pos-neg:5d} {neg:5d} {n:5d}")
    print("  [read] ci excluding 0 => paired-significant; family-stratified resampling stops a large "
          "family from dominating the CI.")


# Pre-registered mechanism-eligibility thresholds (Block 2). Fixed BEFORE looking at results so the
# conditional-exact rate cannot be p-hacked. A row is eligible iff WHERE and VALUE are BOTH confident.
_MECH_WHERE_F1_MIN = 0.70
_MECH_WHERE_FPR_MAX = 0.20
_MECH_BIND_CHANGED_ACC_MIN = 0.70
_MECH_BIND_COVERAGE_MIN = 0.50


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (better than normal approx at small n / p near 0)."""
    import math
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mechanism_conditioned_exact(rows: list) -> None:
    """Block 2 (diagnostic ONLY): on the rows where WHERE and VALUE are both confident, does the
    candidate actually produce an EXACT grid? This is the leading indicator that bridges the per-cell
    -> per-grid gap (e.g. 44.9% changed-acc but 0% exact). Never touches selection, loss, or inference."""
    import math

    def _fin(x):
        return isinstance(x, float) and not math.isnan(x)

    eligible = []
    for r in rows:
        wf1, wfpr = r.get("where_f1"), r.get("where_fpr")
        bca, bcov = r.get("bind_changed_acc"), r.get("bind_coverage")
        if not (_fin(wf1) and _fin(wfpr) and _fin(bca) and _fin(bcov)):
            continue
        if (wf1 >= _MECH_WHERE_F1_MIN and wfpr <= _MECH_WHERE_FPR_MAX
                and bca >= _MECH_BIND_CHANGED_ACC_MIN and bcov >= _MECH_BIND_COVERAGE_MIN):
            eligible.append(r)
    n_elig = len(eligible)
    n_exact = sum(1 for r in eligible
                  if _fin(r.get("candidate_strict_exact")) and r["candidate_strict_exact"] >= 0.5)
    rate = (n_exact / n_elig) if n_elig else float("nan")
    lo, hi = _wilson_ci(n_exact, n_elig)

    # Stricter conditional: rows where the WHERE mask AND the bind values are per-cell EXACT.
    cond = [r for r in rows
            if _fin(r.get("where_mask_exact")) and r["where_mask_exact"] >= 0.5
            and _fin(r.get("bind_changed_exact")) and r["bind_changed_exact"] >= 0.5
            and _fin(r.get("bind_copy_exact")) and r["bind_copy_exact"] >= 0.5]
    cond_rate = (sum(1 for r in cond
                     if _fin(r.get("candidate_strict_exact")) and r["candidate_strict_exact"] >= 0.5)
                 / len(cond)) if cond else float("nan")

    print("\n[MECHANISM-CONDITIONED EXACT]  (diagnostic only -- does confident mechanism => exact grid?)")
    print("  NOTE: 'exact' here = LODO RECONSTRUCTION exactness of the held-out demo on the eval canvas,"
          " NOT ARC test-output exactness. It is the leading indicator, not the score.")
    print(f"  eligibility: WHERE_f1>={_MECH_WHERE_F1_MIN} & WHERE_fpr<={_MECH_WHERE_FPR_MAX} & "
          f"bind_changed_acc>={_MECH_BIND_CHANGED_ACC_MIN} & bind_coverage>={_MECH_BIND_COVERAGE_MIN}")
    print(f"  mechanism_eligible_count = {n_elig}")
    print(f"  mechanism_exact_count    = {n_exact}   (candidate STRICT exact, not floor)")
    print(f"  mechanism_exact_rate     = {rate:.3f}  wilson95=[{lo:.3f},{hi:.3f}]")
    print(f"  exact_given_where_exact_and_bind_exact = {cond_rate:.3f}  (n={len(cond)})")
    if n_elig == 0:
        print("  [note] 0 eligible -- WHERE per-ex needs the positive gate BUILT (--token-gate-where "
              "--positive-where-gate) and bind per-ex needs the binder BUILT (--canonical-value-binder "
              "or --value-ctx-bind); loss weights may be 0 (M0). NaN also means the row's subset is "
              "empty (copy-only / all-changed) or the mechanism is still below the thresholds.")


# P1 gate thresholds (pre-registered; Codex Phase-1 spec). Primary decisions on conditional_recolor.
_P1_MIN_RAW_VS_MARGINAL_PP = 5.0
_P1_MIN_REPLACEMENT_GAIN_PP = 5.0
_P1_MAX_COPY_LOSS_PP = 1.0
_P1_BOOT_N = 10000
_P1_BOOT_SEED = 1234


def p1_value_extraction_report(rows: list, agg: dict, json_path: str) -> bool | None:
    """[P1 VALUE EXTRACTION] -- training-free audit of the canonical binder TABLE. Primary gate is
    computed on conditional_recolor ONLY (the binder is same-position by design; other families are
    diagnostics). Task-level paired bootstrap over fixed_replacement_gain (each task's gain is
    already its own paired replacement-minus-base delta; tasks are resampled, never cells).
    Writes the JSON report BEFORE printing the verdict; never raises past the finalize guard.

    P3A Block 0: when the primary family cannot be evaluated (no conditional_recolor rows, or none
    with a finite gain) the verdict is UNAVAILABLE and the return value is None -- the report NEVER
    substitutes ALL families and stamps PASS/FAIL on the wrong population."""
    import json
    import math
    import os
    import random

    def _fin(x):
        return isinstance(x, float) and math.isfinite(x)

    def _nanmean(vals):
        vals = [v for v in vals if _fin(v)]
        return (sum(vals) / len(vals)) if vals else float("nan")

    def _unavailable(reason: str, n_rows: int):
        report = {"family": "conditional_recolor", "verdict": "UNAVAILABLE", "pass": None,
                  "reason": reason, "n_rows": n_rows}
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print("\n[P1 VALUE EXTRACTION]  (training-free binder audit; "
              "primary family = conditional_recolor)")
        print(f"  {reason}")
        print(f"  [P1 VERDICT] UNAVAILABLE  (report -> {json_path})")
        return None

    if not any(_fin(r.get("fixed_replacement_gain")) for r in rows):
        print("\n[P1 VALUE EXTRACTION] skipped -- no finite fixed_replacement_gain anywhere "
              "(canonical binder not active this run, or no changed cells in the eval sets).")
        return None

    primary = [r for r in rows if r.get("family") == "conditional_recolor"]
    fam_label = "conditional_recolor"
    gains = [r["fixed_replacement_gain"] for r in primary if _fin(r.get("fixed_replacement_gain"))]
    if not primary:
        return _unavailable(
            "no conditional_recolor rows in this run -- pass --per-family-eval so the primary "
            "family exists; PASS/FAIL is undefined on other families", 0)
    if not gains:
        return _unavailable(
            f"{len(primary)} conditional_recolor rows but none carries a finite "
            "fixed_replacement_gain -- the gate cannot be evaluated", len(primary))

    m = {k: _nanmean([r.get(k, float("nan")) for r in primary]) for k in (
        "raw_bind_changed_acc", "raw_bind_copy_acc", "effective_changed_coverage",
        "effective_copy_coverage", "raw_bind_margin", "marginal_bind_changed_acc",
        "fixed_replacement_gain", "fixed_replacement_copy_loss")}
    n_tasks = len(gains)
    supported_cells = sum(r.get("p1_changed_supported_cells", 0.0) for r in primary
                          if _fin(r.get("p1_changed_supported_cells")))
    norm_fails = sum(r.get("p1_norm_fail", 0.0) for r in rows if _fin(r.get("p1_norm_fail")))
    finite_fails = sum(r.get("p1_finite_fail", 0.0) for r in rows if _fin(r.get("p1_finite_fail")))
    collisions = float(agg.get("canonical_key_collisions", float("nan")))

    ci_lo = ci_hi = float("nan")
    if n_tasks > 0:
        rnd = random.Random(_P1_BOOT_SEED)
        boots = []
        for _ in range(_P1_BOOT_N):
            boots.append(sum(gains[rnd.randrange(n_tasks)] for _ in range(n_tasks)) / n_tasks)
        boots.sort()
        ci_lo = boots[int(0.025 * (_P1_BOOT_N - 1))]
        ci_hi = boots[int(0.975 * (_P1_BOOT_N - 1))]

    raw_vs_marginal_pp = (m["raw_bind_changed_acc"] - m["marginal_bind_changed_acc"]) * 100.0
    gain_pp = m["fixed_replacement_gain"] * 100.0
    copy_loss_pp = m["fixed_replacement_copy_loss"] * 100.0
    gates = {
        "key_collisions_zero": collisions == 0.0,
        "raw_vs_marginal_ge_5pp": _fin(raw_vs_marginal_pp) and raw_vs_marginal_pp >= _P1_MIN_RAW_VS_MARGINAL_PP,
        "replacement_gain_ge_5pp": _fin(gain_pp) and gain_pp >= _P1_MIN_REPLACEMENT_GAIN_PP,
        "bootstrap_lower_gt_0": _fin(ci_lo) and ci_lo > 0.0,
        "copy_loss_le_1pp": _fin(copy_loss_pp) and copy_loss_pp <= _P1_MAX_COPY_LOSS_PP,
        "no_contract_failures": norm_fails == 0.0 and finite_fails == 0.0,
    }
    verdict = all(gates.values())

    report = {
        "family": fam_label, "n_tasks_finite": n_tasks,
        "supported_changed_cells": supported_cells,
        "aggregates": m,
        "raw_vs_marginal_pp": raw_vs_marginal_pp,
        "fixed_replacement_gain_pp": gain_pp,
        "fixed_replacement_copy_loss_pp": copy_loss_pp,
        "bootstrap": {"n": _P1_BOOT_N, "seed": _P1_BOOT_SEED, "ci95": [ci_lo, ci_hi]},
        "key_collisions": collisions,
        "normalization_failures": norm_fails, "finite_failures": finite_fails,
        "gates": gates, "pass": verdict,
    }
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[P1 VALUE EXTRACTION]  (training-free binder audit; primary family = {fam_label})")
    print(f"  tasks(finite)={n_tasks}  supported_changed_cells={supported_cells:.0f}  "
          f"collisions={collisions:.0f}  norm_fail={norm_fails:.0f} finite_fail={finite_fails:.0f}")
    print(f"  raw_bind_changed_acc={m['raw_bind_changed_acc']:.3f}  "
          f"marginal={m['marginal_bind_changed_acc']:.3f}  (raw-marginal={raw_vs_marginal_pp:+.1f}pp)")
    print(f"  raw_bind_copy_acc={m['raw_bind_copy_acc']:.3f}  raw_bind_margin={m['raw_bind_margin']:.3f}")
    print(f"  effective_coverage changed={m['effective_changed_coverage']:.3f} "
          f"copy={m['effective_copy_coverage']:.3f}")
    print(f"  fixed_replacement_gain={gain_pp:+.1f}pp  ci95=[{ci_lo * 100.0:+.1f}pp,{ci_hi * 100.0:+.1f}pp]  "
          f"copy_loss={copy_loss_pp:+.2f}pp")
    for name, ok in gates.items():
        print(f"    gate {name:28s} {'PASS' if ok else 'FAIL'}")
    print(f"  [P1 VERDICT] {'PASS' if verdict else 'FAIL'}  (report -> {json_path})")
    if not verdict and n_tasks > 0 and supported_cells < 50:
        print("  [note] supported-cell count is small -- do not over-read accuracy on a negligible subset.")
    return verdict


def p3a_where_report(rows: list) -> None:
    """[P3A WHERE] -- three-arm support counterfactual (correct / matched shuffle / zero) scored on
    identical held-out rows. Task-level PAIRED means: each task contributes its own correct-minus-
    shuffle / correct-minus-zero delta, so the panel is robust to per-task difficulty. Diagnostic
    print only -- the pre-registered P3A acceptance gate is applied to the decision run's numbers."""
    import math

    def _fin(x):
        return isinstance(x, float) and math.isfinite(x)

    trip = [r for r in rows
            if _fin(r.get("where_correct_f1")) and _fin(r.get("where_shuffle_f1"))
            and _fin(r.get("where_zero_f1"))]
    if not trip:
        return
    n = len(trip)
    c = sum(r["where_correct_f1"] for r in trip) / n
    s = sum(r["where_shuffle_f1"] for r in trip) / n
    z = sum(r["where_zero_f1"] for r in trip) / n
    d_s = sum(r["where_correct_f1"] - r["where_shuffle_f1"] for r in trip) / n
    d_z = sum(r["where_correct_f1"] - r["where_zero_f1"] for r in trip) / n
    fprs = [r["where_correct_fpr"] for r in trip if _fin(r.get("where_correct_fpr"))]
    fpr = (sum(fprs) / len(fprs)) if fprs else float("nan")
    print(f"\n[P3A WHERE]  three-arm counterfactual (identical held-out rows; tasks={n})")
    print(f"  macro_f1 correct={c:.3f} shuffle={s:.3f} zero={z:.3f}")
    print(f"  paired correct-shuffle={d_s * 100.0:+.1f}pp  correct-zero={d_z * 100.0:+.1f}pp  "
          f"correct_fpr={fpr:.3f}")


def _task_metrics_finalize(S) -> None:
    """Block 1+2 exit hook (rank-0): per-example CSV, mechanism panel, optional paired report.
    Called from BOTH the normal end-of-run path (AFTER the final checkpoint) and the MAIN-degraded
    abort path, so aborted runs still leave a diagnosable artifact. NONFATAL by design: a
    diagnostics failure is reported loudly but must never change the run's outcome or exit path.
    No-op unless --task-metrics-out is set."""
    if not S.args.task_metrics_out:
        return
    try:
        rows = dump_task_metrics(S, S.args.task_metrics_out)
        mechanism_conditioned_exact(rows)
        _p1_json = os.path.splitext(S.args.task_metrics_out)[0] + "_p1.json"
        p1_value_extraction_report(rows, getattr(S, "task_metrics_agg", {}), _p1_json)
        p3a_where_report(rows)
        if S.args.paired_control_report:
            paired_control_report(S.args.task_metrics_out, S.args.paired_control_report,
                                  S.args.paired_bootstrap_samples, S.args.paired_bootstrap_seed)
    except Exception as e:                                    # noqa: BLE001 -- diagnostics-only lane
        print(f"[task-metrics ERROR] diagnostics failed ({type(e).__name__}: {e}). "
              f"The run itself (and any saved checkpoint) is unaffected.")


def main() -> None:
    args = build_parser().parse_args()
    dist_ctx = init_distributed()
    raw = build_config(args)
    if args.dump_config:
        # G1 hook: the resolved config BEFORE any dataloader/model work, exactly what a run
        # would save as all_config.yaml at the end.
        print(yaml.safe_dump(raw, sort_keys=False))
        return
    config, loader, meta, loss_head, device = build_data_and_model(args, raw, dist_ctx)
    # R6: the frozen eval + family sets are rank-0 consumers; non-main ranks skip the
    # collection (their loaders stay unadvanced -- shards are disjoint, no alignment needed).
    eval_loader = build_eval_loader(config) if dist_ctx.is_main else None
    eval_batches = collect_eval_batches(args, eval_loader, device) if dist_ctx.is_main else []
    family_eval_sets = build_family_eval_sets(args, config, device) if dist_ctx.is_main else None
    S = SimpleNamespace(args=args, raw=raw, config=config, loader=loader, meta=meta,
                        loss_head=loss_head, device=device, eval_batches=eval_batches,
                        family_eval_sets=family_eval_sets, dist=dist_ctx, drv2_warned=False,
                        saved_checkpoint_steps=set())
    _write_run_manifests(S)
    # Diagnostic early-exits (same position as the oracle: after eval/probe setup, before scope).
    if args.zh_check:
        zh_conditioning_report(S)
        return
    if args.shape_debug:
        shape_debug(S)
        return
    params, scope_label = select_scope(args, loss_head)
    opt = build_optimizer(args, loss_head, params, scope_label)
    S.params, S.scope_label, S.opt = params, scope_label, opt
    start_banner(S)      # M-F: the first-minute gate as printed contract
    score_static(S)      # M-G static: evidence-vs-truth scorecard, once (caches dyn masks)
    train(S)
    if dist_ctx.world > 1:
        # All ranks meet here (non-main returns from train() early; rank 0 finishes probes +
        # checkpoint first), then the group tears down cleanly.
        import torch.distributed as dist
        dist.barrier(group=dist_ctx.cpu_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
