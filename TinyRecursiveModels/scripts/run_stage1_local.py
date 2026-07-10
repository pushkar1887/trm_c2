"""LOCAL training probe for the V3 factored head + LODO cross-demo signal.

Runs a short optimizer loop (default 300 steps) on aug-1000 warm-started from step_518071
and prints the MAIN/LODO/LEVER health panels every 10 steps. No wandb; checkpoints are
optional via --save-checkpoint-dir. This is still a local health runner, not the full
pretrain.py trainer.

CMD (from D:\\trm_c2\\TinyRecursiveModels):
  trm\\Scripts\\python.exe scripts\\run_stage1_local.py --c2-relmap --v3-clean --lodo-pad-weight 1.0 --lodo-eos-weight 1.0 --lodo-copy-weight 4.0 --relmap-outside-grid --structure-outside-warm-init --struct-lr 1e-2
"""
from __future__ import annotations

import argparse
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
from models.color_perm import apply_color_perm  # noqa: E402

CONFIG = "checkpoints/TRM-FVR-Experiments/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/all_config.yaml"
CKPT = r"D:\trm_c2\step_518071"
DATASET = r"D:\trm_c2\arc1concept-aug-1000"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=CONFIG,
                    help="Base all_config yaml. Default = the legacy trm_fvr_c2 base (unchanged behaviour). "
                         "Pass config/stage1_local_v2_base.yaml to run the trm_fvr_v2 rewrite (step-0 "
                         "byte-identical at legacy flags; the V2-only evidence flags below need it).")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=None,
                    help="Full-dataset mode: override --steps with the pretrain.py step formula "
                         "epochs * total_groups * mean_puzzle_examples / batch.")
    ap.add_argument("--save-checkpoint-dir", type=str, default="",
                    help="If set, save final model state_dict as step_N plus all_config.yaml in this directory.")
    ap.add_argument("--preserve", type=float, default=1.0, help="c2_delta_preserve_weight (Stage 2 KL)")
    ap.add_argument("--changed-valid", type=float, default=0.0, help="c2_changed_valid_loss_weight (Stage 3)")
    ap.add_argument("--selector-eval", action="store_true",
                    help="VERIFY-AND-SELECT (Take #1 / composition-C): eval-only floor-safe selector. Per task "
                         "scores the two model candidates -- gg=0 (deterministic floor) vs gg=on (head) -- by LODO "
                         "reconstruction, commits the best (tie->floor), reports the selector>=floor GUARANTEE + "
                         "head-vs-floor + selection headroom. No training/model change.")
    ap.add_argument("--floor-candidate-split", action="store_true",
                    help="V3 safety split: MAIN/logits use lm_head FLOOR, while LODO trains/scores the factored "
                         "V3 candidate. Selector metrics compare candidate vs floor and tie to floor.")
    ap.add_argument("--candidate-floor-structure", action="store_true",
                    help="With --floor-candidate-split, build the candidate from FLOOR PAD/EOS/VALID structure "
                         "plus V3 colour values on floor-valid cells. This tests colour improvement without "
                         "asking the candidate to relearn canvas structure.")
    ap.add_argument("--per-family-eval", action="store_true",
                    help="MICROSCOPE (general-not-DSL): break the LODO panel by codex task family on "
                         "aug-0 at baseline + end. Measurement only -- NOT a per-family solver/dispatch.")
    ap.add_argument("--per-family-collect", type=int, default=960,
                    help="aug-0 tasks to scan for the per-family eval (priority tasks are scattered across "
                         "the 960; each family is then capped to keep the eval fast)")
    ap.add_argument("--color-perm", action="store_true",
                    help="Phase 1: per-task colour-palette permutation on TRAINING batches "
                         "(anti-memorisation; fixed-eval stays canonical). MAIN WILL DROP -- that is "
                         "the success signal, not a regression. Use with --no-abort-on-main-drop.")
    ap.add_argument("--router-gate", action="store_true",
                    help="Phase 3: gate the colour-repair head by a per-task self-consistency score "
                         "(demo-only). CLEAN recolours fire the head; relational/structural tasks shut it. "
                         "Validated: accept>=0.9 -> 88pct coverage vs reject -> 24pct.")
    ap.add_argument("--router-threshold", type=float, default=0.9,
                    help="Phase 3: router fires fully at/above this self-consistency; shuts below (band 0.2).")
    ap.add_argument("--palette-constrain", action="store_true",
                    help="Phase 3: penalise output colours absent from the input grid (no-invention prior, "
                         "97pct on relational), strength=(1-router_mult) so CLEAN tasks stay free. Pre-wired "
                         "for the relational path; dormant while relational tasks are router-shut.")
    ap.add_argument("--task-palette", action=argparse.BooleanOptionalAction, default=True,
                    help="V3 colour-head task palette: expose support-in/out + target-input palette as a "
                         "color_head feature AND apply a soft absent-colour bias. DEFAULT ON; disable with "
                         "--no-task-palette (e.g. to A/B the palette prior or restore step-0==floor identity).")
    ap.add_argument("--task-palette-feature", action="store_true",
                    help="Ablation: expose the task palette to color_head input without applying the logit bias.")
    ap.add_argument("--task-palette-bias", action="store_true",
                    help="Ablation: apply the task-palette absent-colour bias without widening color_head input.")
    ap.add_argument("--task-palette-strength", type=float, default=4.0,
                    help="Soft logit penalty for colours absent from support inputs/outputs and target input.")
    ap.add_argument("--task-palette-hard", action="store_true",
                    help="Hard-mask absent task-palette colours. Use only after target coverage is proven safe.")
    ap.add_argument("--rel-where-hint", action="store_true",
                    help="Fold the PASSED ObjectBank/relmap WHERE evidence into the existing V3 color_head "
                         "as an input feature only. No VALUE writer, no candidate acceptance, no PAD/EOS edits.")
    ap.add_argument("--rel-where-topk", type=int, default=1,
                    help="FIX D: expose the top-K rel-where predicate masks (each scaled by its score) as "
                         "evidence columns instead of the single hard winner. 1 = legacy. The WHERE gate "
                         "used by value-v2/quarantine stays channel 0, so their semantics never change.")
    ap.add_argument("--algo-where-maps", action="store_true",
                    help="FIX C: 21 algorithmic WHERE/VALUE evidence columns from cell_conditioning_signature "
                         "cols 11/12 -- enclosed flood-fill flag + enclosing-colour one-hot + nearest-seed-"
                         "colour one-hot. Separate zero-init color_evidence_proj columns, never a writer.")
    ap.add_argument("--pairdelta-intent-hint", action="store_true",
                    help="Fold cheap PairDelta intent diagnostics into the existing V3 color_head as a "
                         "per-task input feature only. PairDelta remains a router/evidence signal, not a solver.")
    ap.add_argument("--transition-hint", action="store_true",
                    help="VALUE binding: per-cell demo-consensus P(out_colour|in_colour) over CHANGED support "
                         "cells, gathered by each target cell's input colour -> 10 zero-init color_head "
                         "columns. Routes the transition evidence DIRECTLY to the colour decision (one linear "
                         "layer), bypassing the frozen recurrence that attenuates input-side demo injections. "
                         "LODO-safe (active-context only), F7-safe (zero-init), never a writer.")
    ap.add_argument("--value-evidence-v2", action="store_true",
                    help="Richer VALUE evidence for the existing color_head: copy-vs-change rates plus "
                         "context-conditioned changed-colour distribution with rel-WHERE gating. Evidence "
                         "only, zero-init, no CTBank writer, no candidate executor, no PAD/EOS edits.")
    ap.add_argument("--value-v2-rich-ctx", action="store_true",
                    help="Use ObjectBank cell_conditioning_signature as VALUE-V2's context key instead of "
                         "the coarse relmap bucket. Requires --value-evidence-v2. Default off; evidence "
                         "only, zero-init, no writer.")
    # --- V2-model-only evidence (require --config config/stage1_local_v2_base.yaml; the old model
    #     ignores unknown c2_* keys as pydantic extras, so setting them against trm_fvr_c2 is a no-op).
    ap.add_argument("--verified-frame-evidence", action="store_true",
                    help="[V2 model] D1/E-1: verified-frame applied grid as 11 color_evidence_proj columns "
                         "(10 one-hot + conf). The Lane-A exactness proof as evidence; zero-init, F7-safe.")
    ap.add_argument("--analogy-evidence", action="store_true",
                    help="[V2 model] D2/E-2: analogy per-cell colour distribution as 11 columns "
                         "(10 dist + conf). Zero-init, F7-safe.")
    ap.add_argument("--value-ctx-gate", action="store_true",
                    help="[V2 model] D7: context-conditioned copy/change gate, 2 columns "
                         "P(change|src,ctx)/P(copy|src,ctx) with source-marginal backoff. Zero-init.")
    ap.add_argument("--value-v2-backoff", action="store_true",
                    help="[V2 model] D4/M1: collision-free bounded VALUE-V2 context key "
                         "(enclosure x seed x bg, no modulo) instead of the hash-%%512 rich-ctx bucket.")
    ap.add_argument("--pairdelta-color-evidence", action="store_true",
                    help="[V2 model] D8/File#5: pair-delta cross-demo agreement + positional WHERE prior "
                         "as 14 color_evidence_proj columns (consensus P(dst|src) counted once-per-demo, "
                         "min change rate over demos, row/col band priors). Zero-init, F7-safe.")
    ap.add_argument("--pairdelta-structure-evidence", action="store_true",
                    help="[V2 model] D9/File#5: verified {preserve, transpose, bbox} extent-family masks "
                         "-> zero-init [6->3] structure_pairdelta_proj (families the extent engine's "
                         "{identity, constant, ratio} set cannot express). Rides the struct-lr group.")
    ap.add_argument("--pairdelta-bidi-evidence", action="store_true",
                    help="[V2 model] D10/SS7: reverse-direction (y->x) evidence as 4 columns -- "
                         "invertibility (bijection => trust the mapping), per-src deletion rate + "
                         "cross-demo min, dst-mass (is this colour a rule OUTPUT). Zero-init, F7-safe.")
    ap.add_argument("--pairdelta-input-conf-gate", action="store_true",
                    help="[V2 model] SS7 reuse: multiply the --pairdelta-input rule_vec broadcast by the "
                         "encoder's own rule_confidence (was dead). Low-signal tasks shrink the hint "
                         "instead of feeding the norm growth that breaks MAIN-ON.")
    ap.add_argument("--value-v2-aux-weight", type=float, default=0.0,
                    help="Explicit LODO loss on the VALUE-V2-only color_head contribution. This forces the "
                         "new evidence columns to become predictive instead of being ignored by the full "
                         "colour head. 0 = off.")
    ap.add_argument("--value-v2-aux-changed-weight", type=float, default=1.0,
                    help="Changed-cell weight inside the VALUE-V2-only auxiliary CE.")
    ap.add_argument("--value-v2-aux-copy-weight", type=float, default=1.0,
                    help="Copy-cell weight inside the VALUE-V2-only auxiliary CE.")
    ap.add_argument("--contrast-per-row", action="store_true",
                    help="Per-task changed-cell contrast hinge instead of the batch-scalar hinge: each LODO row "
                         "is hinged separately and rows WITHOUT a valid wrong-task shuffle are masked (the "
                         "batch-scalar form dilutes the gap and leaks gradient through same-task shuffles).")
    ap.add_argument("--contrast-weight", type=float, default=1.0,
                    help="Final two-region contrast weight. SCHEDULED with the LODO warmup/ramp (was fixed at "
                         "1.0 from step 0 -- full-strength contrast on RANDOM-init adapters drifted z_H and "
                         "broke MAIN strict before the LODO CE even started; measured: strict 100->70 by step "
                         "90 with lodo_w still ramping).")
    ap.add_argument("--color-mlp", type=int, default=0,
                    help="Interaction capacity for the colour head: adds a zero-init-output MLP residual of "
                         "this hidden dim on the SAME feature concat (c2_color_head_mlp_dim). A linear head "
                         "cannot express input-colour x evidence products ('IF colour a THEN b'); 128 suggested. "
                         "F7-safe: step-0 byte-identical to the warm-started linear head. 0 = off.")
    ap.add_argument("--quarantine-candidate", action="store_true",
                    help="PID-QUARANTINED candidate colour head: a small MLP over EXCLUSIVELY PID-free "
                         "evidence (input one-hot + 3x3 neighbourhood + transition hint + rel-where + "
                         "palette + intent + relmap; z_H never read) supplies the CANDIDATE colour choice; "
                         "the canvas stays the solved floor structure. Closes the three measured cheats by "
                         "construction: cannot memorise (no PID), colour-perm leaves only table-reading, "
                         "and with --train-scope quarantine z_H is not in the gradient path (MAIN risk "
                         "structurally zero -- no warmup needed, lodo weight can be 1.0). Warm-init = "
                         "copy-unless-consensus (step-0 == the D1 deterministic baseline). Needs "
                         "--floor-candidate-split; pair with --candidate-floor-structure + --color-perm.")
    ap.add_argument("--quarantine-hidden", type=int, default=256,
                    help="Hidden dim of the quarantine head's MLP residual (c2_quarantine_hidden).")
    ap.add_argument("--pid-dropout", type=float, default=0.0,
                    help="NECESSITY PRESSURE (C-prime): fraction of TRAINING-batch rows whose puzzle_identifier "
                         "is replaced by the blank PID(0), so memorisation cannot reduce their MAIN CE and "
                         "reading the demos becomes the only remaining gradient. Runner-side + per-row: the "
                         "frozen eval batches are NEVER dropped, so the panel/abort guard stay honest. Read "
                         "the [pid-null] report: null-pid MAIN strict rising = the pressure is working. "
                         "Use with --unified (a frozen core cannot re-route around a missing PID) + --color-perm.")
    ap.add_argument("--pid-null-eval", action="store_true",
                    help="Print the [pid-null] diagnostic (MAIN strict with true PID vs blank PID(0) on the "
                         "frozen eval batches) at baseline+end even without --pid-dropout. null-pid strict "
                         "IS demo-only performance -- the honest generalization number.")
    ap.add_argument("--structure-alpha", type=float, default=0.0,
                    help="B2: c2_structure_fusion_alpha - fuse trained structure_head PAD/EOS into the "
                         "output logits. 0.0=aux-only (default); ~0.3 AFTER structure_head trains.")
    ap.add_argument("--lodo-weight", type=float, default=0.05, help="final gated-output LODO CE weight after warmup")
    ap.add_argument("--lodo-pad-weight", type=float, default=0.0,
                    help="PAD term inside c2_delta_lodo canvas CE. Keep tiny under --floor-candidate-split; "
                         "0 isolates colour.")
    ap.add_argument("--lodo-eos-weight", type=float, default=0.0,
                    help="EOS term inside c2_delta_lodo canvas CE. Keep tiny under --floor-candidate-split; "
                         "0 isolates colour. NOTE: eos is ~68 cells -> a high weight (3.0) over-predicts the "
                         "frame and EATS valid cells (collapses shape_task). ~1.0 is the balanced value.")
    ap.add_argument("--lodo-copy-weight", type=float, default=1.0,
                    help="COPY (unchanged-colour) term inside c2_delta_lodo canvas CE = c2_delta_color_weight. "
                         "RAISE this (~3-4) to PROTECT valid cells from being flipped to PAD by the boundary CE "
                         "(the copy cells fold first -> shape_task collapse). Default 1.0 = prior behaviour.")
    ap.add_argument("--lodo-changed-weight", type=float, default=5.0,
                    help="CHANGED (transform) term inside c2_delta_lodo canvas CE = c2_delta_changed_weight. "
                         "Default 5.0 = prior behaviour; the transform cells are the task, keep them up-weighted.")
    ap.add_argument("--lodo-warmup-steps", type=int, default=50, help="steps before ramping final gated-output LODO CE")
    ap.add_argument("--lodo-ramp-steps", type=int, default=50, help="linear ramp length for final gated-output LODO CE")
    ap.add_argument("--train-scope", choices=("delta", "v3-color", "v3-head", "v3-adapter", "quarantine",
                                              "v3-head+quarantine", "v3-adapter+quarantine", "all"),
                    default=None,
                    help=("DEFAULT: v3-head without --unified, FULL stack with --unified. Pass an explicit "
                          "scope only to deliberately override --unified's param set. "
                          "v3-head = train only V3 factored output heads + structure levers, frozen core; "
                          "v3-color = only V3 color_head + relmap input projection, freeze geometry; "
                          "v3-adapter = V3 heads plus C2/relmap/PairDelta adapters, frozen TRM core; "
                          "quarantine = ONLY the PID-quarantined candidate head (quarantine_*) -- z_H not in "
                          "the gradient path, MAIN risk structurally zero; "
                          "v3-head+quarantine = union of v3-head and quarantine (both frozen-core/MAIN-safe); "
                          "v3-adapter+quarantine = v3-adapter plus quarantine, frozen TRM core; "
                          "quarantine params get their own lr (--quarantine-lr) + wd=0 group; "
                          "delta = legacy rule/CTBank/repair modules (most were REMOVED -- near-empty param set); "
                          "all = previous dense optimizer"))
    ap.add_argument("--quarantine-lr", type=float, default=None,
                    help="Dedicated lr for quarantine_* params (wd always 0 for them). Default: --lr. "
                         "Needed under v3-head+quarantine where the warm color_head wants a gentle --lr "
                         "but the quarantine head trains from warm-init at ~3e-3.")
    ap.add_argument("--hint-lr", type=float, default=None,
                    help="Dedicated lr for the zero-init input-side hint adapters (frame_embed, "
                         "rule_hyp_embed, c2_demo_relmap_proj, pairdelta_input_encoder/proj) when "
                         "trained under v3-head+quarantine. Default: --evidence-lr. wd always 0.")
    ap.add_argument("--hint-lr-names", type=str, default="all",
                    help="Comma-separated hint-module name substrings that ride at --hint-lr "
                         "(choices: frame_embed, rule_hyp_embed, c2_demo_relmap_proj, "
                         "pairdelta_input_encoder, delta_rule_input_proj; or 'all'). Hint modules "
                         "NOT listed stay in the scope but train at the core-safe --lr instead -- "
                         "the per-injector damage-control knob after the six-flag MAIN-ON break.")
    ap.add_argument("--no-abort-on-main-drop", action="store_true", help="do not stop when MAIN color/eos/shape degrades")
    # (user request 2026-07-10) the four --abort-main-* threshold flags were REMOVED; the guard now
    # uses these fixed thresholds (the former defaults). --no-abort-on-main-drop still disables it.
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--evidence-lr", type=float, default=None,
                    help="FIX A: dedicated lr for color_evidence_proj (+ color_head_mlp_*), wd=0. The colour "
                         "evidence columns are fresh zero-init tensors that must climb O(1) weights to matter "
                         "against warm-started logits of 5-20; at the core lr 1e-5 they move ~3e-3 in 300 steps "
                         "= inert (measured, all 3 scopes). Default = --struct-lr if set, else 1e-3.")
    ap.add_argument("--struct-lr", type=float, default=None,
                    help="Dedicated lr for structure_relmap_proj (the §15.9 boundary lever), default = --lr. "
                         "The lever is ONE zero-init proj that must climb ~2-5 logits to beat the frozen lm_head's "
                         "(wrong) pad prediction; at the global lr (Adam ~ lr*steps) it moves ~0.01 in 300 steps = "
                         "far too slow frozen-core. Frozen-core is MAIN-safe, so push this high (e.g. 5e-3..1e-2) to "
                         "actually move pad/shape via the lever instead of the core.")
    ap.add_argument("--lr-warmup-steps", type=int, default=20,
                    help="linear LR warmup 0->lr over the first N steps (stabilizes adapters from random init)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-batches", type=int, default=8,
                    help="how many FIXED batches to measure on each log step (stable LODO eval). "
                         "0.4: default raised 4->8 -- 2 batches made LODO swing +-25pct, hiding trends.")
    ap.add_argument("--eval-seed", type=int, default=1234,
                    help="seed for the fixed-eval holdout so the SAME held-out demos are scored every step")
    ap.add_argument("--ckpt", type=str, default=CKPT,
                    help="warm-start checkpoint. DEFAULT is the bare GitHub TRM (c2/structure RANDOM!). "
                         "Pass the trained step_1631 to load trained c2/structure/delta so LODO has a real canvas.")
    ap.add_argument("--unified", action="store_true",
                    help="TRAIN THE WHOLE FLOW from the TRM checkpoint as one chronological pipeline: "
                         "C2 conditioning -> structure(pad/eos/shape) -> delta/LODO -> colour, all training "
                         "together (forces --train-scope all + LODO pad/eos losses ON).")
    ap.add_argument("--zh-check", action="store_true",
                    help="DIAGNOSTIC ONLY (0 training steps, then exit): is the recursed state z_H/grid_z "
                         "actually DEMO-conditioned? Re-runs the REAL recurrence with the per-task demos "
                         "shuffled across the batch and reports how far grid_z + the colour prediction move "
                         "vs an input-shuffle reference. ratio~0 => the recursion ignores the demos (the rule "
                         "never reaches z_H). Run with the SAME model flags you train with so the head matches.")
    ap.add_argument("--zh-trace", action="store_true",
                    help="Run the z_H conditioning report at BASELINE + END of a normal training run "
                         "(before/after), so you can see whether training actually conditions z_H on the demos. "
                         "Composes with --per-family-eval; unlike --zh-check it does NOT early-exit.")
    ap.add_argument("--zh-every", type=int, default=0,
                    help="With --zh-trace, ALSO print the z_H conditioning report every N steps during "
                         "training (0=off). Use 50 so a run stopped early still shows the demo/input relL2 "
                         "TRAJECTORY (is C2 opening?), not just baseline/end.")
    ap.add_argument("--zh-amp", type=str, default="",
                    help="FORCED-SIGNAL sweep (diagnostic, use with --zh-check): comma list of scales, e.g. "
                         "'1,4,10,50'. Re-runs the z_H report with the demo->z_H injections (C2 update, "
                         "PairDelta rule_vec, frame hint) multiplied by K, plus MAIN floor strict at each K. "
                         "Read: relL2 grows with K but flip stays 0 => path works, scale/optimization too weak; "
                         "nothing moves at 50x => path disconnected; MAIN breaks => signal uncontrolled.")
    ap.add_argument("--c2-gate-init", type=float, default=0.0,
                    help="CHANGE 1 (break the C2 cold-start): initial value of C2's gate_patch/gate_global "
                         "(the tanh-gated demos->z_H channel). Default 0.0 => tanh(0)=0 => demos gated OFF. "
                         "Set >0 (try 0.5-1.0) so the channel is OPEN at init: cross_attn gets real gradient "
                         "and demos enter z_H immediately. Verify via --zh-check / --zh-every demo/input relL2 "
                         "(should jump off ~0) while MAIN holds. Locked sweet spot: 0.3.")
    ap.add_argument("--cross-demo-shape", action="store_true",
                    help="CHANGE 3 (molded shape head): enable the supervised cross-demo output-shape head "
                         "(predict the target's output H,W) and UN-DETACH it so the shape CE backprops into "
                         "z_H -> the dense forcing function that makes Change 1's channel carry the demo->shape "
                         "rule (and is the size-change capability). Watch c2_shape_exact/h_acc rise AND "
                         "demo/input relL2 grow, while MAIN holds. ISOLATION: independent of --c2-gate-init / "
                         "the removed repair flags; kept for A/B provenance only.")
    ap.add_argument("--shape-weight", type=float, default=0.3,
                    help="c2_shape_loss_weight for --cross-demo-shape (CE sum over batch; 0.3 = gentle start, "
                         "raise if c2_shape_exact is flat, lower if it perturbs MAIN).")
    ap.add_argument("--shape-debug", action="store_true",
                    help="DIAGNOSTIC (0 steps, exit): run the shape head on the eval batches and print its "
                         "actual H/W predictions vs targets + value/offset histograms, to tell a structural "
                         "bug (collapsed/offset preds) from an untrained head. Use with --cross-demo-shape.")
    ap.add_argument("--c2-relmap", action="store_true", help="Enable relational maps")
    ap.add_argument("--frame-hint", action="store_true",
                    help="Lane B: feed the deterministic solver's verified rearrange-FRAME family as a "
                         "zero-init input-side hint (the rule-hypothesis bus). Dataloader precomputes "
                         "frame_label; the TRM learns the binding. F7-safe (0 at init).")
    ap.add_argument("--rule-hypothesis-hint", action="store_true",
                    help="Default-off A/B: run object_rule_bank.infer_rule_hypotheses inside the model, "
                         "embed the top operation family, and add it to grid_features. F7-safe zero-init; "
                         "evidence only, not a solver or output writer.")
    ap.add_argument("--relmap-outside-grid", action="store_true",
                    help="§15.9.1: place PAD outside the PREDICTED output box (extent_pad_mask; extent from a "
                         "support-verified demo size-rule {identity,constant,ratio} -- same-shape is just the "
                         "identity rule, size-change is constant/ratio). Dedicated [1->3] proj scaled by conf "
                         "(0 -> floor untouched, no task hurt). Zero-init (F7-safe) unless "
                         "--structure-outside-warm-init.")
    ap.add_argument("--structure-outside-warm-init", action="store_true",
                    help="Warm-init the outside_grid PAD row so pad is asserted on the padding from step 0 "
                         "(skips the ~1500-step climb-from-zero). Breaks step-0==floor; frozen-core is MAIN-safe. "
                         "Needs --relmap-outside-grid.")
    ap.add_argument("--structure-outside-warm-init-value", type=float, default=1000.0,
                    help="Half the outside-grid pad-vs-valid swing (pad +V, eos/valid -V => swing 2V). The floor's "
                         "colour-over-pad gap on padding is mean~177/max~620 (scripts/verify_outside_grid_lever.py); "
                         "default V=1000 -> swing 2000 dominates it with margin, and the verifier asserts gap<2V.")
    ap.add_argument("--relmap-eos-grid", action="store_true",
                    help="Thin-L EOS analogue of --relmap-outside-grid: build an EOS boundary mask from the "
                         "support-verified output extent and feed it to a separate structure_eos_proj. "
                         "Default off; does not touch color.")
    ap.add_argument("--structure-eos-warm-init", action="store_true",
                    help="Warm-init the EOS boundary row so EOS is asserted on the predicted thin-L boundary "
                         "from step 0. Needs --relmap-eos-grid.")
    ap.add_argument("--structure-eos-warm-init-value", type=float, default=1000.0,
                    help="Half the eos-vs-pad/valid swing for --structure-eos-warm-init (pad/valid -V, eos +V).")
    ap.add_argument("--c2-relmap-demos", action="store_true",
                    help="section 15.2-A: feed SUPPORT-side relational maps into TestConditionedC2's demo features "
                         "before the cross-attention (separate zero-init proj => no-op at init; F7-safe). "
                         "Needs --c2-relmap. The cross-demo upgrade: A/B this vs map-only.")
    ap.add_argument("--pairdelta-input", action="store_true",
                    help="section 15.2-B: add the PairDeltaEncoder cross-demo rule_vec as a zero-init INPUT-ONLY "
                         "hint (broadcast to grid_features). No output writer. Demoted PairDelta role.")
    ap.add_argument("--v3-clean", action="store_true",
                    help="section 15 V3-CLEAN: route output through the factored structure/color head "
                         "(c2_dual_output_head=True) so the relational-map colour LOOKUP is actually READ at "
                         "output, and FORCE every section 15.3 output-side writer OFF (the factored head is the sole "
                         "writer; no fighting). The loaded yaml pins c2_dual_output_head=false, which silently "
                         "runs legacy lm_head + an inert relmap input residual -- this flag fixes that. PAIR "
                         "WITH --c2-relmap and --train-scope v3-color/v3-head/v3-adapter so the factored heads actually "
                         "train via the base CE; --train-scope delta will NOT train color_head/structure_head.")
    ap.add_argument("--fresh-structure-head", action="store_true",
                    help="section 15.8 A/B: keep the FRESH 3-way structure_head as the dual-output structure "
                         "source instead of deriving PAD/EOS/VALID from lm_head's logsumexp. Default (off) = "
                         "structure-from-lm_head, which reproduces the floor's pad/eos/shape partition exactly and "
                         "fixes the factored LODO pad/shape regression. Pass this only to A/B the old fresh head.")
    args = ap.parse_args()
    # GATE (C' postmortem 2026-07-03): --train-scope's old default 'v3-head' silently OVERRODE
    # --unified's full-stack param set -- the branch chain checks train_scope before args.unified,
    # and argparse cannot tell default from explicit. The "unified" C' run trained 6 frozen-core
    # tensors for 500 steps: z_H bit-identical at every trace, MAIN pinned at 100 under 40%
    # pid-dropout, loss = irreducible noise. None-sentinel: only an EXPLICIT --train-scope
    # may override --unified. (Second occurrence of the V3-1 silent-default pattern.)
    args.train_scope_explicit = args.train_scope is not None
    if args.train_scope is None:
        args.train_scope = "all" if args.unified else "v3-head"
    # NOTE: --unified trains the FULL stack (core + adapters) at lr<=1e-5 with LR warmup + NaN
    # guard -- the naive lr 3e-5 full-stack run diverged to NaN by step ~25.

    raw = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    raw["load_checkpoint"] = args.ckpt
    raw["data_paths"] = [DATASET]
    raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0
    raw["global_batch_size"] = args.batch
    raw["run_name"] = "stage1_local_probe"
    raw["checkpoint_path"] = (str(Path(args.save_checkpoint_dir).resolve())
                              if args.save_checkpoint_dir else "reports/stage1_local_probe")
    arch = raw.setdefault("arch", {})
    # delta branch (same flags as the validated integrated runs)
    # B2: fuse the trained structure_head PAD/EOS into the output logits (0.0 = aux-only = today's
    # behaviour). Turn on (~0.3) ONLY AFTER structure_head has trained, else it injects noise.
    arch["c2_structure_fusion_alpha"] = args.structure_alpha
    # Stage 0 exposure
    # Stage 4: ColorTransitionBank fuse + Bayesian direct colour prior
    # Consolidated colour-repair head (dense, local 3x3, rule-fed). Exclusive with old branch + priors.
    # B1: route the learned cross-demo rule into the colour head (zero-init, off by default).
    # S2 RULE BUS forces this on -- the head must have a rule_proj to consume the fused rule.
    arch["c2_relmap"] = args.c2_relmap
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
    arch["c2_task_palette_feature"] = bool(args.task_palette or args.task_palette_feature)
    arch["c2_task_palette_bias"] = bool(args.task_palette or args.task_palette_bias)
    arch["c2_task_palette_strength"] = float(args.task_palette_strength)
    arch["c2_task_palette_hard"] = bool(args.task_palette_hard)
    arch["c2_rel_where_hint"] = bool(args.rel_where_hint)
    arch["c2_rel_where_topk"] = max(1, int(args.rel_where_topk))
    arch["c2_algo_where_maps"] = bool(args.algo_where_maps)
    arch["c2_pairdelta_intent_hint"] = bool(args.pairdelta_intent_hint)
    arch["c2_transition_hint"] = bool(args.transition_hint)
    arch["c2_value_evidence_v2"] = bool(args.value_evidence_v2)
    arch["c2_value_v2_rich_ctx"] = bool(args.value_v2_rich_ctx)
    # V2-model evidence (trm_fvr_v2 only; harmless pydantic extras on the old model)
    arch["c2_verified_frame_evidence"] = bool(args.verified_frame_evidence)
    arch["c2_analogy_evidence"] = bool(args.analogy_evidence)
    arch["c2_value_ctx_gate"] = bool(args.value_ctx_gate)
    arch["c2_value_v2_backoff"] = bool(args.value_v2_backoff)
    arch["c2_pairdelta_color_evidence"] = bool(args.pairdelta_color_evidence)
    arch["c2_pairdelta_structure_evidence"] = bool(args.pairdelta_structure_evidence)
    arch["c2_pairdelta_bidi_evidence"] = bool(args.pairdelta_bidi_evidence)
    arch["c2_pairdelta_input_conf_gate"] = bool(args.pairdelta_input_conf_gate)
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
        if not args.unified and args.train_scope in ("v3-color", "v3-head", "v3-adapter", "quarantine",
                                                     "v3-head+quarantine", "v3-adapter+quarantine"):
            print("[pid-dropout WARNING] frozen-core scope: the recursion cannot re-route around a missing "
                  "PID, so pid-dropout mostly adds noise here. Intended for --unified (C-prime).")
    if args.color_mlp > 0:
        print(f"[V3 color-mlp] interaction residual ON (dim={args.color_mlp}) | zero-init output | "
              f"step-0 == linear head | adds input-colour x evidence products.")
    if args.transition_hint:
        print("[V3 transition-hint] VALUE binding ON | per-cell demo-consensus P(out|in) over changed "
              "support cells -> 10 zero-init color_head columns | LODO-safe | evidence only, no writer.")
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
    # --- §15 V3-CLEAN SWITCH ------------------------------------------------------------------------
    # CLEANUP (2026-07-01): the 14 legacy inert CLI flags (--color-repair, --rule-bus, --ctbank,
    # --color-prior, --color-feature, --repair-use-rule-vec, --repair-rule-source, --repair-changed-value,
    # --gate-positional, --gate-object, --value-object-cond, --value-copy-safe, --demote-lookup,
    # --copy-relation) plus --color-force/--apply-canvas/--copy-structure were DELETED, along with the 11
    # competing output writers in trm_fvr_c2.py they drove. --palette-constrain is kept.
    if args.v3_clean:
        # The factored head becomes the SOLE output writer; the relmap colour-lookup is read at output.
        # (The 11 competing §15.3 output writers were physically DELETED, so there is no longer anything to
        # force off -- the factored structure/color head is the only writer that exists.)
        arch["c2_dual_output_head"] = True
        arch["c2_structure_fusion_alpha"] = 0.0
        # §15.8: recover the factored pad/shape regression -- derive PAD/EOS/VALID from the trained lm_head
        # (floor-EXACT partition) instead of the fresh ~300-step structure_head. --fresh-structure-head A/Bs back.
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
        if args.train_scope not in ("v3-color", "v3-head", "v3-adapter", "quarantine",
                                    "v3-head+quarantine", "v3-adapter+quarantine", "all") and not args.unified:
            print("[v3-clean WARNING] --train-scope is 'delta': color_head/structure_head are NOT in the "
                  "delta param set, so the factored heads will NOT train. Re-run with --train-scope v3-color "
                  "for colour calibration, v3-head for structure+colour, or v3-adapter after the floor is stable.")
        if args.unified:
            print("[v3-clean WARNING] --unified trains the broad/full stack and can move the TRM core. "
                  "For floor-safe V3 calibration prefer --train-scope v3-head, then v3-adapter.")
        # CONFLICT GUARD: --unified intends a full-stack run, but an explicit frozen-core --train-scope
        # is checked FIRST in the param selector below, so it SILENTLY wins and freezes the core. The
        # output canvas is lm_head(z_H); a frozen core pins LODO pad/shape at the floor (proven flat).
        if args.unified and args.train_scope_explicit and args.train_scope in (
            "v3-color", "v3-head", "v3-adapter", "v3-head+quarantine", "v3-adapter+quarantine"
        ):
            print(f"[v3-clean WARNING] --unified + EXPLICIT --train-scope {args.train_scope}: the frozen-core "
                  f"scope OVERRIDES --unified's full-stack param set, so the TRM core stays FROZEN and "
                  f"LODO pad/shape CANNOT move off the floor. Drop --train-scope to let --unified train the core.")
        # The structure (pad/eos) gradient on the LODO canvas is weighted by --lodo-pad-weight/--lodo-eos-weight
        # (NOT auto-enabled by --unified). At 0.0 the boundary CE is multiplied by zero -> structure_relmap_proj
        # gets no gradient and pad/shape never trains, regardless of scope.
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
    # it is pure waste -- MEASURED 81 s/step on the 400-step Q-run (~33% of every step). The
    # graded colour-choice reads (chgCEc/copyCEc) come from the aux logits and do not need it.
    arch["c2_lodo_force_shuffle"] = bool(args.contrast_weight > 0)
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
    # Contrast starts at 0 and is SCHEDULED with the LODO warmup/ramp in the training loop below.
    # (It was hardcoded 1.0 from step 0: full-strength discriminative pressure on random-init C2
    # drifts z_H through the SHARED forward and decays MAIN strict before the canvas CE ever ramps.)
    loss["c2_delta_contrast_weight"] = 0.0
    loss["c2_delta_contrast_margin"] = 0.5
    loss["c2_delta_contrast_per_row"] = bool(args.contrast_per_row)
    # Stage 0 diagnostic panel + Stage 2 preservation KL. (The Stage-1 NCE/cons and repair
    # gate/colour loss knobs were REMOVED 2026-07-02 -- their model-side producers died in the
    # delta-branch deletion, so the flags were silent no-ops printing NaN metrics.)
    loss["c2_delta_diag"] = True
    loss["c2_delta_preserve_weight"] = args.preserve
    loss["c2_value_v2_aux_weight"] = float(args.value_v2_aux_weight)
    loss["c2_value_v2_aux_changed_weight"] = float(args.value_v2_aux_changed_weight)
    loss["c2_value_v2_aux_copy_weight"] = float(args.value_v2_aux_copy_weight)
    loss["c2_changed_valid_loss_weight"] = args.changed_valid
    # CHANGE 3: cross-demo output-shape head. Enable + UN-DETACH (shape CE backprops into z_H = the
    # forcing function that makes the C2 channel carry the demo->shape rule). Reuses the fully-built
    # c2_shape_head + c2_shape_loss infra (predicts target_height/width; metrics c2_shape_exact/h/w_acc).
    arch["c2_shape_head"] = args.cross_demo_shape        # H/W readout (detach flag removed; canvas writer removed)
    arch["c2_shape_pool"] = "zH_rowcol"                  # H/W-separable pool (mean pool is dimension-blind)
    loss["c2_shape_loss_weight"] = args.shape_weight if args.cross_demo_shape else 0.0

    if args.unified:
        # Full-stack training needs a safe lr: 3e-5 (no warmup) diverged to NaN by step 25.
        # Match the recipe that produced step_1631 (lr 1e-5) + the LR warmup + NaN guard below.
        args.lr = min(args.lr, 1.0e-5)
        # ===== THE UNIFIED FLOW (chronological, components complement) =====
        # Train every layer together from the TRM checkpoint, in dependency order:
        #   L1 CONDITION (C2)      -> trained by train_scope=all (below) via every loss
        #   L2 STRUCTURE (pad/eos) -> structure_head aux losses (c2_pad/valid, inherited from the
        #                             base config) + the LODO canvas pad/eos CE turned ON here
        #   L3 RULE / LODO (delta) -> delta-LODO CE + contrast (+ optional nce/cons/preserve)
        #   L4 COLOUR              -> repair-colour/gate head on the now-correct canvas
        # Structure must be explicit, not implicit: PAD/EOS weights now come from
        # --lodo-pad-weight / --lodo-eos-weight so a command line faithfully defines the run.
        # Keep preservation so correct cells are not clobbered.  Do not force-enable
        # LODO here: --lodo-weight 0 is the calibration mode used to isolate MAIN
        # stability before support-reconstruction pressure is introduced.
        if args.preserve <= 0:
            args.preserve = 1.0
            loss["c2_delta_preserve_weight"] = 1.0
        _pe_on = (args.lodo_pad_weight > 0.0) or (args.lodo_eos_weight > 0.0)
        # This banner must not lie about the param set: an explicit frozen-core scope wins over
        # --unified (the C' postmortem bug was this banner printing "FULL stack" over 6 tensors).
        if args.train_scope_explicit and args.train_scope != "all":
            _u_stack = f"scope OVERRIDDEN by explicit --train-scope {args.train_scope} (TRM core FROZEN)"
        else:
            _u_stack = "FULL stack (TRM core + adapters)"
        print(f"[unified] {_u_stack} @ lr<=1e-5 | "
              f"LODO pad/eos CE {'ON' if _pe_on else 'OFF (pass --lodo-pad-weight/--lodo-eos-weight)'} "
              f"(pad={args.lodo_pad_weight}, eos={args.lodo_eos_weight}) | C2+structure+delta+colour train together")

    config = pretrain.PretrainConfig(**raw)
    loader, meta = pretrain.create_dataloader(
        config, "train", 0, 1, test_set_mode=False,
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
    # REPRODUCIBLE MODEL INIT: the 63 newly-initialised keys (C2 cross_attn, colour/repair heads) are
    # drawn from the global RNG at construction. With c2_gate_init>0 the step-0 baseline DEPENDS on the
    # random cross_attn draw, so without this seed an A/B that adds a module (e.g. rule_proj) shifts the
    # draw and the baselines drift (observed: LODO changed base 33 vs 19 across runs). Seed here so each
    # config's missing-key init is byte-identical run-to-run and only the INTENDED change moves numbers.
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    loss_head, _, _ = pretrain.create_model(config, meta, rank=0, world_size=1)
    loss_head.train()
    device = torch.device("cuda")

    # ---- FIXED LODO eval: the SAME held-out demos scored every log step, so numbers are
    # comparable. The training loop draws a random batch + random holdout each step, which is
    # incomparable noise (step 199 had 2 changed cells). Here we freeze a few batches and seed
    # the holdout, then micro-average the health panel over them. THIS is what we read. ----
    # CROSS-RUN DETERMINISM: model creation consumes a config-dependent amount of RNG, which shifted
    # which batches the loader yields here (COUNTS drifted run-to-run: 446 vs 428 -> different held-out
    # tasks -> router/no-router A/Bs were NOT comparable). Seed right before collecting so the frozen
    # eval set is byte-identical across runs regardless of which model flags are on.
    import numpy as _np
    import random as _random
    torch.manual_seed(args.eval_seed)
    _np.random.seed(args.eval_seed)             # puzzle_dataset samples context demos with NUMPY RNG
    _random.seed(args.eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.eval_seed)
    eval_batches = []
    for _set, _cb, _g in loader:
        if "context_inputs" not in _cb or _cb["context_inputs"].shape[1] < 2:
            continue
        eval_batches.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in _cb.items()})
        if len(eval_batches) >= args.eval_batches:
            break
    print(f"[fixed-eval] {len(eval_batches)} frozen batches, seed={args.eval_seed} "
          f"(same held-out demos scored every {args.log_every} steps)")

    @torch.no_grad()
    def eval_fixed() -> dict:
        """Macro-average the health panel over the frozen eval batches (seeded holdout).
        Metrics are stored as mean*count, so each batch is normalized by its OWN count FIRST
        (recovering the true per-batch value) and then averaged. This avoids >100% inflation when
        the halt-count differs from the metric denominator (e.g. a trained q_head changes halting)."""
        rng = torch.random.get_rng_state()
        crng = torch.cuda.get_rng_state_all()
        agg: dict = {}
        nb = 0
        for i, eb in enumerate(eval_batches):
            torch.manual_seed(args.eval_seed + i)          # deterministic LODO holdout per batch
            torch.cuda.manual_seed_all(args.eval_seed + i)
            with torch.device("cuda"):
                c0 = loss_head.initial_carry(eb)
            _c, _l, m, _, _ = loss_head(carry=c0, batch=eb, return_keys=[])
            cnt = float(m["count"].item()) if "count" in m else 0.0
            denom = cnt if cnt > 0 else 1.0
            nb += 1
            for k, v in m.items():
                if torch.is_tensor(v) and v.numel() == 1:
                    agg[k] = agg.get(k, 0.0) + v.detach().float() / denom   # per-batch true value
        # panel computes g(k)=agg[k]/agg["count"]; set count=nb so g = mean of per-batch values.
        agg["count"] = torch.tensor(float(nb))
        torch.random.set_rng_state(rng)
        torch.cuda.set_rng_state_all(crng)
        return agg

    # ---- PER-FAMILY EVAL (general-not-DSL MICROSCOPE): break the LODO panel by codex family on
    # aug-0 (where puzzle_identifier -> task hash -> family). Tells us whether a lever helps the GENERAL
    # model PER family -- measurement, NOT a per-family solver. Collect aug-0 tasks, tag each by family,
    # regroup into family-homogeneous batches, run the same loss_head eval on each. ----
    family_eval_sets = None
    if args.per_family_eval:
        try:
            import sys as _psys
            _psys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from scripts.oracle_eval import _load_family_map, PROBE_DATASETS
            _aug0 = PROBE_DATASETS["aug0"]
            _id2hash, _hash2fam = _load_family_map(_aug0)
            if _id2hash is None:
                print("[per-family] skipped (no aug-0 identifiers.json / atlas CSV).")
            else:
                import numpy as _npf, random as _rndf
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

    @torch.no_grad()
    def eval_per_family(tag: str) -> None:
        if not family_eval_sets:
            return
        rng = torch.random.get_rng_state(); crng = torch.cuda.get_rng_state_all()
        print(f"\n[PER-FAMILY EVAL @ {tag}]  LODO changed/unchg/color_exact/strict + WHERE/VALUE split")
        print(f"  (judge VALUE binding on conditional_recolor ONLY -- it is the family with proven "
              f"extractable evidence; rearrangement is Lane-B, size_change is structure-solved)")
        order = ["conditional_recolor", "size_change", "rearrangement", "other"]
        for fam in order + [f for f in family_eval_sets if f not in order]:
            bset = family_eval_sets.get(fam)
            if not bset:
                continue
            agg: dict = {}; nb = 0
            for i, eb in enumerate(bset):
                torch.manual_seed(args.eval_seed + i); torch.cuda.manual_seed_all(args.eval_seed + i)
                with torch.device("cuda"):
                    c0 = loss_head.initial_carry(eb)
                _c, _l, m, _, _ = loss_head(carry=c0, batch=eb, return_keys=[])
                cnt = float(m["count"].item()) if "count" in m else 0.0
                denom = cnt if cnt > 0 else 1.0
                nb += 1
                for k, v in m.items():
                    if torch.is_tensor(v) and v.numel() == 1:
                        agg[k] = agg.get(k, 0.0) + v.detach().float() / denom
            gg = lambda k: (float(agg[k]) / nb) if k in agg else float("nan")
            print(f"  {fam:20s} n={len(bset) * config.global_batch_size:3d}  "
                  f"changed={gg('lodo_changed_color_acc_pct'):5.1f} "
                  f"unchg={gg('lodo_unchanged_color_acc_pct'):5.1f} "
                  f"exact={gg('lodo_color_exact_pct'):5.1f} "
                  f"strict={gg('lodo_strict_exact_pct'):5.1f} | "
                  f"where_f1={gg('lodo_where_f1_pct'):5.1f} "
                  f"value_pred={gg('lodo_value_on_pred_changed_pct'):5.1f} "
                  f"chgCEc={gg('lodo_changed_color_ce'):5.2f} "
                  f"copyCEc={gg('lodo_copy_color_ce'):5.2f}")
        torch.random.set_rng_state(rng); torch.cuda.set_rng_state_all(crng)

    @torch.no_grad()
    def eval_selector(tag: str) -> None:
        """VERIFY-AND-SELECT (composition-C, Take #1): per task, score the two MODEL candidates -- the
        lm_head FLOOR vs the factored V3 CANDIDATE -- by LODO reconstruction of the held-out demo,
        commit the best (tie->floor). ONE forward per batch: under --floor-candidate-split the loss
        panel already emits per-example floor AND candidate views from the same aux pass. (The old
        two-pass _force_delta_off toggle was DEAD -- the model no longer reads that attribute, so both
        passes returned IDENTICAL logits and head_chosen was float noise.)"""
        if not args.selector_eval:
            return
        if not args.floor_candidate_split:
            print(f"\n[SELECTOR @ {tag}] SKIPPED: needs --floor-candidate-split (the split exposes the "
                  f"per-example floor vs candidate views the selector compares).")
            return
        rng = torch.random.get_rng_state(); crng = torch.cuda.get_rng_state_all()

        def _per_example(batches):
            ce, cs, fe, fs = [], [], [], []
            for i, eb in enumerate(batches):
                torch.manual_seed(args.eval_seed + i); torch.cuda.manual_seed_all(args.eval_seed + i)
                with torch.device("cuda"):
                    c0 = loss_head.initial_carry(eb)
                _c, _l, m, _, _ = loss_head(carry=c0, batch=eb, return_keys=[])
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

        # Run on ALL frozen eval + EACH per-family set: the floor-safe RECOVERY only SHOWS where the floor
        # solves >0 (eval_batches alone can be floor=0; R26's net-negative lives in the per-family panel).
        sets = [("ALL", eval_batches)]
        if family_eval_sets:
            order = ["conditional_recolor", "size_change", "rearrangement", "other"]
            sets += [(f, family_eval_sets[f]) for f in order if f in family_eval_sets]
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
        torch.random.set_rng_state(rng); torch.cuda.set_rng_state_all(crng)
        print(f"  (sel>=floor MUST be 100 = the R13 kill; the floor-safe RECOVERY shows where floor>0, e.g.")
        print(f"   rearrangement. 1 fold/forward -> SELECT = headroom; deployable non-peeking = multi-fold")
        print(f"   LOOCV verify_and_select_candidates.py, PROVEN 2.7>=1.5, 100%.)")

    # ---- Z_H CONDITIONING CHECK (general-not-DSL diagnostic #1): the colour head reads
    # grid_z = z_H[grid positions]. If the RULE never reaches the recursion, grid_z is just an
    # encoding of the target INPUT and the head can only fall back to the deterministic lookup.
    # Test it directly: run the REAL recurrence (fresh_carry -> _input_embeddings -> _run_recurrence,
    # the model's own LODO path, lines ~1286-1289 of trm_fvr_c2.py) twice -- once with the true
    # per-task demos, once with the demos rolled across the batch so each target gets a DIFFERENT
    # task's demos -- and measure how far grid_z (and the colour prediction) move. Calibrate against
    # an INPUT-shuffle (z_H obviously reacts to the target input). demo/input relL2 ~0 => the
    # recursion ignores the demos. This is MEASUREMENT, not a new lever. ----
    @torch.no_grad()
    def zh_conditioning_report(tag: str = "") -> None:
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
            # Under --floor-candidate-split `logits` IS the frozen floor: stablemax-saturated
            # (measured colour-over-pad gap mean~177), so its argmax flipping ~0 under demo swap is
            # EXPECTED and says nothing about rule uptake. The CANDIDATE head is what LODO trains --
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
            # have near-constant low-norm z_H where cosine is pure noise (that polluted sanity to 0.977
            # and NaN'd "other"); the colour cells are the meaningful, well-conditioned ones.
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
        sets = [("ALL", eval_batches)]
        if family_eval_sets:
            order = ["conditional_recolor", "size_change", "rearrangement", "other"]
            sets += [(f, family_eval_sets[f]) for f in order if f in family_eval_sets]
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
        # numbers. Distinguishes 'path too weak' (relL2 grows with K) from 'path disconnected' (flat)
        # from 'signal uncontrolled' (MAIN strict collapses). Restores scale=1.0 afterwards. ----
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
                res = _eval_set(eval_batches)
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
    def main_recovery_report(tag: str) -> None:
        """FLOOR RECOVERABILITY: MAIN strict with the demo injections ON vs OFF (scale=0). The demo
        paths (C2 update, PairDelta rule_vec, frame hint) are ADDITIVE, so scale=0 removes exactly
        what adapter training drifted. OFF ~100 => the floor is intact and deployable from the
        injection-off forward -- the MAIN-strict abort is then a TRAINING-forward symptom, not a
        deployment loss. OFF < 100 would mean something outside the demo injections moved (real damage)."""
        from models.losses_fvr import IGNORE_LABEL_ID as _IGN
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
            for b in eval_batches:
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
        print(f"\n[floor-recovery @ {tag}] MAIN strict: injections ON={on * 100:.1f}%  "
              f"OFF={off * 100:.1f}%  ({verdict})")
        if was_training:
            loss_head.train()

    @torch.no_grad()
    def pid_null_report(tag: str) -> None:
        """NECESSITY-PRESSURE READ: MAIN strict on the frozen eval batches with the true PID vs the
        blank PID(0). null-pid strict IS demo-only performance -- memorisation cannot help when the
        answer key is withheld, so under --pid-dropout this number RISING is the model reconstructing
        the mapping from the demos (the C-prime success metric; MAIN-with-pid only shows recall).
        The LODO panel is already blank-PID by construction; this measures the MAIN forward the
        same way. Eval batches are never pid-dropped in training, so both columns stay comparable."""
        if not (args.pid_null_eval or args.pid_dropout > 0):
            return
        from models.losses_fvr import IGNORE_LABEL_ID as _IGN
        inner = loss_head.model.inner
        was_training = loss_head.training
        loss_head.eval()
        seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)

        def _strict(null_pid: bool) -> float:
            tot, n = 0.0, 0
            for b in eval_batches:
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
    def shape_debug() -> None:
        import collections
        inner = loss_head.model.inner
        if not getattr(inner.config, "c2_shape_head", False):
            print("[shape-debug] c2_shape_head OFF; pass --cross-demo-shape."); return
        was_training = loss_head.training
        loss_head.eval()
        seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
        tot = hc = wc = 0
        hoff = collections.Counter(); hpv = collections.Counter(); wpv = collections.Counter()
        rows = []
        for eb in eval_batches:
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

    if args.zh_check:
        zh_conditioning_report()
        return
    if args.shape_debug:
        shape_debug()
        return

    # Default: branch-local optimizer. Updating the backbone/lm/structure heads made the
    # repair probe an uncontrolled full-model finetune and could damage PAD/EOS.
    delta_names = (
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
    # factored heads; only then add C2/relmap/PairDelta adapters. `--unified` is kept
    # as the broad legacy experiment, but it can move the TRM core and has repeatedly
    # shown MAIN-shape instability during V3 bring-up.
    v3_color_names = (
        "color_head",
        "color_evidence_proj",
        # ".inner.relmap_proj" (not bare "relmap_proj"): substring matching also caught
        # structure_relmap_proj + c2_demo_relmap_proj, silently training STRUCTURE under a
        # scope labelled "frozen-structure".
        "inner.relmap_proj",
    )
    v3_head_names = (
        "color_head",
        "color_evidence_proj",
        "structure_head",
        "structure_relmap_proj",
        "structure_outside_proj",
        "structure_eos_proj",
        "structure_pairdelta_proj",   # D9 (File #5): exists only under --pairdelta-structure-evidence
    )
    v3_adapter_names = v3_head_names + (
        "frame_embed",
        "rule_hyp_embed",
        "c2.",
        "c2_demo_relmap_proj",
        "pairdelta_input_encoder",
        "delta_rule_input_proj",
        "pid_task_modulator",
        "pid_task_gate",
    )
    adapter_names = delta_names + ("c2.", "structure_head", "pid_task_modulator")
    # Zero-init input-side hint adapters (each exists only when its flag is on, so the name match
    # is naturally gated): --frame-hint, --rule-hypothesis-hint, --c2-relmap-demos, --pairdelta-input.
    # NOTE: --task-palette-feature needs no entry (its 10 evidence cols train through color_head)
    # and --task-palette-bias is param-free logit masking.
    hint_adapter_names = (
        "frame_embed",
        "rule_hyp_embed",
        "c2_demo_relmap_proj",
        "pairdelta_input_encoder",
        "delta_rule_input_proj",
    )
    if args.train_scope == "v3-color":
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and any(k in n for k in v3_color_names)]
        scope_label = "v3-color(frozen-structure,core)"
    elif args.train_scope == "v3-head":
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and any(k in n for k in v3_head_names)]
        scope_label = "v3-head(frozen-core)"
    elif args.train_scope == "v3-adapter":
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and any(k in n for k in v3_adapter_names)]
        scope_label = "v3-adapter(frozen-core)"
    elif args.train_scope == "quarantine":
        # ONLY the PID-quarantined candidate head. z_H, lm_head, structure levers, C2 -- all frozen:
        # the LODO gradient flows through candidate colour -> quarantine_* and stops there, so MAIN
        # cannot move and the abort guard is a formality. Full-strength lodo weight is safe here.
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and "quarantine_" in n]
        scope_label = "quarantine(everything-else-frozen)"
    elif args.train_scope == "v3-head+quarantine":
        # Union scope for the combined A+B+C+D check run: V3 factored heads + structure levers +
        # evidence proj (v3-head set) AND the PID-quarantined candidate head. Both halves are
        # frozen-core, so MAIN safety matches v3-head. lr split: quarantine_* gets --quarantine-lr
        # (wd=0 group below); the rest follow the v3-head recipe (--lr / --struct-lr / --evidence-lr).
        # PLUS the zero-init INPUT-SIDE hint adapters (frame/rule-hyp embeds, demo-relmap proj,
        # pairdelta input encoder) when their flags are on -- previously these existed but were
        # excluded here, so --frame-hint etc. sat at norm=0.0000 forever (measured, 3 runs). They
        # perturb grid_features feeding the FROZEN core, so once nonzero MAIN is no longer
        # byte-invariant -- the MAIN-strict abort guard is load-bearing for this scope now.
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and (any(k in n for k in v3_head_names)
                                          or any(k in n for k in hint_adapter_names)
                                          or "quarantine_" in n)]
        scope_label = "v3-head+quarantine(frozen-core,+hint-adapters)"
    elif args.train_scope == "v3-adapter+quarantine":
        # TRM-focused union scope: train the full V3 adapter stack (C2/demo-relmap/PairDelta/
        # PID modulators + heads/evidence) and the PID-quarantined candidate, while keeping
        # the pretrained recurrence/lm floor frozen. Hot LR is still opt-in per hint module via
        # --hint-lr-names; all non-hot adapters ride at the core-safe --lr.
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and (any(k in n for k in v3_adapter_names)
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
        params = [p for n, p in loss_head.named_parameters()
                  if p.requires_grad and any(k in n for k in delta_names)]
        scope_label = "delta"
    if not params:
        raise SystemExit(
            f"[stage1] train scope '{scope_label}' selected 0 trainable tensors -- nothing would train. "
            f"The 'delta' scope names mostly-REMOVED legacy modules; use --train-scope v3-head (default), "
            f"v3-color, v3-adapter, or --unified.")
    # HARD GATE (C' postmortem): --unified without an explicit scope MUST select the full stack.
    # The bug this catches: a scope default winning the branch chain above -> "unified" run that
    # silently trains a handful of frozen-core tensors (measured: 6 tensors, 500 wasted steps).
    _total_trainable = sum(1 for _n, _p in loss_head.named_parameters() if _p.requires_grad)
    if args.unified and not args.train_scope_explicit and len(params) < int(0.9 * _total_trainable):
        raise SystemExit(
            f"[stage1 GATE] --unified resolved to scope '{scope_label}' with only "
            f"{len(params)}/{_total_trainable} trainable tensors -- a silent frozen-core override. "
            f"Refusing to start; the full-stack param selection is miswired.")
    # §15.9 boundary lever needs its OWN lr: it is a single zero-init proj that must climb ~2-5 logits to beat the
    # frozen lm_head, so at the global lr it trains far too slowly frozen-core (proven: pad flat at 2% for 130 steps).
    # Frozen-core is MAIN-safe, so give structure_relmap_proj a high dedicated lr while color_head stays gentle.
    struct_lr = args.struct_lr if args.struct_lr is not None else args.lr
    # FIX A: the colour EVIDENCE projection (color_evidence_proj + the optional color_head_mlp_*
    # residual) gets its own group, exactly like the structure levers -- fresh zero-init tensors
    # cannot train at the core-safe lr (measured: welded evidence columns moved 6e-6 -> 6.7e-3
    # weight in 300 steps at lr 1e-5 = logit_abs 4e-5 = inert in every run). wd=0: same
    # lever-erosion rule as struct/quarantine.
    evidence_lr = args.evidence_lr if args.evidence_lr is not None else (
        args.struct_lr if args.struct_lr is not None else 1e-3)
    quarantine_lr = args.quarantine_lr if args.quarantine_lr is not None else args.lr
    # Hint adapters are zero-init like the evidence proj: at the core-safe --lr they move ~1e-3
    # weight in 300 steps (the FIX A weld lesson). Default to the evidence lr.
    hint_lr = args.hint_lr if args.hint_lr is not None else evidence_lr
    if (struct_lr != args.lr or evidence_lr != args.lr) and len(params) > 0:
        _struct_ids = {id(p) for n, p in loss_head.named_parameters()
                       if ("structure_relmap_proj" in n
                           or "structure_outside_proj" in n
                           or "structure_eos_proj" in n
                           or "structure_pairdelta_proj" in n)}   # D9: zero-init climber, same lever rules
        _evidence_ids = {id(p) for n, p in loss_head.named_parameters()
                         if ("color_evidence_proj" in n or "color_head_mlp_" in n)}
        # quarantine_* ALWAYS rides its own wd=0 group here: the warm-init +4/+8 columns are held
        # values (lever-erosion rule). Before this split, a quarantine-scope run that ALSO tripped
        # this branch (evidence_lr default != --lr after FIX A) silently put them in the wd=0.01
        # default group -- the third lever-erosion reappearance.
        _quar_ids = {id(p) for n, p in loss_head.named_parameters() if "quarantine_" in n}
        # Hint adapters (frame/rule-hyp embeds, demo-relmap proj, pairdelta encoder): own wd=0
        # group at hint_lr. wd=0 because they are zero-init climbers (lever-erosion rule).
        # --hint-lr-names restricts WHICH hint modules get the hot lr; the rest ride at --lr.
        if args.hint_lr_names.strip().lower() == "all":
            _hot_hint_names = hint_adapter_names
        else:
            _requested = tuple(s.strip() for s in args.hint_lr_names.split(",") if s.strip())
            _unknown = [s for s in _requested if s not in hint_adapter_names]
            if _unknown:
                raise SystemExit(f"[stage1] --hint-lr-names unknown module(s) {_unknown}; "
                                 f"valid: {list(hint_adapter_names)} or 'all'")
            _hot_hint_names = _requested
        _hint_ids = {id(p) for n, p in loss_head.named_parameters()
                     if any(k in n for k in _hot_hint_names)}
        struct_params = [p for p in params if id(p) in _struct_ids]
        evidence_params = [p for p in params if id(p) in _evidence_ids]
        quar_params = [p for p in params if id(p) in _quar_ids]
        hint_params = [p for p in params if id(p) in _hint_ids
                       and id(p) not in _struct_ids and id(p) not in _evidence_ids
                       and id(p) not in _quar_ids]
        other_params = [p for p in params
                        if id(p) not in _struct_ids and id(p) not in _evidence_ids
                        and id(p) not in _quar_ids and id(p) not in _hint_ids]
        # weight_decay=0 on the lever group: AdamW decay is lr*wd*theta, and at struct_lr=1e-2 that is
        # 0.17/step on the +-1000 warm-init levers -- MEASURED leak 1732.04 -> 1717.85 over 90 steps
        # (~40% of the override gone by 3000 steps). Levers must climb from zero or HOLD a warm value;
        # decay toward zero is the opposite of both jobs.
        groups = [{"params": other_params, "lr": args.lr}]
        if struct_params:
            groups.append({"params": struct_params, "lr": struct_lr, "weight_decay": 0.0})
        if evidence_params:
            groups.append({"params": evidence_params, "lr": evidence_lr, "weight_decay": 0.0})
        if quar_params:
            groups.append({"params": quar_params, "lr": quarantine_lr, "weight_decay": 0.0})
        if hint_params:
            groups.append({"params": hint_params, "lr": hint_lr, "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
        print(f"[struct-lr] structure levers ({len(struct_params)} tensors) lr={struct_lr}, wd=0; "
              f"[evidence-lr] color_evidence_proj/mlp ({len(evidence_params)} tensors) lr={evidence_lr}, wd=0; "
              f"[quarantine-lr] quarantine_* ({len(quar_params)} tensors) lr={quarantine_lr}, wd=0; "
              f"[hint-lr] hint adapters ({len(hint_params)} tensors: {','.join(_hot_hint_names) if hint_params else 'none'}) "
              f"lr={hint_lr}, wd=0; rest at lr={args.lr}.")
    else:
        # Quarantine scope: wd=0. The head's warm-init columns (copy +4 / consensus +8, lin_norm
        # 28.28) are HELD values exactly like the extent levers; AdamW decay at lr 1e-2 erodes
        # them lr*wd*theta per step (measured on the first Q-run: lin_norm 28.28 -> 26.86 by
        # step 120, part decay part gradient) -- the third appearance of the lever-erosion bug.
        _wd = 0.0 if args.train_scope in ("quarantine", "v3-head+quarantine", "v3-adapter+quarantine") else 0.01
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=_wd)
        if _wd == 0.0:
            print("[quarantine] optimizer wd=0 (warm-init columns are held values; decay is the lever-erosion bug).")
    print(f"[stage1-probe] steps={args.steps} lr={args.lr} "
          f"batch={args.batch} scope={scope_label} lodo_target={args.lodo_weight} "
          f"lodo_pad={args.lodo_pad_weight} lodo_eos={args.lodo_eos_weight} "
          f"lodo_copy={args.lodo_copy_weight} lodo_changed={args.lodo_changed_weight} "
          f"lodo_warmup={args.lodo_warmup_steps} lodo_ramp={args.lodo_ramp_steps} | "
          f"trainable tensors={len(params)}")

    def scheduled_lodo_weight(step: int) -> float:
        if args.lodo_weight <= 0:
            return 0.0
        if step < args.lodo_warmup_steps:
            return 0.0
        if args.lodo_ramp_steps <= 0:
            return float(args.lodo_weight)
        frac = min(1.0, (step - args.lodo_warmup_steps + 1) / max(1, args.lodo_ramp_steps))
        return float(args.lodo_weight) * frac

    def panel(step: int, loss_val: float, m: dict, lodo_w: float) -> bool:
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
        # chgCE/gap are the GRADED trend (visible long before color_exact moves): chgCE = mean CE on
        # true changed cells (must FALL); gap = shuffle - real changed CE (>0 and rising = the head is
        # using the demos, not a global prior). color_exact is a final exam, not a progress meter.
        if "lodo_where_f1_pct" in m:
            # chgCEc/copyCEc = COLOUR-CHOICE CE (log_softmax over the 10 colour channels -> the
            # candidate's floor anchor cancels exactly; the full-vocab chgCE was dominated by the
            # frozen floor at ~43 nats and never moved). lodo_raw = the actual training objective
            # on the fixed eval. Read: chgCEc FALLING with copyCEc flat = binding is being learned;
            # copyCEc RISING = calibration eroding the copy basin (the changed:copy weight trade).
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
                    # FIX A: V2 columns live in the dedicated color_evidence_proj (own lr group),
                    # no longer inside color_head -- offset no longer includes hidden_size.
                    _ch = getattr(_inner, "color_evidence_proj", None)
                    if _ch is not None and getattr(args, "value_evidence_v2", False):
                        from models.recursive_reasoning.object_bank import REL_MAP_CHANNELS
                        from models.recursive_reasoning.trm_fvr_c2 import VALUE_EVIDENCE_V2_DIM
                        _cfg = getattr(_inner, "config", None)
                        _off = 0
                        _off += REL_MAP_CHANNELS if bool(getattr(_cfg, "c2_relmap", False)) else 0
                        _off += 10 if bool(getattr(_cfg, "c2_task_palette_feature", False)) else 0
                        _off += (max(1, int(getattr(_cfg, "c2_rel_where_topk", 1)))
                                 if bool(getattr(_cfg, "c2_rel_where_hint", False)) else 0)
                        _off += 1 if bool(getattr(_cfg, "c2_pairdelta_intent_hint", False)) else 0
                        _off += 10 if bool(getattr(_cfg, "c2_transition_hint", False)) else 0
                        _w = _ch.weight[:, _off:_off + VALUE_EVIDENCE_V2_DIM].detach().float()
                        # color_evidence_proj.weight.grad must be non-zero here; else V2AUX cannot train the tail.
                        _cg = _ch.weight.grad
                        _gw = (_cg[:, _off:_off + VALUE_EVIDENCE_V2_DIM].detach().float()
                               if _cg is not None else None)
                        _gn = float(_gw.norm()) if _gw is not None else float("nan")
                        _gm = float(_gw.abs().max()) if _gw is not None else float("nan")
                        print(f"  V2TAIL: off={_off} w_norm={float(_w.norm()):.4e} "
                              f"grad_norm={_gn:.4e} grad_max={_gm:.4e} "
                              f"logit_std={g('c2_value_v2_aux_logit_std'):.4e} "
                              f"logit_abs={g('c2_value_v2_aux_logit_abs_mean'):.4e}")
        # LEVER growth: is structure_relmap_proj actually moving? Static norms across steps => the lever is
        # not learning (feature missing or gradient dead); rising norms => it is training (whether it HELPS
        # is pad/shape_task's job). outside=the §15.9.1 outside_grid pad-row weight when the channel is on.
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
                  f"cand_chosen={pct_s('lodo_candidate_chosen_pct')}")
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
        if "c2_rel_where_hint_mean" in m or "c2_pairdelta_conditional_score" in m:
            print(f"  PDINTENT: conditional={g('c2_pairdelta_conditional_score'):.3f} "
                  f"global={g('c2_pairdelta_global_score'):.3f} "
                  f"shape_pres={g('c2_pairdelta_shape_preserved'):.3f} "
                  f"changed_rate={g('c2_pairdelta_changed_rate'):.3f}")
        # PDEV: the D-block evidence coverage line. The 500-step D8/D9 A/B ran BLIND -- none of the
        # pd/verified-frame/analogy coverage scalars were printed, so "did the evidence even fire on
        # the frozen batches" was unanswerable from the log. One line fixes that.
        if any(k in m for k in ("c2_pd_color_consensus_mass", "c2_pd_struct_conf",
                                "c2_pd_bidi_invertibility", "c2_verified_frame_coverage")):
            print(f"  PDEV  : consensus={g('c2_pd_color_consensus_mass'):.3f} "
                  f"minchg={g('c2_pd_color_min_change'):.3f} "
                  f"pos={g('c2_pd_color_pos_prior'):.3f} "
                  f"struct_conf={g('c2_pd_struct_conf'):.3f} "
                  f"invert={g('c2_pd_bidi_invertibility'):.3f} "
                  f"del={g('c2_pd_bidi_del_rate'):.3f} "
                  f"vframe_cov={g('c2_verified_frame_coverage'):.3f} "
                  f"analogy_cov={g('c2_analogy_coverage'):.3f} "
                  f"in_conf={g('c2_pairdelta_input_conf'):.3f}")
        if "c2_rule_hyp_norm" in m:
            print(f"  RULEHYP: norm={g('c2_rule_hyp_norm'):.4e} "
                  f"nonzero={g('c2_rule_hyp_nonzero_frac'):.3f}")
        # HINTS attribution line: ALL broadcast-injector norms side by side. The six-flag run broke
        # MAIN-ON with only RULEHYP's norm printed -- frame/pairdelta grew invisibly. Any injector
        # norm >>0 here while MAIN-ON drops names the culprit without an ablation run.
        if "c2_frame_hint_norm" in m or "c2_pairdelta_input_norm" in m:
            print(f"  HINTS : frame_norm={g('c2_frame_hint_norm'):.4e} "
                  f"frame_nonzero={g('c2_frame_hint_nonzero_frac'):.3f} "
                  f"pairdelta_norm={g('c2_pairdelta_input_norm'):.4e}")
        if args.cross_demo_shape:
            print(f"  SHAPE %: exact={g('c2_shape_exact') * 100:.1f} h_acc={g('c2_shape_h_acc') * 100:.1f} "
                  f"w_acc={g('c2_shape_w_acc') * 100:.1f} loss={g('c2_shape_loss'):.2f}")
        print(f"  BLOCK %: shape={g('lodo_block_shape_pct'):.0f} "
              f"color={g('lodo_block_color_pct'):.0f} pad={g('lodo_block_pad_pct'):.0f} "
              f"eos={g('lodo_block_eos_pct'):.0f}")
        print(f"  COUNTS : lodo_color={g('n_lodo_color_cells'):.0f} changed={g('n_lodo_changed_cells'):.0f} "
              f"unchanged={g('n_lodo_unchanged_cells'):.0f} pad={g('n_lodo_pad_cells'):.0f} "
              f"eos={g('n_lodo_eos_cells'):.0f} rows={g('n_lodo_rows'):.0f}")
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

    step = 0
    nan_count = 0
    eval_per_family("baseline (step 0, pre-train)")    # general-not-DSL microscope (no-op unless flag)
    eval_selector("baseline (step 0, pre-train)")      # floor-safe verify-and-select (no-op unless --selector-eval)
    pid_null_report("baseline (step 0, pre-train)")    # demo-only MAIN strict (no-op unless pid flags)
    if args.zh_trace:
        zh_conditioning_report("baseline (step 0, pre-train)")
    # Phase 1: dedicated generator so colour-perm doesn't perturb the fixed-eval RNG (eval_fixed
    # saves/restores its own state). TRAINING batches only -> the fixed-eval stays canonical/comparable.
    color_perm_gen = torch.Generator().manual_seed(args.eval_seed + 777) if args.color_perm else None
    # C-prime necessity pressure: dedicated CPU generator, same isolation rationale as colour-perm.
    pid_drop_gen = torch.Generator().manual_seed(args.eval_seed + 888) if args.pid_dropout > 0 else None
    while step < args.steps:
        for _set, cb, _g in loader:
            if "context_inputs" not in cb or cb["context_inputs"].shape[1] < 2:
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
            lodo_w = scheduled_lodo_weight(step)
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
            with torch.device("cuda"):
                carry = loss_head.initial_carry(batch)
            carry, loss_val, metrics, _, _ = loss_head(carry=carry, batch=batch, return_keys=[])
            # NaN GUARD: a single non-finite loss must NOT corrupt the weights -> skip the step.
            if not torch.isfinite(loss_val):
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
            # GRAD GUARD: the loss can be FINITE while the backward produces a non-finite GRADIENT
            # (e.g. 0*inf when a near-one-hot head distribution meets a contradicting target the
            # instant the gate opens). clip_grad_norm_ does NOT sanitise this -- a NaN/inf total
            # norm scales every grad to NaN, and opt.step() then corrupts ALL weights, so the
            # loss-side guard above only catches it on the NEXT step, after the damage. Skip the
            # step whenever the PRE-clip grad norm is non-finite -> weights stay finite, run continues.
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(grad_norm):
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
            if step % args.log_every == 0 or step == args.steps - 1:
                # Read the FIXED eval (comparable), not this step's random training batch.
                eval_m = eval_fixed()
                loss_scalar = float(loss_val.item())
                if not panel(step, loss_scalar, eval_m, lodo_w):
                    pid_null_report(f"abort (step {step})")
                    main_recovery_report(f"abort (step {step})")
                    return
            if args.zh_trace and args.zh_every > 0 and step > 0 and step % args.zh_every == 0:
                zh_conditioning_report(f"step {step}")   # TRAJECTORY: is C2 opening yet?
            step += 1
            if step >= args.steps:
                break

    eval_per_family(f"end (step {step}, trained)")     # before/after vs the baseline call above
    eval_selector(f"end (step {step}, trained)")       # floor-safe verify-and-select (no-op unless --selector-eval)
    pid_null_report(f"end (step {step}, trained)")     # did demo-only MAIN strict move?
    if args.zh_trace:
        zh_conditioning_report(f"end (step {step}, trained)")
    main_recovery_report(f"end (step {step})")         # is the floor deployable regardless of ON-drift?

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
    if args.save_checkpoint_dir:
        out_dir = Path(args.save_checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / f"step_{step}"
        torch.save(loss_head.state_dict(), ckpt_path)
        config_path = out_dir / "all_config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        print(f"[checkpoint] saved final model -> {ckpt_path}")
        print(f"[checkpoint] saved resolved config -> {config_path}")


if __name__ == "__main__":
    main()
