# START HERE — TRM_C2 orientation for a fresh agent

You are looking at a modified **TinyRecursiveModels (TRM)** fork aimed at **ARC-AGI-1**. This file is
the single entry point: read it top-to-bottom and you will have the whole picture — what the system is,
what works, what is measured-dead (so you don't re-run dead ends), the invariants you must not break,
the current plan, and a pointer map to every active doc/plan/memory. Written 2026-07-05.

Owner: Pushkar (ARC-AGI researcher). Style in this repo's sessions: direct, evidence-first, name exact
numbers/files, no praise-padding. Discipline that governs everything below: **build the check before
the feature; verify, don't assume; gate-off, don't delete; report before large runs.**

---

## 0. THE ONE MENTAL MODEL (read this first)

There are **TWO LANES** that produce ARC answers. Almost every mistake in this project comes from
confusing them.

1. **The MODEL lane (neural).** A TRM recurrent reasoner + a cross-demo conditioning module **C2** +
   a factored colour/structure output head. Trained. Runs a forward pass on a task's support demos
   and target input. Lives in `models/recursive_reasoning/trm_fvr_c2.py`.

2. **The OFFLINE SELECTOR lane (symbolic / verified).** A read-only harness that proposes
   deterministic candidate solutions per task, LODO-verifies them on the demos, and selects the ones
   that provably fit (never below the floor). This is where the **actual banked exact-solve yield**
   lives. Lives in `scripts/verify_and_select_candidates.py` + `object_rule_bank.py`.

**Load-bearing fact:** the composed **offline selector = 23/400 (5.8%)** held-out ARC 2-attempt. The
model lane's exact-solve contribution beyond memorization is ~0 so far. When a lever is "measured
dead," it means dead **in the model lane**; the selector lane is the ROI path. Do not wire a model
feature whose offline oracle ceiling is at the floor.

North star (memory `general-not-dsl`): general rule-extraction; a DSL of GENERAL primitives is a
permitted prerequisite, with generality RELOCATED to a LEARNED cross-demo selector (the composer),
never per-family dispatch.

---

## 1. THE GOOD (banked, working, trust these)

- **Offline verified selection = 23/400 (5.8%)** via `--compose-test --no-committed` (all deterministic
  families + Lane A relocate, floor-safe LODO ranking). The selector is PROVEN; selection never scores
  below the floor. (memory `offline-candidate-harness`.)
- **Lane A rearrangement/relocate = 8/20 exact** banked (frame×key propose→verify→select). Honest
  deterministic ceiling for that family. (memory `rearrangement-relocate-solver`.)
- **Structure-from-lm_head (§15.8)**: PAD/EOS/VALID derived from `lm_head` logsumexp, floor-EXACT
  (verified 1.9e-6). This is the load-bearing guarantee that the factored head inherits pad/eos/shape.
  `--v3-clean` default on. (memory `v3-run-wiring`, doc `COLOR_FLOW.md`.)
- **Deterministic relational substrate** (`object_bank.relational_maps`, 13 channels) + **object/
  analogy/rule inference** (`object_rule_bank.py`): connected components, distance transforms,
  per-label bboxes, object slots, analogy recolour, frame/binding hypotheses. Self-tested, cheap,
  correct. This is the reusable compute you build new WHERE-masks on.
- **F7 / zero-init discipline works**: every added feature is zero-init so step-0 forward is
  byte-identical and default (flags off) is unchanged. 26+ integration tests
  (`scripts/test_relmap_integration.py`) enforce it.
- **Warm-init** (color_head from lm_head) banked +8pp on the colour lane.

---

## 2. THE DEAD / KILL-GATED (may be verdict to strong , cus of our incompitance we may have over said that its dead or kill , but it isnt and should be rechecked re analyisi and improve and rerun )

Each was measured, not guessed. Receipts in the named memory/plan.

- **Colour VALUE lane (per-cell/per-object)** — multi-target `conditional_recolor` (52/75 tasks) caps
  at **~31–34% val_acc, FLOOR exact**, five independent oracle-WHERE probes converging: exact-bucket
  34.4, soft-NN 31.4, +container no-help, object identity/ordinal, object relational. The disambiguator
  is not a per-cell feature. (memory `offline-candidate-harness`; plan `PLAN_where_value_binding.md`.)
- **Correspondence-to-demo (object identity/ordinal/relational) = NO-GO** (`--correspondence-probe`,
  2026-07-05): fg_cover 42% (rest is background fill), best fg_val ~34, gExact ≤3.8. The 5th converging
  negative. (memory `offline-candidate-harness`.)
- **WHERE×VALUE object ceiling = KILL**: conditional_recolor needs_object = **0/75**
  (`--where-value-ceiling`). Root cause: the ambiguous source is colour-0 = background FILL, not object
  recolor. (plan `PLAN_where_value_binding.md`.)
- **Rule-hypothesis TOKEN bus = DOA** (`--rule-probe` ~3/20 actionable; core already infers family
  from histogram). Now wired F7-safe default-OFF as `c2_rule_hypothesis_hint` for A/B only — NOT a
  solve. (plan `PLAN_where_value_binding.md` Appendix Q; memory `v3-run-wiring`.)
- **C′ full-stack necessity-pressure run = 4th negative**: the unified core does not convert added
  evidence into exact solves under pressure. **Adapter-pressure DEAD (3× Goodhart), Q-lane CLOSED,
  z_H-only DEAD.** (memory `v3-run-wiring`, `general-not-dsl`.)
- **conditional_recolor is dead to deterministic colour rules from every angle** (floor, object-WHERE,
  cell-context, relations, global-D4 symmetry, gradient). It is a documented boundary-of-expressibility,
  not a "try harder." (memory `where-value` notes in `offline-candidate-harness`.)

**Why this matters:** the recurring failure mode here is re-attacking colour VALUE with a "richer key."
It is the trap. The open frontier is COMPUTE (flood-fill / ray / bbox WHERE masks) + LOCAL value, and
cross-demo analogy in C2 (genuinely open, four runs couldn't crack it).

---

## 3. THE TRAPS / INVARIANTS (break one and your run is silently wrong)

- **Silent-default run kills (verify EVERY unified run's first minute).** Two historical disasters:
  (1) `c2_dual_output_head` must be TRUE or the V3 factored head is a no-op; (2) `--train-scope`
  default once selected `v3-head` and silently froze the core. Fixed with None-sentinel + hard
  start-gate. Confirm `scope=unified-full(core+adapters)` + trainable tensors in the HUNDREDS. (memory
  `v3-run-wiring` — READ before any run.)
- **MAIN 100% = MEMORIZATION, not skill.** Only a bare-TRM checkpoint truly survives; c2 is
  random+untrained in some paths. Judge generalization by LODO held-out colour-exact, never MAIN.
  (memory `lodo-structure-foundation` — READ FIRST for anything training-related.)
- **Token convention:** PAD=0, EOS=1, **colour = colour_value + 2** (`COLOR_OFFSET=2`). Colour 0 is
  token 2 (a REAL colour, not pad). Off-by-two here corrupts every mask.
- **F7-safety:** any new model feature must be zero-init so step-0 is byte-identical; default path
  (flag off) must be unchanged. Enforced by `test_relmap_integration.py`.
- **LODO-safety:** support-derived features must read the held-out-correct context
  (`_active_context_inputs/_outputs/_mask`, or the params passed by `_run_aux_logits`), never leak the
  target demo.
- 
- **Env split:** the user runs >200-step training in his own terminal. An assistant supplies commands
  and only runs read-only offline probes / 0-step diagnostics. Never launch a full training run.

---

## 4. THE CURRENT PLAN (forward frontier)

**COMPUTE-the-WHERE, LEARN-the-VALUE** — plan `reports/PLAN_colour_taxonomy_where_compute.md`.
The colour families (§1 Enclosure 25, §2 Adjacency 66, §3 Projection/BBox 151) each need multi-step
ops whose first steps are COMPUTATIONS (flood-fill, ray cast, bbox) the 13 relmap channels don't run.
Precompute those as WHERE masks (checkpoint-safe carrier tensor + zero-init proj), gate each family by
an offline ceiling probe, deliver in BOTH lanes (offline candidate = banked exact; model channel =
compose upside). Gating first step: the **enclosure ceiling probe** (`--enclosure-ceiling`, read-only).

Longer-horizon open bet: make **cross-demo analogy work in C2** (correspondence in latent space).
Genuinely unsolved; treat as research, not a fix.

---

## 5. THE MAP — every active artifact (what / why / read-order)

### 5a. CODE — entry points (all under `TinyRecursiveModels/`)
| File | What / why | Status |
|---|---|---|
| `models/recursive_reasoning/trm_fvr_c2.py` (137KB) | THE model: TRM recurrence + C2 + factored head + all `c2_*` feature flags. Config `FVR_C2_Config`. | active, hot |
| `models/recursive_reasoning/object_bank.py` (40KB) | Deterministic per-cell substrate: `relational_maps` (13ch), `cell_conditioning_signature`, connected_components, distance_transform. | active |
| `models/recursive_reasoning/object_rule_bank.py` (79KB) | Symbolic engine: object slots, analogy recolour, `infer_rule_hypotheses`, frame/relocate solvers (Lane A). | active |
| `scripts/verify_and_select_candidates.py` (128KB) | THE offline harness: candidate families, LODO verify/select, all `--*-probe`/`--*-ceiling`/`--compose-test`. | active, hot |
| `puzzle_dataset.py` | Dataloader; `_collate_batch` precomputes `rel_maps`/`frame_label` (where new masks go). | active |
| `pretrain.py` | Training entry / config plumbing. | active |
| `scripts/run_stage1_local.py` | Unified-run launcher; the train-scope gate lives here. | active |
| `models/losses_fvr.py` | Loss head (colour/structure CE, aux). | active |
| `models/recursive_reasoning/color_repair_head.py` | ColorRepairHead (colour mechanisms compose here). | active |
| `scripts/test_relmap_integration.py` | 27 F7/LODO integration tests — run after any model edit. | active |
| `scripts/lodo_refiner.py` | LODO recipe refinement for the selector. | support |

### 5b. DOCS — top-level (all under `TinyRecursiveModels/`)
| Doc | What / why | Read when |
|---|---|---|
| `ARCHITECTURE_V2.md` (1300 lines) | **CANONICAL living doc.** Part 1 = frozen design intent (rule→colour integrated system); Part 2 = living log. The integration target. | deep dive |
| `ARCHITECTURE.md` (253 lines) | File-by-file map + data flow (baseline 2026-06-01). Fastest way to learn what each file does. | onboarding |
| `RUN_COMMANDS.md` | Every experiment command (PowerShell, from the repo dir). | before running |
| `COLOR_FLOW.md` | The ONE chronological colour pipeline order (so mechanisms compose, not fight). | colour work |
| `EXECUTION_FAILURE_LEDGER.md` | Every documented execution failure + whether the fix was validated. Read to avoid repeating them. | before big changes |
| `CHOLLET_IDEOLOGY.md` | North-star reference (Chollet/Ndea vision). Companion to `general-not-dsl`. | strategy |

### 5c. REPORTS (`TinyRecursiveModels/reports/`)
| Report | What / why |
|---|---|
| `PLAN_colour_taxonomy_where_compute.md` | **The current forward plan** (compute-WHERE/learn-VALUE, per-fix spec). |
| `PLAN_where_value_binding.md` | Colour VALUE lane: Phase-0 KILL results + Appendix Q (rule-hypothesis DOA + wiring). |
| `arc_task_taxonomy_report.md` | 800-task taxonomy + quality verification (the classification backbone). |
| `arc_conditional_recolor_subclassification.md` | The conditional_recolor sub-breakdown. |
| `chronological_c2_bothfail_conversion_summary.md` | C2 both-fail→conversion history (older, backstory). |

### 5d. PLANS (out-of-tree, `C:\Users\PUSHKAR\.claude\plans\`)
| Plan | What / why |
|---|---|
| `clean_plan-rewrite.md` | **Living handoff log** (Codex + assistant both edit). Most recent implementation logs: correspondence probe, rule-hypothesis wiring, Fix1/Fix2 value probes. Read for the latest chronological state. |

### 5e. MEMORY (out-of-tree, `C:\Users\PUSHKAR\.claude\projects\D--trm-c2\memory\`)
Index = `MEMORY.md`. Highest-value files, in read order:
1. `lodo-structure-foundation.md` — **READ FIRST.** why pad/eos/shape were broken; MAIN=memorization.
2. `general-not-dsl.md` — the north star (general rule-extraction; learned cross-demo selection).
3. `v3-run-wiring.md` — **the run GATE.** silent-default kills; what's dead (adapter/Q/z_H); C′ verdict.
4. `offline-candidate-harness.md` — the selector, all probe verdicts, 23/400, the colour-lane deaths.
5. `rearrangement-relocate-solver.md` — Lane A (8/20 relocate).
6. `color-lodo-pipeline.md` — the colour LODO target + ~37% extraction ceiling.
7. `architecture-v2-rule-bus.md` — the rule-bus integration target (keep/merge/add/shut).
8. `dataset-landscape.md` — the 3 datasets (aug-1000 train / aug-0 probe / full400 held-out).
9. `unified-training-dynamics.md` — lr/warmup/NaN/OOM/collapse knobs for --unified.
10. `codex-integration-audit.md`, `arc-lang-verified-selector.md`, `phase0-foundation.md`,
    `aaai-paper-writeup.md` — supporting.

---

## 6. HOW TO RUN (safe vs not)

- **Offline probes (SAFE, run freely):** `python scripts/verify_and_select_candidates.py --<probe>`
  e.g. `--compose-test --no-committed`, `--where-value-ceiling`, `--correspondence-probe`,
  `--enclosure-ceiling` (planned). Read-only, no model, no training.
- **Integration tests:** `python scripts/test_relmap_integration.py` (needs the full env:
  torch+einops+coolname). Run after any model edit.
- **Training (USER ONLY):** unified runs via `scripts/run_stage1_local.py` — assistant supplies the
  command, user runs it and pastes the log. Verify the start-gate banner every time (see §3).
- **Env note:** local shells here have a partial torch env; the full model/train env is the user's
  terminal. Probes needing only torch (object_bank / object_rule_bank / most of the candidate harness)
  do run locally.

---

## 7. GLOSSARY (the jargon you'll hit)

- **C2** — cross-demo conditioning module (test-conditioned attention over support demos).
- **z_H / z_L** — TRM high/low latent recurrent states.
- **LODO** — Leave-One-Demo-Out: hold out one train demo as the target to measure generalization.
- **floor** — the trivial per-colour modal map baseline; selection must never score below it.
- **[pid-null] / pid-dropout** — puzzle-id ablation; rising [pid-null] = the model uses evidence, not
  the memorized puzzle id. THE metric for C′-style runs.
- **F7** — the zero-init invariant (step-0 byte-identical; default path unchanged).
- **WHERE × VALUE** — WHERE = which cells change; VALUE = what colour they become. The colour problem
  factorizes into these; VALUE (for multi-target) is the measured-dead part.
- **compose-test** — the offline composed-selector evaluation (`--compose-test`), yields 23/400.
- **Lane A / Lane B** — A = offline relocate/rearrange selector (banked); B = per-cell kinematic
  channels into the model (proposed, gated).
- **v3-clean / dual_output_head / structure-from-lmhead** — the factored output head that inherits
  pad/eos/shape from lm_head (floor-exact).
- **multi-target** — one source colour maps to >1 output colour across a task (needs conditioning;
  the hard 52/75 of conditional_recolor).

---

*If you change what's true here, update this file, `clean_plan-rewrite.md`, and the relevant memory.
This doc is the map; keep it honest or it becomes a trap.*
