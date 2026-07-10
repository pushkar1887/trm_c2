"""Step-0 BYTE-IDENTITY harness for the trm_fvr_c2 -> V2 rewrite (File #4).

The oracle for this file is the LIVE MODEL: same config + same WEIGHTS + same batch + same seed must
produce identical logits / q_logits / every extras tensor, old-file vs V2. This is stronger than the
function-equality oracle of files #1/#2 -- the whole forward graph (warm-started heads included) must
agree tensor-for-tensor.

Method (weight-robust): build model A, snapshot its state_dict, build model B (any init), LOAD A's
state_dict into B, run BOTH on one fixed batch in eval() mode, assert torch.equal on every output.
Loading the shared state_dict isolates a LOGIC difference from an init-order difference -- so when V2
lands, restructured code with identical logic + identical param names passes, and a real logic change
fails loudly.

Block 0 (this file, before any V2 exists): run OLD-vs-OLD to prove the harness itself is sound --
(1) identical weights => identical forward (the positive gate is real), and
(2) a one-weight perturbation IS detected (the gate is not vacuously passing).

Later blocks pass build_b=<the V2 module> to compare old-vs-new.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_COMPILE", "1")          # local env has no triton (mirrors the test script)
os.environ.setdefault("WANDB_MODE", "disabled")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# trm_fvr_c2 top-imports `models.visual_arc_renderer.ARCTokenSpec`, whose source was removed (only a
# stale .pyc remains). ARCTokenSpec is instantiated ONLY under c2_visual_rule_adapter=True -- a
# documented-DEAD lane, off in every identity config -- so a stub that satisfies the import (never the
# call) is sufficient and keeps the harness free of the missing dependency. V2 inherits the same stub.
import types as _types
if "models.visual_arc_renderer" not in sys.modules:
    try:                                          # prefer the real module if the environment has it
        import models.visual_arc_renderer  # noqa: F401
    except ModuleNotFoundError:
        _stub = _types.ModuleType("models.visual_arc_renderer")

        class ARCTokenSpec:                       # placeholder: visual adapter is default-off everywhere here
            def __init__(self, *a, **k):
                raise RuntimeError("visual_arc_renderer stub: c2_visual_rule_adapter must stay off")

        _stub.ARCTokenSpec = ARCTokenSpec
        sys.modules["models.visual_arc_renderer"] = _stub

import torch

from models.recursive_reasoning.object_bank import REL_MAP_CHANNELS, relational_maps


# --- config variants the gate must cover (plan §7.1) -------------------------------------------------
def _base(seq_len: int = 36, hidden: int = 64) -> dict:
    return dict(
        batch_size=2, seq_len=seq_len, puzzle_emb_ndim=hidden, num_puzzle_identifiers=4, vocab_size=12,
        H_cycles=1, L_cycles=1, H_layers=1, L_layers=1, hidden_size=hidden, expansion=2, num_heads=4,
        pos_encodings="rope", halt_max_steps=1, halt_exploration_prob=0.0, forward_dtype="float32",
        c2_enabled=True, c2_num_context=3,
    )


CONFIGS = {
    # (a) V3-clean factored dual head + relmap evidence (the production shape)
    "a_dual_relmap": dict(c2_dual_output_head=True, c2_relmap=True),
    # (b) legacy lm_head writer (dual off) -- the other output regime
    "b_legacy_lmhead": dict(c2_dual_output_head=False, c2_relmap=False),
    # (c) quarantine + floor/candidate split + value-evidence-v2 (the densest evidence path)
    "c_quarantine_split": dict(
        c2_dual_output_head=True, c2_relmap=True, c2_floor_candidate_split=True,
        c2_quarantine_candidate=True, c2_value_evidence_v2=True, c2_transition_hint=True,
        c2_task_palette_feature=True, c2_rel_where_hint=True,
    ),
}


def _make_batch(cfg_dict: dict, seq_len: int, device: torch.device, seed: int = 7) -> dict:
    """A deterministic synthetic batch with target + support demos + target rel_maps."""
    g = torch.Generator().manual_seed(seed)
    B, M = 2, 3
    side = int(seq_len ** 0.5)
    assert side * side == seq_len
    inputs = torch.randint(0, 12, (B, seq_len), generator=g)
    ci = torch.randint(0, 12, (B, M, seq_len), generator=g)
    co = torch.randint(0, 12, (B, M, seq_len), generator=g)
    cm = torch.ones(B, M, dtype=torch.bool)
    pid = torch.randint(0, 4, (B,), generator=g)
    batch = {
        "inputs": inputs.to(device),
        "puzzle_identifiers": pid.to(device),
        "context_inputs": ci.to(device),
        "context_outputs": co.to(device),
        "context_mask": cm.to(device),
    }
    # target rel_maps (the wired paths stash these; providing them avoids the inline-fallback warning)
    batch["rel_maps"] = relational_maps(inputs, side=side).to(device)
    batch["context_rel_maps"] = relational_maps(
        ci.reshape(B * M, seq_len), side=side).view(B, M, seq_len, REL_MAP_CHANNELS).to(device)
    batch["context_output_rel_maps"] = relational_maps(
        co.reshape(B * M, seq_len), side=side).view(B, M, seq_len, REL_MAP_CHANNELS).to(device)
    batch["labels"] = torch.randint(0, 12, (B, seq_len), generator=g).to(device)
    return batch


def _build(module, cfg_dict: dict, seq_len: int, hidden: int, seed: int):
    torch.manual_seed(seed)
    cfg = module.FVR_C2_Config(**{**_base(seq_len, hidden), **cfg_dict})
    m = module.TinyRecursiveReasoningModel_ACTV1_Inner(cfg)
    m.eval()
    return m


@torch.no_grad()
def _run(model, batch) -> dict:
    """One eval forward -> flat dict of every tensor output (output, q0, q1, all extras)."""
    _carry, output, (q0, q1), merged = model.forward(model.fresh_carry(batch["inputs"].shape[0]), batch)
    out = {"__output__": output.float(), "__q0__": q0.float(), "__q1__": q1.float()}
    for k, v in merged.items():
        if torch.is_tensor(v):
            out[k] = v.detach().float()
    return out


def _compare(a: dict, b: dict) -> list[str]:
    """Return a list of human-readable mismatches ([] == byte-identical)."""
    diffs = []
    ka, kb = set(a), set(b)
    if ka != kb:
        diffs.append(f"key-set differs: only-A={sorted(ka - kb)} only-B={sorted(kb - ka)}")
    for k in sorted(ka & kb):
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            diffs.append(f"[{k}] shape {tuple(ta.shape)} != {tuple(tb.shape)}")
        elif not torch.equal(ta, tb):
            md = (ta - tb).abs().max().item()
            diffs.append(f"[{k}] value differs (max_abs_delta={md:.3e})")
    return diffs


def identity_check(build_a, build_b, seq_len: int = 36, hidden: int = 64, verbose: bool = True) -> bool:
    """For every CONFIG variant: A and B (sharing A's state_dict) must produce byte-identical forwards."""
    dev = torch.device("cpu")
    all_ok = True
    for name, cfg_dict in CONFIGS.items():
        ma = build_a(cfg_dict, seq_len, hidden, seed=0)
        mb = build_b(cfg_dict, seq_len, hidden, seed=123)          # different init...
        mb.load_state_dict(ma.state_dict())                        # ...then share A's weights
        batch = _make_batch(cfg_dict, seq_len, dev)
        diffs = _compare(_run(ma, batch), _run(mb, batch))
        ok = not diffs
        all_ok &= ok
        if verbose:
            print(f"  [{name}] {'IDENTICAL' if ok else 'MISMATCH'}"
                  + ("" if ok else "\n    " + "\n    ".join(diffs)))
    return all_ok


def _negative_control(build, seq_len: int = 36, hidden: int = 64) -> bool:
    """Perturb ONE weight after loading -> the harness MUST report a difference (detector is live)."""
    dev = torch.device("cpu")
    cfg_dict = CONFIGS["a_dual_relmap"]
    ma = build(cfg_dict, seq_len, hidden, seed=0)
    mb = build(cfg_dict, seq_len, hidden, seed=0)
    mb.load_state_dict(ma.state_dict())
    with torch.no_grad():
        mb.color_head.weight[0, 0] += 1.0                          # single-weight perturbation
    batch = _make_batch(cfg_dict, seq_len, dev)
    diffs = _compare(_run(ma, batch), _run(mb, batch))
    detected = len(diffs) > 0
    print(f"  [negative-control] perturbation {'DETECTED' if detected else 'MISSED (harness is vacuous!)'}"
          + (f" -> {diffs[0]}" if detected else ""))
    return detected


if __name__ == "__main__":
    import models.recursive_reasoning.trm_fvr_c2 as OLD

    def build_old(cfg_dict, seq_len, hidden, seed):
        return _build(OLD, cfg_dict, seq_len, hidden, seed)

    print("=== Block 0: step-0 identity harness, OLD vs OLD (proving the gate is sound) ===")
    pos = identity_check(build_old, build_old)
    neg = _negative_control(build_old)
    print(f"\nPOSITIVE gate (identical weights -> identical forward): {'PASS' if pos else 'FAIL'}")
    print(f"NEGATIVE control (perturbation detected):              {'PASS' if neg else 'FAIL'}")
    print("BLOCK 0 HARNESS: " + ("PASS" if (pos and neg) else "FAIL"))
    sys.exit(0 if (pos and neg) else 1)
