"""PD-V2 GATE (File #5 Block 0): byte-equality of pair_delta_v2 vs pair_delta_encoder (oracle).

Checks, per batch family (random / ARC-like / edge cases):
  * demo_delta_features: feats + valid byte-equal (torch.equal, dtype+shape included).
  * pairdelta_intent_features: same key set, every tensor byte-equal.
  * (Block 2 extends this file with the shared-weight module identity for
    PairDeltaEncoder / RuleConditionedDecoder.)
NEGATIVE control: a perturbed input must NOT match (proves the comparator can fail).

Run:  ./trm/Scripts/python.exe scripts/pd_v2_gate.py     (from TinyRecursiveModels)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models.recursive_reasoning.pair_delta_encoder as OLD  # noqa: E402
import models.recursive_reasoning.pair_delta_v2 as NEW       # noqa: E402


# ------------------------------------------------------------------ batch builders
def rand_batch(seed: int, B: int = 4, M: int = 3, L: int = 900):
    """Uniform random tokens 0..11 -- stresses every clamp/mask branch."""
    g = torch.Generator().manual_seed(seed)
    ci = torch.randint(0, 12, (B, M, L), generator=g)
    co = torch.randint(0, 12, (B, M, L), generator=g)
    cm = torch.rand(B, M, generator=g) > 0.3
    cm[:, 0] = True
    return ci, co, cm


def arc_like(seed: int, B: int = 4, M: int = 3, side: int = 30):
    """Proper ARC layout: HxW colour block, EOS ring, PAD elsewhere; output = recolour (+ sometimes
    a different extent, to exercise the shape/area scalars)."""
    g = torch.Generator().manual_seed(seed)
    L = side * side
    ci = torch.zeros(B, M, L, dtype=torch.long)
    co = torch.zeros(B, M, L, dtype=torch.long)

    def place(canvas: torch.Tensor, block: torch.Tensor) -> None:
        h, w = block.shape
        canvas[:h, :w] = block
        if h < side:
            canvas[h, :min(w + 1, side)] = 1
        if w < side:
            canvas[:min(h + 1, side), w] = 1

    for b in range(B):
        for m in range(M):
            h = int(torch.randint(3, 12, (1,), generator=g))
            w = int(torch.randint(3, 12, (1,), generator=g))
            block = torch.randint(2, 12, (h, w), generator=g)
            src = int(torch.randint(2, 12, (1,), generator=g))
            dst = int(torch.randint(2, 12, (1,), generator=g))
            out_block = block.clone()
            out_block[block == src] = dst
            if (b + m) % 3 == 2:                      # a third of demos change extent too
                out_block = out_block[: max(2, h - 1), : max(2, w - 1)]
            gi = torch.zeros(side, side, dtype=torch.long)
            go = torch.zeros(side, side, dtype=torch.long)
            place(gi, block)
            place(go, out_block)
            ci[b, m] = gi.reshape(-1)
            co[b, m] = go.reshape(-1)
    cm = torch.ones(B, M, dtype=torch.bool)
    return ci, co, cm


def edge_batches():
    """The branch-coverage set: all-pad, identity (x==y), fully-masked row, single demo."""
    L = 900
    allpad = (torch.zeros(2, 3, L, dtype=torch.long),
              torch.zeros(2, 3, L, dtype=torch.long),
              torch.ones(2, 3, dtype=torch.bool))
    ident_ci = torch.zeros(2, 3, L, dtype=torch.long)
    ident_ci[:, :, :50] = 5
    identity = (ident_ci, ident_ci.clone(), torch.ones(2, 3, dtype=torch.bool))
    ci, co, cm = arc_like(seed=99)
    cm = cm.clone()
    cm[0] = False                                    # a task with NO valid demos
    masked = (ci, co, cm)
    ci1, co1, cm1 = arc_like(seed=123, M=1)          # single-demo task
    return {"allpad": allpad, "identity": identity, "masked_row": masked, "single_demo": (ci1, co1, cm1)}


# ------------------------------------------------------------------ comparators
def eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def check_batch(name: str, ci, co, cm) -> None:
    fo, vo = OLD.demo_delta_features(ci, co, cm)
    fn, vn = NEW.demo_delta_features(ci, co, cm)
    assert eq(fo, fn), f"[{name}] demo_delta_features.feats mismatch"
    assert eq(vo, vn), f"[{name}] demo_delta_features.valid mismatch"
    io = OLD.pairdelta_intent_features(ci, co, cm)
    inw = NEW.pairdelta_intent_features(ci, co, cm)
    assert set(io) == set(inw), f"[{name}] intent key sets differ: {set(io) ^ set(inw)}"
    for k in io:
        assert eq(io[k], inw[k]), f"[{name}] intent['{k}'] mismatch"


def check_modules() -> None:
    """Block 2: shared-weight forward identity. OLD's state_dict must load strict=True into
    NEW (param-name compatibility = checkpoint compatibility), and with identical weights the
    forwards must be byte-equal in eval mode."""
    torch.manual_seed(0)
    enc_old = OLD.PairDeltaEncoder(hidden_dim=64, n_slots=8, n_heads=4).eval()
    enc_new = NEW.PairDeltaEncoder(hidden_dim=64, n_slots=8, n_heads=4).eval()
    enc_new.load_state_dict(enc_old.state_dict(), strict=True)
    dec_old = OLD.RuleConditionedDecoder(hidden_dim=64, n_heads=4, n_layers=2).eval()
    dec_new = NEW.RuleConditionedDecoder(hidden_dim=64, n_heads=4, n_layers=2).eval()
    dec_new.load_state_dict(dec_old.state_dict(), strict=True)

    with torch.no_grad():
        for seed in (0, 3):
            ci, co, cm = arc_like(seed)
            oo = enc_old(ci, co, cm)
            on = enc_new(ci, co, cm)
            assert set(oo) == set(on)
            for k in oo:
                assert eq(oo[k], on[k]), f"encoder['{k}'] mismatch (seed {seed})"
            xq = ci[:, 0]
            assert eq(dec_old(xq, oo["rule_slots"]), dec_new(xq, on["rule_slots"])), \
                f"decoder(slots) mismatch (seed {seed})"
            assert eq(dec_old(xq, None), dec_new(xq, None)), f"decoder(None) mismatch (seed {seed})"

        # empty-task path (all demos masked) must stay byte-equal too
        ci, co, cm = arc_like(seed=11)
        cm = torch.zeros_like(cm)
        oo, on = enc_old(ci, co, cm), enc_new(ci, co, cm)
        for k in oo:
            assert eq(oo[k], on[k]), f"encoder empty-task ['{k}'] mismatch"
        assert float(on["rule_confidence"].abs().max()) == 0.0

        # NEGATIVE module control: perturb one weight -> forwards must differ.
        enc_new.slot_queries.add_(0.05)
        on2 = enc_new(*arc_like(0))
        oo2 = enc_old(*arc_like(0))
        assert not eq(oo2["rule_vec"], on2["rule_vec"]), "module negative control failed"
    print("MODULE gate: shared-weight identity (encoder+decoder) + negative control ... PASS")


def check_fast_path() -> None:
    """Block 2: SS2 scatter kernel vs verbatim one_hot math -- allclose (accumulation order
    differs), valid mask byte-equal, intent counts-derived outputs allclose."""
    for seed in (0, 1, 2):
        for ci, co, cm in (rand_batch(seed), arc_like(seed)):
            fs, vs = NEW.demo_delta_features(ci, co, cm, fast=False)
            ff, vf = NEW.demo_delta_features(ci, co, cm, fast=True)
            assert eq(vs, vf)
            assert torch.allclose(fs, ff, rtol=1e-5, atol=1e-6), f"fast feats drift (seed {seed})"
            is_ = NEW.pairdelta_intent_features(ci, co, cm, fast=False)
            if_ = NEW.pairdelta_intent_features(ci, co, cm, fast=True)
            for k in is_:
                assert torch.allclose(is_[k], if_[k], rtol=1e-5, atol=1e-6), \
                    f"fast intent['{k}'] drift (seed {seed})"
    print("FAST-PATH gate: scatter kernel allclose vs verbatim ... PASS")


# ------------------------------------------------------------------ Blocks 3/4: constructed tasks
def grid(side: int = 30, h: int = 0, w: int = 0, fill: int = 0) -> torch.Tensor:
    """Canonical ARC layout: h x w block of `fill`, thin-L EOS ring, PAD elsewhere. [side*side]."""
    g = torch.zeros(side, side, dtype=torch.long)
    if h and w:
        g[:h, :w] = fill
        if h < side:
            g[h, :min(w + 1, side)] = 1
        if w < side:
            g[:min(h + 1, side), w] = 1
    return g.reshape(-1)


def check_pd_color() -> None:
    side, L = 30, 900
    C = NEW  # builders live only in v2

    # (1) global recolor: 3 demos of colours {3,4} where 4 -> 7 everywhere; colour 3 copies.
    ci = torch.zeros(1, 3, L, dtype=torch.long)
    co = torch.zeros(1, 3, L, dtype=torch.long)
    for m in range(3):
        gi = grid(side, 6, 6, fill=5)                       # colour 3
        gi2 = gi.clone()
        gi2[torch.arange(0, 6 * side, side)] = 6            # col 0 = colour 4
        go = gi2.clone()
        go[gi2 == 6] = 9                                    # 4 -> 7
        ci[0, m] = gi2
        co[0, m] = go
    cm = torch.ones(1, 3, dtype=torch.bool)
    tgt = grid(side, 6, 6, fill=5)
    tgt[torch.arange(0, 6 * side, side)] = 6
    tgt = tgt.unsqueeze(0)                                  # [1, L]
    f, _ = C.pd_color_evidence(ci, co, cm, tgt)
    is4 = tgt == 6
    is3 = tgt == 5
    assert torch.all(f[..., C.PDC_CONSENSUS + 7][is4] == 1.0), "consensus 4->7 should be 1.0"
    assert torch.all(f[..., C.PDC_MIN_CHANGE][is4] == 1.0), "colour 4 changes in every demo"
    assert torch.all(f[..., C.PDC_MIN_CHANGE][is3] == 0.0), "colour 3 never changes"
    assert torch.all(f[..., C.PDC_SUPPORT][is4] == 1.0)
    pad_eos = (tgt < 2)
    assert torch.all(f[pad_eos] == 0.0), "pad/eos cells must be zeroed"

    # (2) one DISSENTING demo (identity output: 4 does NOT change there) must weaken agreement.
    co_shuf = co.clone()
    co_shuf[0, 0] = ci[0, 0].clone()
    fs, _ = C.pd_color_evidence(ci, co_shuf, cm, tgt)
    assert torch.all(fs[..., C.PDC_MIN_CHANGE][is4] == 0.0), "min over demos must drop to 0"
    got_c = fs[..., C.PDC_CONSENSUS + 7][is4]
    assert torch.allclose(got_c, torch.full_like(got_c, 2.0 / 3.0)), "consensus must fall to 2/3"

    # (3) identity task -> agreement channels all zero.
    fi, _ = C.pd_color_evidence(ci, ci.clone(), cm, tgt)
    assert torch.all(fi[..., :C.PDC_SUPPORT] == 0.0), "identity task: consensus+min_change zero"

    # (4) positional: only the TOP ROW of each demo extent changes (3 -> 4). Demo 6x6; target 12x12.
    ci2 = torch.zeros(1, 3, L, dtype=torch.long)
    co2 = torch.zeros(1, 3, L, dtype=torch.long)
    for m in range(3):
        gi = grid(side, 6, 6, fill=5)
        go = gi.clone()
        go[:6] = 6                                          # row 0, cols 0..5 only: 3 -> 4
        ci2[0, m] = gi
        co2[0, m] = go
    tgt2 = grid(side, 12, 12, fill=5).unsqueeze(0)          # [1, L]
    f2, _ = C.pd_color_evidence(ci2, co2, cm, tgt2)
    t2 = tgt2.view(side, side)
    fr = f2[0, :, C.PDC_ROW_PRIOR].view(side, side)
    assert torch.all(fr[0:2, :12][t2[0:2, :12] >= 2] == 1.0), "target rows 0-1 = band 0 -> prior 1"
    assert torch.all(fr[2:12, :12][t2[2:12, :12] >= 2] == 0.0), "lower bands never change"
    fc = f2[0, :, C.PDC_COL_PRIOR].view(side, side)
    expect = 1.0 / 6.0
    got = fc[0:12, 0:12][t2[0:12, 0:12] >= 2]
    assert torch.allclose(got, torch.full_like(got, expect), atol=1e-5), "col prior = 1/6 everywhere"
    print("PD-COLOR gate: recolor/garbage/identity/positional constructed tasks ... PASS")


def check_pd_struct() -> None:
    side, L = 30, 900
    C = NEW
    cm = torch.ones(1, 2, dtype=torch.bool)

    def task(pairs, tgt):
        ci = torch.stack([p[0] for p in pairs]).unsqueeze(0)
        co = torch.stack([p[1] for p in pairs]).unsqueeze(0)
        return C.pd_structure_evidence(ci, co, cm[:, :len(pairs)], tgt.unsqueeze(0))

    def expect_masks(f, offset, h, w, name):
        vm = f[0, :, offset].view(side, side)
        em = f[0, :, offset + 1].view(side, side)
        ref_v = torch.zeros(side, side)
        ref_v[:h, :w] = 1.0
        ref_e = torch.zeros(side, side)
        if h < side:
            ref_e[h, :min(w + 1, side)] = 1.0
        if w < side:
            ref_e[:min(h + 1, side), w] = 1.0
        assert torch.equal(vm, ref_v), f"{name}: valid mask wrong"
        assert torch.equal(em, ref_e), f"{name}: eos mask wrong"

    # (1) preserve: 2 demos same extents in/out; target 5x7.
    f, _ = task([(grid(side, 4, 6, 5), grid(side, 4, 6, 7)),
                 (grid(side, 9, 3, 5), grid(side, 9, 3, 7))], grid(side, 5, 7, 5))
    expect_masks(f, C.PDS_PRESERVE, 5, 7, "preserve")
    assert torch.all(f[..., C.PDS_TRANSPOSE:] == 0.0), "transpose/bbox must NOT fire"

    # (2) transpose: out h,w == in w,h with a non-square demo; target 3x9 -> predict 9x3.
    f, _ = task([(grid(side, 4, 6, 5), grid(side, 6, 4, 7)),
                 (grid(side, 2, 8, 5), grid(side, 8, 2, 7))], grid(side, 3, 9, 5))
    expect_masks(f, C.PDS_TRANSPOSE, 9, 3, "transpose")
    assert torch.all(f[..., C.PDS_PRESERVE:C.PDS_PRESERVE + 2] == 0.0), "preserve must NOT fire"

    # (3) bbox: input 8x8 extent, non-bg content only 3x5 (rest colour 0); out extent 3x5.
    def bbox_in():
        g = grid(side, 8, 8, fill=2)                        # colour 0 = background
        gg = g.view(side, side)
        gg[:3, :5] = 7                                      # non-bg content
        return gg.reshape(-1)

    f, _ = task([(bbox_in(), grid(side, 3, 5, 7)),
                 (bbox_in(), grid(side, 3, 5, 7))], bbox_in())
    expect_masks(f, C.PDS_BBOX, 3, 5, "bbox")
    assert torch.all(f[..., C.PDS_PRESERVE:C.PDS_PRESERVE + 2] == 0.0)

    # (4) inconsistent demos -> nothing fires.
    f, _ = task([(grid(side, 4, 6, 5), grid(side, 4, 6, 7)),
                 (grid(side, 9, 3, 5), grid(side, 5, 5, 7))], grid(side, 5, 7, 5))
    assert torch.all(f == 0.0), "no family verifies on inconsistent demos"
    print("PD-STRUCT gate: preserve/transpose/bbox/inconsistent constructed tasks ... PASS")


def check_pd_bidi() -> None:
    side, L = 30, 900
    C = NEW
    cm = torch.ones(1, 2, dtype=torch.bool)

    def two(gi, go):
        ci = torch.stack([gi, gi]).unsqueeze(0)
        co = torch.stack([go, go]).unsqueeze(0)
        return ci, co

    # (1) bijective recolor (4->7, 3->5): invertibility 1; dst_mass 1 on 7-or-5-coloured target cells.
    gi = grid(side, 6, 6, fill=6)                          # colour 4
    gg = gi.view(side, side)
    gg[3:, :] = torch.where(gg[3:, :] == 6, torch.tensor(5), gg[3:, :])   # bottom half colour 3
    go = gi.clone()
    go[gi == 6] = 9                                        # 4 -> 7
    go[gi == 5] = 7                                        # 3 -> 5
    ci, co = two(gi, go)
    tgt = torch.zeros(1, L, dtype=torch.long)
    tgt[0, :3] = torch.tensor([9, 7, 6])                   # colours 7, 5, 4
    f, _ = C.pd_bidi_evidence(ci, co, cm, tgt)
    assert torch.all(f[0, :3, C.PDB_INVERT] == 1.0), "bijection must score invertibility 1"
    assert abs(float(f[0, 0, C.PDB_DST_MASS]) - 0.5) < 1e-5, "half the mass arrives at colour 7"
    assert float(f[0, 2, C.PDB_DST_MASS]) == 0.0, "nothing arrives at colour 4"

    # (2) many-to-one (3->7 AND 4->7, equal mass): fwd 1, bwd 0.5 -> invertibility 0.5.
    go2 = gi.clone()
    go2[(gi == 6) | (gi == 5)] = 9
    ci, co = two(gi, go2)
    f2, _ = C.pd_bidi_evidence(ci, co, cm, tgt)
    assert abs(float(f2[0, 0, C.PDB_INVERT]) - 0.5) < 1e-5, "many-to-one must halve invertibility"

    # (3) deletion: colour 4 column falls off the grid (6x6 -> 6x5); colour 3 stays.
    gi3 = grid(side, 6, 6, fill=5)
    g3 = gi3.view(side, side)
    g3[:6, 5] = 6                                          # last col colour 4
    go3 = grid(side, 6, 5, fill=5)
    ci, co = two(gi3, go3)
    f3, _ = C.pd_bidi_evidence(ci, co, cm, tgt)            # tgt cell 2 = colour 4, cells 0/1 not in {3}
    assert float(f3[0, 2, C.PDB_DEL_RATE]) == 1.0, "colour 4 always deleted"
    assert float(f3[0, 2, C.PDB_DEL_MIN]) == 1.0, "deleted in EVERY demo"
    tgt3 = torch.zeros(1, L, dtype=torch.long)
    tgt3[0, 0] = 5                                         # colour 3
    f3b, _ = C.pd_bidi_evidence(ci, co, cm, tgt3)
    assert float(f3b[0, 0, C.PDB_DEL_RATE]) == 0.0, "colour 3 never deleted"
    print("PD-BIDI gate: bijective/many-to-one/deletion constructed tasks ... PASS")


def main() -> None:
    # constants must agree
    for c in ("PAD_TOKEN", "EOS_TOKEN", "COLOR_OFFSET", "N_COLORS", "N_TRANSITIONS",
              "GRID_SIDE", "GRID_LEN", "VOCAB", "FEATURE_DIM"):
        assert getattr(OLD, c) == getattr(NEW, c), f"constant {c} drifted"

    n = 0
    for seed in (0, 1, 2, 7):
        check_batch(f"rand{seed}", *rand_batch(seed)); n += 1
        check_batch(f"arc{seed}", *arc_like(seed)); n += 1
    for name, (ci, co, cm) in edge_batches().items():
        check_batch(name, ci, co, cm); n += 1
    print(f"POSITIVE gate: {n} batches byte-equal ... PASS")

    # NEGATIVE control: perturb one token -> comparator MUST fail somewhere.
    ci, co, cm = arc_like(seed=5)
    co2 = co.clone()
    co2[0, 0, 0] = (int(co2[0, 0, 0]) + 3) % 12
    fo, _ = OLD.demo_delta_features(ci, co, cm)
    fn, _ = NEW.demo_delta_features(ci, co2, cm)
    assert not eq(fo, fn), "NEGATIVE control failed: perturbed batch compared equal"
    print("NEGATIVE control: perturbation detected ... PASS")

    check_modules()      # Block 2
    check_fast_path()    # Block 2
    check_pd_color()     # Block 3
    check_pd_struct()    # Block 4
    check_pd_bidi()      # SS7 / D10
    print("PD-V2 GATE: PASS")


if __name__ == "__main__":
    main()
