"""0.2 - the CROSS-DEMO EXTRACTION CHECK: the gate every rule extractor must pass.

An extractor that produces a "rule" which does NOT depend on which demos it sees (or how their
inputs pair with their outputs) is useless. That is exactly what C2 collapsed to: gate ~0,
real_vs_shuffle ~0 -- a "rule" invariant to the task. This check makes that impossible to miss.

Before any extractor is allowed to feed the Rule Bus it must pass these tests on synthetic tasks
with a KNOWN injected rule (recolour src -> dst, consistent across a task's demos):

  1. DEMO-SENSITIVITY  corrupt ONE demo's rule -> the descriptor must MOVE  (proves it reads
                       every demo; an extractor that ignores demos won't react)
  2. AGREEMENT         the per-demo signals are consistent on a real task   (diagnostic)
  3. REAL-vs-SHUFFLE   real demos vs pairing-broken demos -> descriptor must DIFFER  (key test)
  4. KNOWN-RULE        on injected src->dst tasks, recover dst from src      (correctness)
  5. COVERAGE          fraction of changed cells the rule explains (dpcc-like)

VERDICT = PASS iff (sensitivity) AND (real_vs_shuffle) AND (known_rule).

Self-validation: run on TWO extractors; the check MUST tell them apart --
  * ColorTransitionBank.cond_inout/cond_changed  -> PASS (task-specific by construction)
  * ShuffleInvariantPooler (input/output colour MARGINALS, ignores in->out pairing) -> FAIL
    (reproduces C2's collapse). The synthetic makes the rule visible ONLY through pairing
    (a dominant NOISE colour hides dst from the output marginal), so the pooler cannot recover it.
  If the check can't separate these, the check is broken.

Run:  trm\\Scripts\\python.exe scripts\\check_extraction.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from models.recursive_reasoning.color_transition_bank import ColorTransitionBank
from models.recursive_reasoning.object_bank import is_singleton_object, size_bucket

COLOR_OFFSET = 2
N_COLORS = 10
GRID_SIDE = 30
GRID_LEN = GRID_SIDE * GRID_SIDE
NOISE = 0                  # dominant background colour (hides dst from the output marginal)

# Real-data loader constants (mirror run_stage1_local.py so the ranker reads the SAME dataset the
# training panel scores). Used only by --real.
CONFIG_PATH = "checkpoints/TRM-FVR-Experiments/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/all_config.yaml"
CKPT_PATH = r"D:\trm_c2\step_518071"
DATASET_PATH = r"D:\trm_c2\arc1concept-aug-1000"

# Selectable diagnostic probe datasets (each value is the ROOT dir containing train/; the loader
# appends the split, see puzzle_dataset._lazy_load_dataset). aug0 is the UN-AUGMENTED 960-task seed
# of aug-1000: identical intra-puzzle LODO structure (context = the OTHER examples of the SAME
# puzzle, _context_from_flat_examples), but canonical palette, 960 unique tasks, loads in ~7MB.
# It is NOT held out (its tasks overlap aug-1000 training) -- a clean MEASURING stick, not a
# generalization test. aug1000 = the actual training distribution (~913x augmented).
PROBE_DATASETS = {
    "aug1000": DATASET_PATH,
    "aug0": r"D:\trm_c2\TinyRecursiveModels\data\arc1concept-aug-0",
}


# ---------------------------------------------------------------------------
# Synthetic tasks. Task b recolours src_b -> dst_b in every demo. Grids are mostly NOISE so the
# rule is recoverable ONLY from the input->output PAIRING, never from the output marginal.
# ---------------------------------------------------------------------------
from parse import _components_2d, _components_2d_adj, _parse_same, _neighbour_key, _position_key, _shape_key, _shape_class_key, _rank_key, _touch_key, _topology_key, _count_key, _combined_key, _extract_grid, _mode, _d4, _shift, _grid_bg, _enclosed_bg, _objgroups, _d4_canon_hash, _object_keymap, KEYS, KEYS_COND
from solve import _DihedralOp, _TranslateOp, _TileOp, _CropOp, _ScaleOp, _SymmetrizeOp, _FillOp, _ExtractObjectOp, _PanelOp, _FractalOp, _MirrorConcatOp, _fit_recolor_2d, _fit_recolor_map, _fit_combined_recolor_map, _fit_neighbour_recolor_map, _geo_routed, _compose2_solve, _object_recolor_solve, _object_relrecolor_solve, _set_cover_solve, _distance_recolor_solve, _hole_filler_recolor_solve, _legend_match_solve, _execute_recipe_solve, _committed_solve, _default_geo_ops, _op1_options, _apply_recolor_2d, _apply_recolor, _apply_combined_recolor, _apply_neighbour_recolor, _object_relrecolor_predict, _set_cover_predict
def make_recolor_tasks(B: int = 8, M: int = 4, L: int = GRID_LEN, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ci = torch.zeros(B, M, L, dtype=torch.long)
    co = torch.zeros(B, M, L, dtype=torch.long)
    src = torch.zeros(B, dtype=torch.long)
    dst = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        s = int(torch.randint(1, N_COLORS, (1,), generator=g))       # src,dst in 1..9 (!= NOISE)
        d = int(torch.randint(1, N_COLORS, (1,), generator=g))
        while d == s:
            d = int(torch.randint(1, N_COLORS, (1,), generator=g))
        src[b], dst[b] = s, d
        for m in range(M):
            inp = torch.full((L,), NOISE + COLOR_OFFSET, dtype=torch.long)   # mostly NOISE
            inp[:12] = s + COLOR_OFFSET                               # guaranteed src block (changes)
            idx = torch.randperm(L, generator=g)[:20] + 12           # a few scattered OTHER colours
            idx = idx[idx < L]
            inp[idx] = torch.randint(1, N_COLORS, (idx.numel(),), generator=g) + COLOR_OFFSET
            out = inp.clone()
            out[inp == (s + COLOR_OFFSET)] = d + COLOR_OFFSET         # APPLY src->dst
            ci[b, m], co[b, m] = inp, out
    cm = torch.ones(B, M, dtype=torch.bool)
    return ci, co, cm, src, dst


# ---------------------------------------------------------------------------
# Two extractors with the SAME interface: descriptor() + recover().
# ---------------------------------------------------------------------------
class CTBankExtractor:
    """Good extractor: explicit colour-transition consensus (task-specific by construction)."""
    name = "ColorTransitionBank(cond_inout/cond_changed)"

    def __init__(self):
        self.bank = ColorTransitionBank(hidden_dim=128, rule_tokens=8)

    def descriptor(self, ci, co, cm):
        out = self.bank(ci, co, cm, compute_metrics=False, compute_rule_tokens=False)
        return out["cond_inout"].reshape(ci.shape[0], -1)            # [B,100]

    def recover(self, ci, co, cm, src):
        out = self.bank(ci, co, cm, compute_metrics=False, compute_rule_tokens=False)
        cc = out["cond_changed"]                                     # [B,10,10] P(out|in,changed)
        row = cc.gather(1, src.view(-1, 1, 1).expand(-1, 1, N_COLORS)).squeeze(1)  # [B,10]
        return row.argmax(-1)                                        # predicted dst


class ShuffleInvariantPooler:
    """Bad extractor: pools input/output colour MARGINALS, ignoring the in->out pairing.
    Reproduces C2's collapse -- a 'rule' invariant to which output goes with which input."""
    name = "ShuffleInvariantPooler(marginals, no pairing)"

    @staticmethod
    def _hist(grid, cm):
        x = grid.long()
        col = x >= COLOR_OFFSET
        xc = (x - COLOR_OFFSET).clamp(0, 9)
        oh = F.one_hot(xc, N_COLORS).float() * (col & cm.unsqueeze(-1)).unsqueeze(-1).float()
        h = oh.sum(dim=(1, 2))                                       # [B,10] pooled over demos+cells
        return h / h.sum(-1, keepdim=True).clamp_min(1e-6)

    def descriptor(self, ci, co, cm):
        return torch.cat([self._hist(ci, cm), self._hist(co, cm)], -1)   # [B,20] marginals only

    def recover(self, ci, co, cm, src):
        return self._hist(co, cm).argmax(-1)                        # most common output colour (= NOISE, not dst)


# ---------------------------------------------------------------------------
# The five tests.
# ---------------------------------------------------------------------------
def _rel_dist(a, b):
    return (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6)     # [B]


def run_check(ext, ci, co, cm, src, dst):
    base = ext.descriptor(ci, co, cm)                               # [B,D]
    B, M = ci.shape[0], ci.shape[1]

    # 1. DEMO-SENSITIVITY: corrupt ONE demo's rule (src -> a WRONG colour) -> descriptor must move.
    co_cor = co.clone()
    for b in range(B):
        wrong = next(c for c in range(1, N_COLORS)
                     if c != int(src[b].item()) and c != int(dst[b].item()))
        mask0 = ci[b, 0] == (src[b] + COLOR_OFFSET)
        d0 = co_cor[b, 0]
        d0[mask0] = wrong + COLOR_OFFSET
        co_cor[b, 0] = d0
    sensitivity = _rel_dist(base, ext.descriptor(ci, co_cor, cm)).mean().item()

    # 2. AGREEMENT: per-demo descriptors should be mutually consistent (mean pairwise cosine).
    per = [F.normalize(ext.descriptor(ci[:, m:m + 1], co[:, m:m + 1], cm[:, m:m + 1]), dim=-1)
           for m in range(M)]
    cos = [(per[i] * per[j]).sum(-1) for i in range(M) for j in range(i + 1, M)]
    agreement = torch.stack(cos).mean().item() if cos else float("nan")

    # 3. REAL-vs-SHUFFLE: break the in->out pairing (roll outputs within each task).
    real_vs_shuffle = _rel_dist(base, ext.descriptor(ci, co.roll(1, dims=1), cm)).mean().item()

    # 4. KNOWN-RULE recovery: predict dst from src.
    known_rule = (ext.recover(ci, co, cm, src) == dst).float().mean().item()

    # 5. COVERAGE (single-rule tasks: = recovery acc; a multi-rule extractor would do per-cell).
    coverage = known_rule

    g_sens = sensitivity > 0.02
    g_shuf = real_vs_shuffle > 0.30
    g_known = known_rule > 0.50
    verdict = g_sens and g_shuf and g_known
    return {"sensitivity": sensitivity, "agreement": agreement, "real_vs_shuffle": real_vs_shuffle,
            "known_rule": known_rule, "coverage": coverage,
            "g_sens": g_sens, "g_shuf": g_shuf, "g_known": g_known, "verdict": verdict}


def _fmt(r):
    return (f"sensitivity={r['sensitivity']:.3f}{'OK' if r['g_sens'] else 'XX'}  "
            f"agree={r['agreement']:.2f}  "
            f"real_vs_shuffle={r['real_vs_shuffle']:.3f}{'OK' if r['g_shuf'] else 'XX'}  "
            f"known_rule={r['known_rule']:.2f}{'OK' if r['g_known'] else 'XX'}  "
            f"=> {'PASS' if r['verdict'] else 'FAIL'}")


def make_object_conditioned_tasks(B: int = 8, M: int = 4, seed: int = 1):
    """task: colour a -> b INSIDE the largest object (a 5x6 block), a -> c for scattered singletons.
    A cell-colour-only rule must pick ONE dst for a (the majority, b) and is WRONG on the minority;
    an object-conditioned rule (split by is-largest) recovers BOTH. The proof that the VALUE needs
    object context, not just cell colour (Phase 2c)."""
    S, L = GRID_SIDE, GRID_LEN
    g = torch.Generator().manual_seed(seed)
    ci = torch.full((B, M, L), NOISE + COLOR_OFFSET, dtype=torch.long)
    co = torch.full((B, M, L), NOISE + COLOR_OFFSET, dtype=torch.long)
    meta = []
    scatter = [(20, 20), (22, 24), (25, 21), (27, 27), (24, 28)]               # isolated a-cells
    for b in range(B):
        a = int(torch.randint(1, N_COLORS, (1,), generator=g))
        bb = int(torch.randint(1, N_COLORS, (1,), generator=g))
        while bb == a:
            bb = int(torch.randint(1, N_COLORS, (1,), generator=g))
        cc = int(torch.randint(1, N_COLORS, (1,), generator=g))
        while cc in (a, bb):
            cc = int(torch.randint(1, N_COLORS, (1,), generator=g))
        for m in range(M):
            grid = torch.full((S, S), NOISE + COLOR_OFFSET, dtype=torch.long)
            grid[0:5, 0:6] = a + COLOR_OFFSET                                  # 30-cell block (largest a-object)
            for (r, c) in scatter:
                grid[r, c] = a + COLOR_OFFSET
            out = grid.clone()
            out[0:5, 0:6] = bb + COLOR_OFFSET                                  # block -> b
            for (r, c) in scatter:
                out[r, c] = cc + COLOR_OFFSET                                  # singletons -> c
            ci[b, m] = grid.reshape(L)
            co[b, m] = out.reshape(L)
        meta.append((a, bb, cc))
    return ci, co, torch.ones(B, M, dtype=torch.bool), meta


def _demo_object_conditioned():
    """Recover the conditional recolour from demos 1..M-1, score on held-out demo 0. The
    object-conditioned consensus must beat the cell-colour-only consensus on changed cells."""
    ci, co, cm, _meta = make_object_conditioned_tasks()
    B, M, _ = ci.shape
    S = GRID_SIDE
    q_in, q_out = ci[:, 0], co[:, 0]
    changed_q = (q_in != q_out) & (q_in >= COLOR_OFFSET)
    q_single = is_singleton_object(q_in, S)                                    # the discriminating property
    cell_c = obj_c = tot = 0.0
    for b in range(B):
        cell_votes: dict = {}
        obj_votes: dict = {}
        for mm in range(1, M):
            din, dou = ci[b, mm], co[b, mm]
            dsingle = is_singleton_object(din.unsqueeze(0), S)[0]
            ch = (din != dou) & (din >= COLOR_OFFSET)
            for i in torch.nonzero(ch).flatten().tolist():
                col = int(din[i]) - COLOR_OFFSET
                dst = int(dou[i]) - COLOR_OFFSET
                cell_votes.setdefault(col, []).append(dst)
                obj_votes.setdefault((col, int(dsingle[i])), []).append(dst)
        modal = lambda lst: max(set(lst), key=lst.count)
        for i in torch.nonzero(changed_q[b]).flatten().tolist():
            col = int(q_in[b, i]) - COLOR_OFFSET
            lg = int(q_single[b, i])
            act = int(q_out[b, i]) - COLOR_OFFSET
            cp = modal(cell_votes[col]) if col in cell_votes else -1
            op = (modal(obj_votes[(col, lg)]) if (col, lg) in obj_votes
                  else (modal(cell_votes[col]) if col in cell_votes else -1))
            cell_c += float(cp == act)
            obj_c += float(op == act)
            tot += 1.0
    print(f"[OBJECT-CONDITIONED]  changed cells={tot:.0f}  "
          f"cell-only cov={100 * cell_c / tot:.1f}%  object-conditioned cov={100 * obj_c / tot:.1f}%")
    assert obj_c / tot > cell_c / tot + 0.05, "object-conditioning must beat cell-only on conditional recolours"
    print("object-conditioning demo PASS (recovers the conditional recolour cell-colour cannot).")


# ===========================================================================
# REAL-DATA KEY RANKING (Phase 2c follow-up).
# Size FAILED on the real held-out set (it only won the SYNTHETIC size task, then made the live
# VALUE worse: HEAD changed 53.5 < dpcc 57.6, color_exact 9.4 -> 0). So before building another
# conditioned VALUE we RANK candidate conditioning keys on the ACTUAL LODO held-out demos:
#   cell (baseline=dpcc) | size | singleton | position | neighbour
# Method = leave-one-demo-out. For each task & held-out demo h, build the recolour consensus
# (full-key -> modal dst; cell-colour -> modal dst for FALLBACK) from the OTHER demos, then score
# the held-out demo's CHANGED cells. A key only WINS if (lift over cell > 0) AND (resolved% high):
# a finer key that mostly falls back to cell-colour shows ~0 lift with low resolved%.
# ===========================================================================
def _core_knowledge_self_check():
    """Build-the-check-first: validate the core-knowledge keys COMPUTE the concept (counting / topology)."""
    S = GRID_SIDE
    g = torch.zeros(S, S, dtype=torch.long)
    g[2, 2] = 3 + COLOR_OFFSET; g[2, 3] = 3 + COLOR_OFFSET              # ONE 2-cell colour-3 object
    g[5, 5] = 4 + COLOR_OFFSET; g[8, 8] = 4 + COLOR_OFFSET              # TWO separate colour-4 objects
    ck = _count_key(g.view(1, -1), S)[0]
    assert int(ck[2 * S + 2]) == 1 and int(ck[5 * S + 5]) == 2 and int(ck[8 * S + 8]) == 2, \
        f"count_key wrong: {(int(ck[2*S+2]), int(ck[5*S+5]), int(ck[8*S+8]))}"
    gt = torch.zeros(S, S, dtype=torch.long)
    gt[10:15, 10:15] = 4 + COLOR_OFFSET                                # solid colour-4 box
    gt[12, 12] = 0 + COLOR_OFFSET                                      # a colour-0 cell enclosed inside it
    tk = _topology_key(gt.view(1, -1), S)[0]
    assert int(tk[12 * S + 12]) == 1, "topology: enclosed cell must be INSIDE (1)"
    assert int(tk[10 * S + 10]) == 0, "topology: boundary box cell must be OUTSIDE (0)"
    print("core-knowledge self-check PASS (counting counts same-colour objects; topology marks enclosed vs boundary).")


def conditional_recolour_subset(batches, side, thresh=0.90):
    """Sub-list of (ci,co,dv) tasks that are CONDITIONAL recolour-in-place: cells stay (same grid shape,
    colour multiset changes) for most demos, AND the pooled cell-colour map self-consistency < thresh
    (so a plain recolour does NOT fit). This is the ~26%-of-all bucket where a shape/rank/relational key
    could win -- ranking globally washes it out under the 58% clean recolours."""
    sub = []
    for ci, co, dv in batches:
        for b in range(ci.shape[0]):
            valid = [m for m in range(ci.shape[1]) if bool(dv[b, m])]
            if len(valid) < 2:
                continue
            r = _task_recolour_consistency(ci[b], co[b], dv[b])
            if r is None or r[0] >= thresh:
                continue                                   # no changed cells, or a CLEAN recolour
            inplace = 0
            for m in valid:
                gi, go = _extract_grid(ci[b, m], side), _extract_grid(co[b, m], side)
                if gi is None or go is None or gi.shape != go.shape:
                    continue
                hi = torch.bincount(gi.flatten(), minlength=COLOR_OFFSET + N_COLORS)
                ho = torch.bincount(go.flatten(), minlength=COLOR_OFFSET + N_COLORS)
                if not bool((hi == ho).all()):             # same shape + multiset CHANGED = recolour-in-place
                    inplace += 1
            if inplace >= max(1, len(valid) // 2):
                sub.append((ci[b:b + 1], co[b:b + 1], dv[b:b + 1]))
    return sub


def score_key(batches, key_fn, side):
    """Leave-one-demo-out changed-cell coverage of a conditioning key over a LIST of (ci,co,dv)
    batches (ci/co [B,M,L]; dv [B,M] demo-valid). Returns (coverage%, resolved%, n_changed)."""
    correct = total = resolved = 0.0
    modal = lambda lst: max(set(lst), key=lst.count)
    for ci, co, dv in batches:
        B, M, _ = ci.shape
        ctx = torch.stack([key_fn(ci[:, m], side) for m in range(M)], dim=1)   # [B,M,L]
        for b in range(B):
            valid_m = [m for m in range(M) if bool(dv[b, m])]
            if len(valid_m) < 2:
                continue
            for h in valid_m:
                full_votes, cell_votes = {}, {}
                for m in valid_m:
                    if m == h:
                        continue
                    xin, yout, kx = ci[b, m], co[b, m], ctx[b, m]
                    ch = (xin != yout) & (xin >= COLOR_OFFSET) & (yout >= COLOR_OFFSET)
                    for i in torch.nonzero(ch).flatten().tolist():
                        cidx, dst = int(xin[i]) - COLOR_OFFSET, int(yout[i]) - COLOR_OFFSET
                        full_votes.setdefault((cidx, int(kx[i])), []).append(dst)
                        cell_votes.setdefault(cidx, []).append(dst)
                xin, yout, kx = ci[b, h], co[b, h], ctx[b, h]
                ch = (xin != yout) & (xin >= COLOR_OFFSET) & (yout >= COLOR_OFFSET)
                for i in torch.nonzero(ch).flatten().tolist():
                    cidx, act = int(xin[i]) - COLOR_OFFSET, int(yout[i]) - COLOR_OFFSET
                    fk = (cidx, int(kx[i]))
                    if fk in full_votes:
                        pred = modal(full_votes[fk]); resolved += 1.0
                    elif cidx in cell_votes:
                        pred = modal(cell_votes[cidx])           # fall back to cell-colour consensus
                    else:
                        pred = -1
                    correct += float(pred == act)
                    total += 1.0
    t = max(total, 1.0)
    return 100.0 * correct / t, 100.0 * resolved / t, total


def rank_keys(batches, side, title, keys=None):
    """Print coverage/resolved/lift for every key. A non-cell key WINS only if lift>1pt AND it is
    actually USED (resolved%>20) -- otherwise the apparent lift is just cell-colour fallback."""
    keys = keys if keys is not None else KEYS
    print(f"\n[KEY RANKING] {title}  (leave-one-demo-out, changed-cell coverage)")
    print(f"  {'key':<12}{'coverage%':>11}{'resolved%':>11}{'lift_vs_cell':>14}")
    rows, base = [], 0.0
    for name, fn in keys.items():
        cov, res, tot = score_key(batches, fn, side)
        rows.append((name, cov, res, tot))
        if name == "cell":
            base = cov
    winner = None
    for name, cov, res, tot in rows:
        lift = cov - base
        win = name != "cell" and lift > 1.0 and res > 20.0
        if win and (winner is None or lift > winner[1]):
            winner = (name, lift)
        print(f"  {name:<12}{cov:>11.1f}{res:>11.1f}{lift:>+14.1f}{'   <-- WINS' if win else ''}")
    n = int(rows[0][3]) if rows else 0
    print(f"  (changed cells scored: {n})  "
          f"=> {'best key = ' + winner[0] if winner else 'NO key beats cell-colour'}")
    return rows, winner


def _task_recolour_consistency(ci_b, co_b, dv_b):
    """One task (ci_b/co_b [M,L], dv_b [M]) -> (self_consistency, n_changed) or None if no changed
    cells. Pool ALL valid demos, fit ONE cell-colour map (in-colour -> modal out-colour over changed
    cells), return the fraction of all changed cells that single map explains. High = a plain per-cell
    recolour fits the WHOLE task (the colour head's addressable family); low = conditional recolour OR
    a geometric/move task where in->out at a fixed index is incoherent."""
    votes, cells = {}, []
    for m in range(ci_b.shape[0]):
        if not bool(dv_b[m]):
            continue
        xin, yout = ci_b[m], co_b[m]
        ch = (xin != yout) & (xin >= COLOR_OFFSET) & (yout >= COLOR_OFFSET)
        for i in torch.nonzero(ch).flatten().tolist():
            a, d = int(xin[i]) - COLOR_OFFSET, int(yout[i]) - COLOR_OFFSET
            votes.setdefault(a, []).append(d)
            cells.append((a, d))
    if not cells:
        return None
    modal = {a: max(set(v), key=v.count) for a, v in votes.items()}
    ok = sum(1 for a, d in cells if modal[a] == d)
    return ok / len(cells), len(cells)


def task_type_split(batches, side, thresh=0.90):
    """Split tasks into recolour-CONSISTENT (a single cell-colour map self-explains >= thresh of the
    task's changed cells) vs the rest, then report LODO held-out cell-colour coverage on EACH subset.
    Sizes the routing prize: if the recolour subset is near-100%, the colour head is already maxed on
    its addressable tasks and the entire remaining gap is task-family ROUTING, not a finer VALUE."""
    recolour, other = [], []
    n_rc = n_ot = ch_rc = ch_ot = 0
    for ci, co, dv in batches:
        for b in range(ci.shape[0]):
            res = _task_recolour_consistency(ci[b], co[b], dv[b])
            if res is None:
                continue
            cons, nch = res
            tup = (ci[b:b + 1], co[b:b + 1], dv[b:b + 1])
            if cons >= thresh:
                recolour.append(tup); n_rc += 1; ch_rc += nch
            else:
                other.append(tup); n_ot += 1; ch_ot += nch
    cell = KEYS["cell"]
    cov_rc = score_key(recolour, cell, side)[0] if recolour else float("nan")
    cov_ot = score_key(other, cell, side)[0] if other else float("nan")
    cov_all = score_key(batches, cell, side)[0]
    tot_ch = max(ch_rc + ch_ot, 1)
    print(f"\n[TASK-TYPE SPLIT] self-consistency threshold = {thresh:.2f}  (cell-colour LODO coverage per subset)")
    print(f"  {'subset':<22}{'tasks':>7}{'chg-cell share':>16}{'cell coverage%':>16}")
    print(f"  {'recolour-consistent':<22}{n_rc:>7}{100 * ch_rc / tot_ch:>15.1f}%{cov_rc:>16.1f}")
    print(f"  {'other (cond/geometric)':<22}{n_ot:>7}{100 * ch_ot / tot_ch:>15.1f}%{cov_ot:>16.1f}")
    print(f"  {'ALL':<22}{n_rc + n_ot:>7}{100.0:>15.1f}%{cov_all:>16.1f}")
    print(f"  => routing prize: {100 * n_ot / max(n_rc + n_ot, 1):.0f}% of tasks are NOT clean recolours "
          f"(colour head should stay shut on them); the colour lane is ~{cov_rc:.0f}% on the rest.")
    return {"n_recolour": n_rc, "n_other": n_ot, "cov_recolour": cov_rc, "cov_other": cov_ot}


# ===========================================================================
# GEOMETRIC (tier-B) OPERATION DETECTORS. The recolour gate routes ~53% of tasks; the rejected ~45%
# are conditional recolours OR GEOMETRY (reflect/rotate/shift/tile). Geometric ops are PARAMETER-FREE
# (or a single demo-agreed param) and EXACT -> a detector that fires also SOLVES the task deterministically,
# no neural head. This sizes that prize: per held-out demo, identify the op from the SUPPORT demos and
# check it reconstructs the held-out output EXACTLY. Grids live at an arbitrary canvas offset (aug), so
# every test runs on the colour-bbox CROP (offset normalised away).
# ===========================================================================
def geometric_detector_diagnostic(batches, side, recolour_thresh=0.90):
    """Per held-out demo: (a) ORACLE = does any geometric op fit the held-out in->out directly (upper
    bound); (b) SUPPORT-ROUTED = the op+param AGREED by all support demos, applied to the held-out input,
    must reconstruct the held-out output EXACTLY (the realistic router). Also reports the rescue rate on
    the recolour-REJECTED holdouts -- how much of the 45% geometry deterministically solves."""
    ops = [_DihedralOp(), _TranslateOp(), _TileOp()]
    modal = lambda l: max(set(l), key=l.count)
    tot = oracle = routed = rej_tot = rej_routed = 0
    per_op = {op.name: 0 for op in ops}
    for ci, co, dv in batches:
        for b in range(ci.shape[0]):
            valid = [m for m in range(ci.shape[1]) if bool(dv[b, m])]
            if len(valid) < 2:
                continue
            gin = {m: _extract_grid(ci[b, m], side) for m in valid}
            gout = {m: _extract_grid(co[b, m], side) for m in valid}
            for h in valid:
                if gin[h] is None or gout[h] is None:
                    continue
                sup = [m for m in valid if m != h]
                votes, sc = {}, []
                for m in sup:
                    xin, yout = ci[b, m], co[b, m]
                    chm = (xin != yout) & (xin >= COLOR_OFFSET) & (yout >= COLOR_OFFSET)
                    for i in torch.nonzero(chm).flatten().tolist():
                        a, d = int(xin[i]) - COLOR_OFFSET, int(yout[i]) - COLOR_OFFSET
                        votes.setdefault(a, []).append(d); sc.append((a, d))
                rejected = (sum(1 for a, d in sc if modal(votes[a]) == d) / len(sc)) < recolour_thresh if sc else True
                tot += 1
                rej_tot += int(rejected)
                orc = any(op.candidates(gin[h], gout[h]) for op in ops)
                oracle += int(orc)
                rtd_op = None
                for op in ops:
                    inter, ok = None, True
                    for m in sup:
                        c = op.candidates(gin[m], gout[m])
                        inter = c if inter is None else (inter & c)
                        if not inter:
                            ok = False; break
                    if ok and inter:
                        pred = op.apply(gin[h], sorted(inter)[0])
                        if pred is not None and pred.shape == gout[h].shape and bool((pred == gout[h]).all()):
                            rtd_op = op.name; break
                if rtd_op:
                    routed += 1; per_op[rtd_op] += 1
                    rej_routed += int(rejected)
    pc = lambda n, d: 100.0 * n / max(d, 1)
    print(f"\n[GEOMETRIC DETECTORS] deterministic exact-reconstruction (dihedral / translate / tile)")
    print(f"  total holdouts scored: {tot}")
    print(f"  oracle (any op fits held-out):        {oracle:>4}  ({pc(oracle, tot):.1f}%)")
    print(f"  support-routed (op agreed by support): {routed:>4}  ({pc(routed, tot):.1f}%)   "
          f"per-op: " + " ".join(f"{k}={v}" for k, v in per_op.items()))
    print(f"  of recolour-REJECTED holdouts ({rej_tot}): geometry rescues {rej_routed} "
          f"({pc(rej_routed, rej_tot):.1f}%)  <-- deterministic-applier prize on the non-recolour half")
    return {"oracle": oracle, "routed": routed, "rej_routed": rej_routed, "per_op": per_op}


# ====================== S2: primitive library + propose-and-verify selector =================
# The 1000-step run (R17) RETIRED the z_H colour-forcing keystone: forcing the relational transform
# into a 7M z_H failed the kill-gate (FORCE changed ~10 << dpcc 51) -- a per-cell linear readout of
# z_H cannot represent conditional logic. So the transform is represented EXPLICITLY, as a library of
# general primitives SELECTED per task by cross-demo reconstruction (the 'general router' of the North
# Star, rule 2: learned/reconstruction-scored, never a hand-coded if task==X). This probe MEASURES the
# headroom of that approach with NO TRAINING, before the differentiable composer (A) is built: per
# held-out demo, fit each primitive on the SUPPORT demos and check if it reconstructs the held-out
# output EXACTLY. ROUTED = realizable (support-agreed op transfers); ORACLE = ceiling (any op fits the
# held-out directly). recolor is full-canvas; dihedral/translate/tile are bbox (the existing ops).
def _conditional_geo_solve(gin, gout, sup, h, geo_ops):
    """CONDITIONAL PARAMETER ENGINE (Flavor C): Fits a map from a grid-level property (e.g. background colour,
    number of foreground objects) -> Geometric Parameter.
    Unlike _geo_routed which requires the SAME parameter across all demos, this allows Demo 1 to rotate 90 and
    Demo 2 to rotate 180, PROVIDED the rotation angle is perfectly predicted by the grid's property."""
    if gin[h] is None or gout[h] is None:
        return None
    
    def grid_keys(m):
        g = gin[m]
        bg = int(_mode(g))
        cols = tuple(sorted(int(c) for c in g.unique() if int(c) != bg))
        # Handle cases where bg might not be in the grid if mode fails
        fg_labels = _components_2d_adj(g, bg=bg).unique()
        return {
            "bg": bg,
            "fg_colors": cols,
            "fg_count": len(fg_labels) - 1 if int(bg) in g else len(fg_labels)
        }
    
    for key_name in ("bg", "fg_colors", "fg_count"):
        for op in geo_ops:
            demo_params = {}
            possible = True
            for m in sup:
                cands = op.candidates(gin[m], gout[m])
                if not cands:
                    possible = False
                    break
                demo_params[m] = cands
            
            if not possible:
                continue
                
            key_to_cands = {}
            for m in sup:
                k = grid_keys(m)[key_name]
                if k not in key_to_cands:
                    key_to_cands[k] = demo_params[m]
                else:
                    key_to_cands[k] = key_to_cands[k] & demo_params[m]
                    
            if any(not cands for cands in key_to_cands.values()):
                continue
                
            final_map = {k: sorted(cands)[0] for k, cands in key_to_cands.items()}
            
            h_key = grid_keys(h)[key_name]
            if h_key not in final_map:
                continue
                
            pred = op.apply(gin[h], final_map[h_key])
            if pred is not None and pred.shape == gout[h].shape and bool((pred == gout[h]).all()):
                if len(set(final_map.values())) > 1:
                    return "cond_geo"
    return None


def _conditional_geo_self_check():
    """Gemini: conditional geo solver test."""
    gi1 = torch.full((3, 3), COLOR_OFFSET + 1, dtype=torch.long); gi1[0, 1] = COLOR_OFFSET + 9
    gi2 = torch.full((3, 3), COLOR_OFFSET + 2, dtype=torch.long); gi2[0, 1] = COLOR_OFFSET + 9
    gi3 = torch.full((3, 3), COLOR_OFFSET + 2, dtype=torch.long); gi3[0, 1] = COLOR_OFFSET + 9
    go1 = torch.flip(gi1, (1,))
    go2 = torch.rot90(gi2, 2, (0, 1))
    go3 = torch.rot90(gi3, 2, (0, 1))
    gin = {0: gi1, 1: gi2, 2: gi3}
    gout = {0: go1, 1: go2, 2: go3}
    from __main__ import _DihedralOp
    do = _DihedralOp()
    assert _geo_routed(gin, gout, [0, 1], 2, [do]) is None, "_geo_routed must refuse conditional geometry"
    assert _conditional_geo_solve(gin, gout, [0, 1], 2, [do]) is not None, "cond geo must solve it"
    print("conditional geo self-check PASS.")


def _distance_recolor_self_check():
    """Gemini: concentric rings self check"""
    L = 10
    ti1 = torch.full((L, L), COLOR_OFFSET + 0, dtype=torch.long)
    ti1[0, :] = COLOR_OFFSET + 5; ti1[-1, :] = COLOR_OFFSET + 5
    ti1[:, 0] = COLOR_OFFSET + 5; ti1[:, -1] = COLOR_OFFSET + 5
    to1 = ti1.clone()
    to1[1:-1, 1:-1] = COLOR_OFFSET + 2
    to1[2:-2, 2:-2] = COLOR_OFFSET + 5
    to1[3:-3, 3:-3] = COLOR_OFFSET + 0
    ti2 = torch.full((8, 8), COLOR_OFFSET + 0, dtype=torch.long)
    ti2[0, :] = COLOR_OFFSET + 5; ti2[-1, :] = COLOR_OFFSET + 5
    ti2[:, 0] = COLOR_OFFSET + 5; ti2[:, -1] = COLOR_OFFSET + 5
    to2 = ti2.clone()
    to2[1:-1, 1:-1] = COLOR_OFFSET + 2
    to2[2:-2, 2:-2] = COLOR_OFFSET + 5
    to2[3:-3, 3:-3] = COLOR_OFFSET + 0
    gin = {0: ti1, 1: ti2}
    gout = {0: to1, 1: to2}
    assert _distance_recolor_solve(gin, gout, [0], 1) is True
    print("distance recolor self-check PASS.")

def _legend_match_self_check():
    """Gemini: legend match self check"""
    ti1 = torch.full((5, 5), COLOR_OFFSET + 0, dtype=torch.long)
    ti1[1, 1] = COLOR_OFFSET + 1
    ti1[1, 2] = COLOR_OFFSET + 7
    to1 = ti1.clone()
    to1[1, 1] = COLOR_OFFSET + 3
    to1[1, 2] = COLOR_OFFSET + 0
    ti2 = torch.full((5, 5), COLOR_OFFSET + 0, dtype=torch.long)
    ti2[1, 1] = COLOR_OFFSET + 1
    ti2[1, 2] = COLOR_OFFSET + 7; ti2[2, 2] = COLOR_OFFSET + 7
    to2 = ti2.clone()
    to2[1, 1] = COLOR_OFFSET + 6
    to2[1, 2] = COLOR_OFFSET + 0; to2[2, 2] = COLOR_OFFSET + 0
    ti3 = torch.full((5, 5), COLOR_OFFSET + 0, dtype=torch.long)
    ti3[3, 3] = COLOR_OFFSET + 1
    ti3[3, 4] = COLOR_OFFSET + 7
    to3 = ti3.clone()
    to3[3, 3] = COLOR_OFFSET + 3
    to3[3, 4] = COLOR_OFFSET + 0
    gin = {0: ti1, 1: ti2, 2: ti3}
    gout = {0: to1, 1: to2, 2: to3}
    assert _legend_match_solve(gin, gout, [0, 1], 2) is True
    print("legend match self-check PASS.")


def _s2_self_check():
    """Build-the-check-first: recolor must SOLVE a clean per-colour map and REFUSE a conditional one."""
    L = GRID_LEN
    ti = torch.zeros(L, dtype=torch.long); ti[:30] = 3 + COLOR_OFFSET; ti[30:60] = 4 + COLOR_OFFSET
    to = torch.zeros(L, dtype=torch.long); to[:30] = 7 + COLOR_OFFSET; to[30:60] = 5 + COLOR_OFFSET
    rm = _fit_recolor_map([(ti, to)])
    assert rm is not None and bool((_apply_recolor(ti, rm) == to).all()), "recolor must solve a clean map"
    to2 = to.clone(); to2[:15] = 8 + COLOR_OFFSET            # input colour 3 -> 8 AND 7 (conditional)
    assert _fit_recolor_map([(ti, to2)]) is None, "recolor must REFUSE a conditional (3->7 and 3->8)"
    # NEIGHBOUR-CONDITIONED: 3 adjacent-to-5 -> 7 ; 3 adjacent-to-6 -> 8 (flat REFUSES, neighbour SOLVES)
    S = GRID_SIDE
    gi = torch.zeros(S, S, dtype=torch.long); go = torch.zeros(S, S, dtype=torch.long)
    gi[2, 2] = 3 + COLOR_OFFSET; gi[3, 2] = 5 + COLOR_OFFSET     # domino A: 3 over 5
    go[2, 2] = 7 + COLOR_OFFSET; go[3, 2] = 5 + COLOR_OFFSET     #   -> the 3 becomes 7
    gi[2, 6] = 3 + COLOR_OFFSET; gi[3, 6] = 6 + COLOR_OFFSET     # domino B: 3 over 6
    go[2, 6] = 8 + COLOR_OFFSET; go[3, 6] = 6 + COLOR_OFFSET     #   -> the 3 becomes 8
    nti, nto = gi.view(-1), go.view(-1)
    assert _fit_recolor_map([(nti, nto)]) is None, "flat recolor must REFUSE the neighbour-conditioned rule"
    nm = _fit_neighbour_recolor_map([(nti, nto)], S)
    assert nm is not None and bool((_apply_neighbour_recolor(nti, nm, S) == nto).all()), \
        "neighbour_recolor must SOLVE the neighbour-conditioned rule that flat recolor refused"
    cm = _fit_combined_recolor_map([(nti, nto)], S)
    assert cm is not None and bool((_apply_combined_recolor(nti, cm, S) == nto).all()), \
        "combined_recolor must SOLVE the neighbour-conditioned rule too (its joint key includes neighbour)"
    print("S2 selector self-check PASS (recolor solves clean / refuses conditional; "
          "neighbour_recolor + combined_recolor solve neighbour-conditioned).")


def _compose_self_check():
    """Build-the-check-first: a scale-THEN-recolor task NO single op can solve, but compose2 can."""
    geo_ops = [_DihedralOp(), _TranslateOp(), _TileOp(), _CropOp(), _ScaleOp()]
    bases = [torch.tensor([[3, 4], [5, 3]]) + COLOR_OFFSET,
             torch.tensor([[4, 3], [3, 5]]) + COLOR_OFFSET,
             torch.tensor([[5, 5], [4, 3]]) + COLOR_OFFSET]
    gin, gout = {}, {}
    for i, bs in enumerate(bases):
        up = bs.repeat_interleave(2, 0).repeat_interleave(2, 1)              # op1: scale up 2
        rec = up.clone(); rec[rec == 3 + COLOR_OFFSET] = 7 + COLOR_OFFSET     # op2: recolor 3->7
        gin[i], gout[i] = bs, rec
    assert _geo_routed(gin, gout, [0, 1], 2, geo_ops) is None, "no single op solves scale-then-recolor"
    assert _compose2_solve(gin, gout, [0, 1], 2, geo_ops) == ["scale", "recolor"], \
        "compose2 must solve scale->recolor and return the executable recipe"
    print("compose2 self-check PASS (scale->recolor solved by composition, refused by every 1-op).")


def _parser_self_check():
    """Build-the-check-first: (a) colour-agnostic CC keeps a checkerboard whole where same-colour CC shatters
    it; (b) object-recolor (rank) solves largest-object->colour where per-cell recolor REFUSES."""
    cb = torch.zeros(3, 3, dtype=torch.long)
    for r in range(3):
        for c in range(3):
            cb[r, c] = (3 if (r + c) % 2 == 0 else 4) + COLOR_OFFSET
    n_same = len({int(x) for x in _parse_same(cb).reshape(-1).tolist()})
    n_adj = len({int(x) for x in _components_2d_adj(cb, bg=-1).reshape(-1).tolist() if int(x) >= 0})
    assert n_same > 1, f"same-colour CC must shatter the checkerboard (got {n_same})"
    assert n_adj == 1, f"colour-agnostic CC must keep the checkerboard whole (got {n_adj})"
    gin_t, gout_t = {}, {}
    for i, s in enumerate((3, 2, 4)):
        g = torch.full((8, 8), COLOR_OFFSET, dtype=torch.long)        # bg = colour 0
        g[0:s, 0:s] = 3 + COLOR_OFFSET                                # big square colour 3, size s
        g[7, 7] = 3 + COLOR_OFFSET                                    # small object colour 3, size 1
        o = g.clone(); o[0:s, 0:s] = 7 + COLOR_OFFSET                 # big -> 7 ; small stays 3
        gin_t[i], gout_t[i] = g, o
    assert _fit_recolor_map([(gin_t[m].reshape(-1), gout_t[m].reshape(-1)) for m in (0, 1)]) is None, \
        "per-cell recolor must REFUSE largest-object recolor (colour 3 -> both 7 and 3)"
    labs = {m: _parse_same(gin_t[m]) for m in (0, 1, 2)}
    assert _object_recolor_solve(gin_t, gout_t, [0, 1], 2, labs, "rank"), \
        "object-recolor (rank) must SOLVE largest-object->7"
    print("parser self-check PASS (colour-agnostic CC keeps checkerboard whole; object-recolor solves largest->X).")


def _symfill_self_check():
    """Build-the-check-first for the two new families: SymmetrizeOp restores an occluded mirror-symmetric grid;
    FillOp floods an enclosed background region (and must NOT fire when nothing is enclosed); shape_d4 is
    rotation-invariant where the plain shape hash is not."""
    so, fo = _SymmetrizeOp(), _FillOp()
    # --- symmetrize: a flipH-symmetric base, right column occluded with colour 9 -> restored from the mirror ---
    base = torch.tensor([[3, 4, 4, 3], [5, 6, 6, 5], [3, 3, 3, 3]], dtype=torch.long) + COLOR_OFFSET
    occ = base.clone(); k = 9 + COLOR_OFFSET; occ[0, 3] = k; occ[1, 3] = k
    assert ("flipH", int(k)) in so.candidates(occ, base), "symmetrize must detect flipH + occlusion colour"
    assert bool((so.apply(occ, ("flipH", int(k))) == base).all()), "symmetrize must restore the mirror"
    # --- fill: a ring of colour 4 around a single enclosed bg cell -> filled with colour 7 ---
    g = torch.full((5, 5), COLOR_OFFSET, dtype=torch.long)            # bg = colour 0
    g[1, 1:4] = 4 + COLOR_OFFSET; g[3, 1:4] = 4 + COLOR_OFFSET
    g[1:4, 1] = 4 + COLOR_OFFSET; g[1:4, 3] = 4 + COLOR_OFFSET        # closed ring; (2,2) stays bg = enclosed
    o = g.clone(); o[2, 2] = 7 + COLOR_OFFSET
    assert (7 + COLOR_OFFSET) in fo.candidates(g, o), "fill must detect the enclosed bg + fill colour"
    assert bool((fo.apply(g, 7 + COLOR_OFFSET) == o).all()), "fill must flood the enclosed cell"
    assert not fo.candidates(g, g), "fill must NOT fire when nothing is enclosed/changed"
    # --- shape_d4: an L and its 90-degree rotation share a canonical hash; the plain shape hash does not ---
    L = [(0, 0), (1, 0), (2, 0), (2, 1)]; L90 = [(0, 0), (0, 1), (0, 2), (1, 0)]
    assert _d4_canon_hash(L) == _d4_canon_hash(L90), "shape_d4 must be rotation-invariant"
    _plain = lambda offs: hash(tuple(sorted(offs))) & 0x7fffffff
    assert _plain(L) != _plain(L90), "plain shape hash must distinguish the rotation (sanity for the dual key)"
    print("symmetrize/fill/shape_d4 self-check PASS.")


def _relational_self_check():
    """U5: a RELATIONAL rule -- the object INSIDE a frame -> 7, an identical-colour FREE object stays -- which
    per-cell recolor REFUSES (one colour, two fates) but the `inside` relational key SOLVES."""
    def draw_frame(g, r0, r1, c0, c1, col):
        g[r0, c0:c1 + 1] = col; g[r1, c0:c1 + 1] = col
        g[r0:r1 + 1, c0] = col; g[r0:r1 + 1, c1] = col
    def mk(frame, inside, free):
        g = torch.full((12, 12), COLOR_OFFSET, dtype=torch.long)
        draw_frame(g, *frame, 3 + COLOR_OFFSET)
        g[inside[0]:inside[1] + 1, inside[2]:inside[3] + 1] = 4 + COLOR_OFFSET
        g[free[0]:free[1] + 1, free[2]:free[3] + 1] = 4 + COLOR_OFFSET
        o = g.clone(); o[inside[0]:inside[1] + 1, inside[2]:inside[3] + 1] = 7 + COLOR_OFFSET
        return g, o
    specs = [((1, 7, 1, 7), (3, 5, 3, 5), (9, 10, 9, 10)),
             ((1, 9, 1, 5), (3, 7, 2, 4), (10, 11, 9, 11)),
             ((2, 6, 5, 11), (3, 5, 7, 9), (9, 11, 1, 2))]
    gin_t, gout_t = {}, {}
    for i, (fr, ib, fb) in enumerate(specs):
        gin_t[i], gout_t[i] = mk(fr, ib, fb)
    labs = {i: _parse_same(gin_t[i]) for i in range(3)}
    assert _fit_recolor_map([(gin_t[m].reshape(-1), gout_t[m].reshape(-1)) for m in (0, 1)]) is None, \
        "per-cell recolor must REFUSE (colour 4 -> both 7 (inside) and 4 (free))"
    assert _object_recolor_solve(gin_t, gout_t, [0, 1], 2, labs, "inside"), \
        "the `inside` relational key must SOLVE the inside-object recolor the per-cell map cannot"
    print("relational self-check PASS (inside-object recolor solved by `inside`, refused by per-cell).")


def _extract_panel_self_check():
    """U5/extra: extract-object crops the selected object to its bbox; _PanelOp combines two separated panels."""
    eo, po = _ExtractObjectOp(), _PanelOp()
    gi = torch.full((8, 8), COLOR_OFFSET, dtype=torch.long)
    gi[1:4, 1:4] = 4 + COLOR_OFFSET; gi[6, 6] = 5 + COLOR_OFFSET; gi[0, 7] = 5 + COLOR_OFFSET
    go = torch.full((3, 3), 4 + COLOR_OFFSET, dtype=torch.long)
    assert "largest" in eo.candidates(gi, go), "extract-object must detect largest-object extraction"
    assert bool((eo.apply(gi, "largest") == go).all()), "extract-object must reproduce the bbox crop"
    gp = torch.full((3, 7), COLOR_OFFSET, dtype=torch.long); gp[:, 3] = 5 + COLOR_OFFSET
    gp[0, 0] = 4 + COLOR_OFFSET; gp[1, 1] = 4 + COLOR_OFFSET
    gp[0, 4] = 4 + COLOR_OFFSET; gp[2, 6] = 4 + COLOR_OFFSET
    op = torch.full((3, 3), COLOR_OFFSET, dtype=torch.long)
    op[0, 0] = 8 + COLOR_OFFSET; op[1, 1] = 8 + COLOR_OFFSET; op[2, 2] = 8 + COLOR_OFFSET
    assert ("or", 8 + COLOR_OFFSET) in po.candidates(gp, op), "panel OR-combine must be detected"
    assert bool((po.apply(gp, ("or", 8 + COLOR_OFFSET)) == op).all()), "panel apply must reproduce the OR-combine"
    print("extract/panel self-check PASS (largest-object crop; two-panel OR-combine).")


def _relrecolor_self_check():
    """U11: copy-by-relation -- every object takes the LARGEST object's colour (which varies per demo) -- which
    a key->fixed-colour map (object_recolor by rank) REFUSES, but the `largest` copy-by-relation SOLVES."""
    def mk(big_colour, big_box, smalls):
        g = torch.full((12, 12), COLOR_OFFSET, dtype=torch.long)
        g[big_box[0]:big_box[1] + 1, big_box[2]:big_box[3] + 1] = big_colour + COLOR_OFFSET
        for box, col in smalls:
            g[box[0]:box[1] + 1, box[2]:box[3] + 1] = col + COLOR_OFFSET
        o = g.clone()
        for box, _ in smalls:
            o[box[0]:box[1] + 1, box[2]:box[3] + 1] = big_colour + COLOR_OFFSET   # smalls -> the big object's colour
        return g, o
    specs = [(4, (1, 4, 1, 4), [((1, 2, 7, 8), 5), ((7, 8, 2, 3), 6)]),          # big colour 4
             (8, (6, 10, 6, 10), [((1, 2, 1, 2), 5), ((1, 2, 9, 10), 3)]),       # big colour 8
             (3, (2, 6, 2, 6), [((9, 10, 9, 10), 5), ((9, 10, 1, 2), 7)])]       # big colour 3 (held-out)
    gin_t, gout_t = {}, {}
    for i, (bc, bb, sm) in enumerate(specs):
        gin_t[i], gout_t[i] = mk(bc, bb, sm)
    labs = {i: _parse_same(gin_t[i]) for i in range(3)}
    assert _object_relrecolor_solve(gin_t, gout_t, [0, 1], 2, labs, "largest"), \
        "copy-by-relation `largest` must SOLVE everything-takes-the-largest-object's-colour"
    assert not _object_recolor_solve(gin_t, gout_t, [0, 1], 2, labs, "rank"), \
        "object_recolor(rank) must REFUSE (target colour is the largest's, varies per demo -> not a fixed key map)"
    print("copy-by-relation self-check PASS (largest-colour propagation solved by relation, refused by key-map).")


def _set_cover_self_check():
    """U6: a TWO-RULE task NO single key can solve -- BORDER-touching objects -> 7 AND FRAME-contained objects
    -> 8, plus a FREE object that STAYS. All three foreground objects share ONE colour and ONE size, so
    colour/rank/shape cannot separate them; and each single key alone lumps two of the three groups
    (touch_border=0 mixes contained+free; inside=0 mixes border+free). Only the UNION of two safe clauses
    (touch_border=1 -> 7) + (inside=2 -> 8) reconstructs. Set-cover SOLVES; object_recolor by any single key
    REFUSES. (This is also the regression guard against accidental overfit clauses: same colour/size => none.)"""
    OC = 4 + COLOR_OFFSET
    def frame(g, r0, r1, c0, c1):
        g[r0, c0:c1 + 1] = 3 + COLOR_OFFSET; g[r1, c0:c1 + 1] = 3 + COLOR_OFFSET
        g[r0:r1 + 1, c0] = 3 + COLOR_OFFSET; g[r0:r1 + 1, c1] = 3 + COLOR_OFFSET
    def mk(A, F, B, N):                                          # A=border 2x2, F=frame bbox, B=contained 2x2, N=free 2x2
        g = torch.full((14, 14), COLOR_OFFSET, dtype=torch.long)
        frame(g, *F)
        for (r, c) in (A, B, N):
            g[r:r + 2, c:c + 2] = OC                             # all foreground: colour 4, 2x2 (indistinguishable)
        o = g.clone()
        o[A[0]:A[0] + 2, A[1]:A[1] + 2] = 7 + COLOR_OFFSET        # border-touchers -> 7
        o[B[0]:B[0] + 2, B[1]:B[1] + 2] = 8 + COLOR_OFFSET        # frame-contained -> 8 ; free N stays
        return g, o
    specs = [((0, 0), (4, 9, 4, 9), (6, 6), (11, 11)),
             ((12, 0), (3, 8, 6, 11), (5, 8), (10, 2)),
             ((0, 12), (5, 10, 2, 7), (7, 4), (11, 10))]          # held-out: same roles, shifted positions
    gin_t, gout_t = {}, {}
    for i, (A, F, B, N) in enumerate(specs):
        gin_t[i], gout_t[i] = mk(A, F, B, N)
    labs = {i: _parse_same(gin_t[i]) for i in range(3)}
    assert _set_cover_solve(gin_t, gout_t, [0, 1], 2, labs), \
        "set-cover must SOLVE the two-rule (border->7 AND contained->8) task via clause-union"
    for k in ("colour", "touch_border", "inside"):
        assert not _object_recolor_solve(gin_t, gout_t, [0, 1], 2, labs, k), \
            f"object_recolor({k}) must REFUSE the two-rule task (each single key lumps two of the three groups)"
    print("set-cover self-check PASS (two-rule border/contained task solved by clause-union, refused by every single key).")


def _committed_self_check():
    """A1: the 2-attempt committed answer is FLOOR-SAFE. (a) op-solvable task -> attempt1 is EXACT on the held-out;
    (b) unsolvable task (same input, different outputs -> no rule) -> attempt2 (floor) == identity copy, so the
    committed answer is NEVER worse than the floor (committed >= floor by construction)."""
    geo_ops = [_DihedralOp(), _TranslateOp(), _TileOp(), _CropOp(), _ScaleOp()]
    # (a) largest-object -> 7 (object-recolor) : attempt1 must EXACTLY solve the held-out
    gin_t, gout_t = {}, {}
    for i, s in enumerate((3, 2, 4)):
        g = torch.full((8, 8), COLOR_OFFSET, dtype=torch.long)
        g[0:s, 0:s] = 3 + COLOR_OFFSET; g[7, 7] = 3 + COLOR_OFFSET
        o = g.clone(); o[0:s, 0:s] = 7 + COLOR_OFFSET
        gin_t[i], gout_t[i] = g, o
    labs = {m: _parse_same(gin_t[m]) for m in (0, 1, 2)}
    a1, a2 = _committed_solve(gin_t, gout_t, [0, 1], 2, geo_ops, labs)
    assert a1 is not None and a1.shape == gout_t[2].shape and bool((a1 == gout_t[2]).all()), \
        "committed attempt1 must EXACTLY solve the held-out object-recolor task"
    # (b) same input, DIFFERENT outputs -> no rule fits -> floor (attempt2) is the identity copy (never worse)
    u_in, u_out = {}, {}
    base = torch.full((6, 6), COLOR_OFFSET, dtype=torch.long); base[0, 0] = 3 + COLOR_OFFSET
    for i in range(3):
        u_in[i] = base.clone(); o = base.clone(); o[0, 0] = (4 + i) + COLOR_OFFSET; u_out[i] = o
    ulabs = {m: _parse_same(u_in[m]) for m in (0, 1, 2)}
    b1, b2 = _committed_solve(u_in, u_out, [0, 1], 2, geo_ops, ulabs)
    assert b2 is not None and bool((b2 == u_in[2]).all()), "floor (attempt2) must be the identity copy when no map fits"
    assert b1 is not None and bool((b1 == b2).all()), "with no rule, attempt1 falls back to the floor (never worse)"
    print("committed (A1) self-check PASS (op-solvable => attempt1 exact; unsolvable => attempt2 == floor, never worse).")


def _size_change_self_check():
    """B2: size-change ops induce the OUTPUT SHAPE. (a) fractal: a 3x3 input with one non-bg cell -> a 9x9 grid
    with ONE block = the input; (b) mirror_concat: a 1x3 row -> a 1x6 row = [in | flipH]."""
    fo, mo = _FractalOp(), _MirrorConcatOp()
    gi = torch.full((3, 3), COLOR_OFFSET, dtype=torch.long); gi[0, 0] = 4 + COLOR_OFFSET
    go = torch.full((9, 9), COLOR_OFFSET, dtype=torch.long); go[0:3, 0:3] = gi      # only block(0,0) fires
    assert "nonbg" in fo.candidates(gi, go), "fractal must detect the self-tiling (fire on non-bg)"
    assert bool((fo.apply(gi, "nonbg") == go).all()), "fractal must reproduce the 9x9 self-tile"
    row = torch.tensor([[3, 4, 5]], dtype=torch.long) + COLOR_OFFSET                # 1x3
    gor = torch.cat([row, torch.flip(row, [1])], dim=1)                             # 1x6 [3,4,5,5,4,3]
    assert ("h", "right") in mo.candidates(row, gor), "mirror_concat must detect [in | flipH]"
    assert bool((mo.apply(row, ("h", "right")) == gor).all()), "mirror_concat must reproduce the doubled grid"
    print("size-change self-check PASS (fractal self-tile induces 9x9; mirror_concat doubles + reflects).")


def _execute_recipe_self_check():
    """A2: a STORED macro [scale, recolor] re-fits its params on a NEW task and solves the held-out (the params
    are re-induced, never stored). Typed pruning: [recolor, scale] (SHAPE after COLOR) is rejected (None)."""
    geo_ops = [_DihedralOp(), _TranslateOp(), _TileOp(), _CropOp(), _ScaleOp()]
    gin_t, gout_t = {}, {}
    for i, pat in enumerate([[[3, 4], [4, 3]], [[3, 3], [4, 4]], [[4, 3], [3, 4]]]):
        g = torch.tensor(pat, dtype=torch.long) + COLOR_OFFSET                  # 2x2
        up = g.repeat_interleave(2, 0).repeat_interleave(2, 1)                  # scale up 2 -> 4x4
        o = up.clone()
        o[up == 3 + COLOR_OFFSET] = 7 + COLOR_OFFSET                           # recolor 3->7
        o[up == 4 + COLOR_OFFSET] = 8 + COLOR_OFFSET                           # recolor 4->8
        gin_t[i], gout_t[i] = g, o
    assert _execute_recipe_solve(gin_t, gout_t, [0, 1], 2, ["scale", "recolor"], geo_ops) is True, \
        "stored macro [scale, recolor] must re-fit params and solve the held-out scale-then-recolor task"
    assert _execute_recipe_solve(gin_t, gout_t, [0, 1], 2, ["recolor", "scale"], geo_ops) is None, \
        "typed pruning: [recolor, scale] (SHAPE after COLOR) must be rejected"
    from rule_library import RuleLibrary
    lib = RuleLibrary()
    labs = {m: _parse_same(gin_t[m]) for m in (0, 1, 2)}
    _committed_solve(gin_t, gout_t, [0, 1], 2, geo_ops, labs, rule_lib=lib)
    assert len(lib) == 0, "committed prediction must not write unverified recipes into RuleLibrary"
    print("execute-recipe (A2) self-check PASS (stored macro re-fits + solves; SHAPE-after-COLOR pruned).")


def s2_selector_probe(batches, pids, probe_path, side, rule_lib_path=None):
    """S2 HEADROOM: combined library {identity, recolor, dihedral, translate, tile} selected per task by
    cross-demo reconstruction. Per codex family: ROUTED = support-agreed op reconstructs the held-out
    EXACTLY (the color_exact-equivalent the composer could reach by selection); ORACLE = any op fits the
    held-out (ceiling). The number that decides if the differentiable composer is worth building."""
    geo_ops = _default_geo_ops()
    from rule_library import RuleLibrary
    rule_lib = RuleLibrary()                                 # Stage 5: store LODO-exact 2-op macros
    if rule_lib_path:                                        # A2: load macros discovered in PREVIOUS runs (compounding)
        rule_lib.load(rule_lib_path)
    id2hash, hash2fam = _load_family_map(probe_path)
    fam_ok = pids is not None and id2hash is not None and not any(p is None for p in pids)
    OPS = ["identity", "recolor", "neighbour_recolor", "combined_recolor", "object_recolor",
           "object_relrecolor", "set_cover", "cond_geo", "dihedral", "translate", "tile", "crop", "scale",
           "symmetrize", "fill", "extract_object", "panel", "fractal", "mirror_concat", "compose2", "macro"]
    agg = {}
    def slot(fam):
        return agg.setdefault(fam, {"hold": 0, "routed": 0, "oracle": 0, "committed": 0, "floor": 0,
                                    "op": {k: 0 for k in OPS}})
    for (ci, co, dv), pid in zip(batches, (pids if fam_ok else [None] * len(batches))):
        B, M, _ = ci.shape
        for b in range(B):
            valid = [m for m in range(M) if bool(dv[b, m])]
            if len(valid) < 2:
                continue
            if fam_ok:
                idx = int(pid[b]); hsh = id2hash[idx] if 0 <= idx < len(id2hash) else None
                fam = hash2fam.get(hsh, "other(not-in-140)")
            else:
                fam = "all"
            gin = {m: _extract_grid(ci[b, m], side) for m in valid}
            gout = {m: _extract_grid(co[b, m], side) for m in valid}
            _labs_same = {m: _parse_same(gin[m]) for m in valid if gin[m] is not None}
            _labs_adj = {m: _components_2d_adj(gin[m]) for m in valid if gin[m] is not None}
            for h in valid:
                ti_h, to_h = ci[b, h], co[b, h]
                sup = [m for m in valid if m != h]
                a = slot(fam); a["hold"] += 1
                a_all = slot("all") if fam != "all" else None
                if a_all is not None:
                    a_all["hold"] += 1
                # ----- ROUTED: support-agreed primitive reconstructs the held-out exactly -----
                solved = None
                if bool((ti_h == to_h).all()) and all(bool((ci[b, m] == co[b, m]).all()) for m in sup):
                    solved = "identity"
                if solved is None:
                    rm = _fit_recolor_map([(ci[b, m], co[b, m]) for m in sup])
                    if rm is not None and bool((_apply_recolor(ti_h, rm) == to_h).all()):
                        solved = "recolor"
                if solved is None:
                    nm = _fit_neighbour_recolor_map([(ci[b, m], co[b, m]) for m in sup], side)
                    if nm is not None and bool((_apply_neighbour_recolor(ti_h, nm, side) == to_h).all()):
                        solved = "neighbour_recolor"
                if solved is None:
                    cm = _fit_combined_recolor_map([(ci[b, m], co[b, m]) for m in sup], side)
                    if cm is not None and bool((_apply_combined_recolor(ti_h, cm, side) == to_h).all()):
                        solved = "combined_recolor"
                if solved is None:                                       # Stage-1: OBJECT-level recolor
                    for _lab in (_labs_same, _labs_adj):                 # same-colour CC | colour-agnostic CC
                        # R31+R32: property AND relational KEYS both routed 0 -> object_recolor's 45 is SATURATED
                        # by rank/shape/colourset. The residual conditional is COPY-BY-RELATION ("flavor b":
                        # out-colour = a RELATED object's colour), which a key->fixed-colour map cannot express ->
                        # needs a copy-by-relation SOLVE, not more keys. Keys gated off; keymap code retained.
                        for _key in ("rank", "shape", "colourset"):
                            if _object_recolor_solve(gin, gout, sup, h, _lab, _key):
                                solved = "object_recolor"; break
                        if solved is not None:
                            break
                if solved is None:                                       # U11: COPY-BY-RELATION (flavor b)
                    for _rel in ("nearest", "container", "contained", "aligned", "between", "largest", "smallest"):
                        if _object_relrecolor_solve(gin, gout, sup, h, _labs_same, _rel):
                            solved = "object_relrecolor"; break
                if solved is None:                                       # U6: SET-COVER (mixed-clause recolor)
                    if _set_cover_solve(gin, gout, sup, h, _labs_same):
                        solved = "set_cover"
                # GATED (R35): _distance_recolor_solve + _legend_match_solve ROUTED 0 on aug0 (distance_recolor is
                # also costly -- per-pixel HxW loops). Floor-safe + self-checked, kept defined for a full400
                # re-test, but UNWIRED from the ladder per the admit-iff-ROUTES>0 rule.
                # GATED (R35, verify-don't-trust): `_hole_filler_recolor_solve` REMOVED from the ladder. It did
                # `gout[h] = pred; return True` -- overwriting the held-out GROUND TRUTH and returning True with NO
                # exact check (peeking + corrupts the shared `gout` for the rest of the eval), and "hole_filler"
                # is not in OPS (KeyError landmine). Unverified (no self-check) + hardcoded. The duplicate
                # _legend_match_solve call that sat here is also removed (legend_match already runs just above).
                if solved is None:
                    solved = _conditional_geo_solve(gin, gout, sup, h, geo_ops)
                if solved is None:
                    solved = _geo_routed(gin, gout, sup, h, geo_ops)
                if solved is None and len(rule_lib):                     # A2: REUSE stored macros (params re-fit) FIRST
                    for _recipe in rule_lib.recipes():
                        if _execute_recipe_solve(gin, gout, sup, h, _recipe, geo_ops):
                            solved = "macro"; break
                if solved is None:
                    c2 = _compose2_solve(gin, gout, sup, h, geo_ops)     # Stage-2 composition (op1->recolor)
                    if c2 is not None:
                        solved = "compose2"; rule_lib.add(c2, fam)       # Stage-5 ABSTRACT: c2 is already a recipe
                if solved is not None:
                    a["routed"] += 1; a["op"][solved] += 1
                    if a_all is not None:
                        a_all["routed"] += 1; a_all["op"][solved] += 1
                # ----- ORACLE: any primitive fits the held-out DIRECTLY (ceiling) -----
                orc = bool((ti_h == to_h).all())
                if not orc:
                    rm_h = _fit_recolor_map([(ti_h, to_h)])
                    orc = rm_h is not None and bool((_apply_recolor(ti_h, rm_h) == to_h).all())
                if not orc:
                    nm_h = _fit_neighbour_recolor_map([(ti_h, to_h)], side)
                    orc = nm_h is not None and bool((_apply_neighbour_recolor(ti_h, nm_h, side) == to_h).all())
                if not orc:
                    cm_h = _fit_combined_recolor_map([(ti_h, to_h)], side)
                    orc = cm_h is not None and bool((_apply_combined_recolor(ti_h, cm_h, side) == to_h).all())
                if not orc:
                    orc = any(bool(op.candidates(gin[h], gout[h])) for op in geo_ops)
                if not orc:
                    orc = _compose2_solve(gin, gout, [h], h, geo_ops) is not None    # compose2 oracle
                if orc:
                    a["oracle"] += 1
                    if a_all is not None:
                        a_all["oracle"] += 1
                # ----- A1: floor-safe 2-attempt COMMITTED answer (committed >= floor BY CONSTRUCTION) -----
                _a1, _a2 = _committed_solve(gin, gout, sup, h, geo_ops, _labs_same)
                _gh = gout.get(h)
                _ge = lambda p: p is not None and _gh is not None and p.shape == _gh.shape and bool((p == _gh).all())
                _flr = _ge(_a2); _comm = _flr or _ge(_a1)
                a["floor"] += int(_flr); a["committed"] += int(_comm)
                if a_all is not None:
                    a_all["floor"] += int(_flr); a_all["committed"] += int(_comm)
    pc = lambda n, d: 100.0 * n / max(d, 1)
    print("\n[S2 SELECTOR PROBE]  primitive library + cross-demo reconstruction selection (NO training)")
    print("  library = identity | recolor | neighbour_recolor | combined_recolor(colour+nb+shape_cls+pos) | dihedral | translate | tile")
    print("  ROUTED = support-agreed op reconstructs held-out EXACTLY (realizable). ORACLE = any op fits held-out (ceiling).")
    print("  family                    hold   ROUTED-exact%   ORACLE%    per-op (routed)")
    order = ["conditional_recolor", "size_change", "rearrangement", "other(not-in-140)", "all"]
    for fam in order + [k for k in agg if k not in order]:
        a = agg.get(fam)
        if not a or a["hold"] == 0:
            continue
        ops_str = " ".join(f"{k}={a['op'][k]}" for k in OPS if a["op"][k])
        print(f"  {fam:24s} {a['hold']:5d}   {pc(a['routed'], a['hold']):11.1f}   "
              f"{pc(a['oracle'], a['hold']):7.1f}    {ops_str}")
    print("  => ROUTED = the color_exact-equivalent the S2 composer could reach by SELECTION alone (vs current ~0-5).")
    print("     ORACLE-ROUTED = selection/transfer loss; (100-ORACLE) = primitives still MISSING from the library.")
    print("\n  [A1 COMMITTED]  floor-safe 2-attempt emit -- attempt2 = floor => COMMITTED >= FLOOR by construction")
    print("  family                    hold   COMMITTED-exact%   FLOOR-exact%    lift")
    safe = True
    for fam in order + [k for k in agg if k not in order]:
        a = agg.get(fam)
        if not a or a["hold"] == 0:
            continue
        cm, fl = pc(a["committed"], a["hold"]), pc(a["floor"], a["hold"])
        safe = safe and a["committed"] >= a["floor"]
        print(f"  {fam:24s} {a['hold']:5d}   {cm:14.1f}   {fl:10.1f}    +{cm - fl:.1f}")
    print(f"  => COMMITTED >= FLOOR on EVERY family: {safe}  (the route is CLOSED: ROUTED wins now become answers)")
    print("  [Stage-5 ABSTRACT] " + rule_lib.summary())
    if rule_lib_path:                                        # A2: persist discovered macros across runs
        rule_lib.save(rule_lib_path)
    return agg


def characterize_rejected(batches, side, recolour_thresh=0.90):
    """The rejected half is neither clean recolour NOR whole-grid geometry -- so WHAT is it? Structural
    3-way split of the recolour-rejected held-out demos:
      (1) size-change      grid_in.shape != grid_out.shape           (crop / scale / extract / tile)
      (2) recolour-in-place same shape, colour MULTISET changes       (cells stay, colour changes ->
                            a CONDITIONAL recolour our per-cell keys missed)
      (3) rearrange         same shape, colour multiset PRESERVED      (cells MOVE -> local geometric)
    Points at which applier the non-recolour half actually needs."""
    modal = lambda l: max(set(l), key=l.count)
    n = sz = recol = rearr = 0
    for ci, co, dv in batches:
        for b in range(ci.shape[0]):
            valid = [m for m in range(ci.shape[1]) if bool(dv[b, m])]
            if len(valid) < 2:
                continue
            for h in valid:
                sup = [m for m in valid if m != h]
                votes, sc = {}, []
                for m in sup:
                    xin, yout = ci[b, m], co[b, m]
                    chm = (xin != yout) & (xin >= COLOR_OFFSET) & (yout >= COLOR_OFFSET)
                    for i in torch.nonzero(chm).flatten().tolist():
                        a, d = int(xin[i]) - COLOR_OFFSET, int(yout[i]) - COLOR_OFFSET
                        votes.setdefault(a, []).append(d); sc.append((a, d))
                rejected = (sum(1 for a, d in sc if modal(votes[a]) == d) / len(sc)) < recolour_thresh if sc else True
                if not rejected:
                    continue
                gi, go = _extract_grid(ci[b, h], side), _extract_grid(co[b, h], side)
                if gi is None or go is None:
                    continue
                n += 1
                if gi.shape != go.shape:
                    sz += 1
                    continue
                hi = torch.bincount(gi.flatten(), minlength=COLOR_OFFSET + N_COLORS)
                ho = torch.bincount(go.flatten(), minlength=COLOR_OFFSET + N_COLORS)
                if bool((hi == ho).all()):
                    rearr += 1
                else:
                    recol += 1
    pc = lambda x: 100.0 * x / max(n, 1)
    print(f"\n[REJECTED CHARACTERISATION] of {n} recolour-rejected holdouts (what the non-recolour half is):")
    print(f"  size-change (crop/scale/extract):   {sz:>4}  ({pc(sz):.1f}%)")
    print(f"  recolour-in-place (CONDITIONAL):    {recol:>4}  ({pc(recol):.1f}%)  <- cells stay, colours change")
    print(f"  rearrange (local move, hist kept):  {rearr:>4}  ({pc(rearr):.1f}%)  <- cells move in place")
    return {"n": n, "size_change": sz, "recolour_in_place": recol, "rearrange": rearr}


def _geometric_self_check():
    """Validate the detectors on KNOWN ops before trusting them on real tasks."""
    torch.manual_seed(0)
    gi = torch.randint(COLOR_OFFSET, COLOR_OFFSET + N_COLORS, (4, 5))
    do, to, tl = _DihedralOp(), _TranslateOp(), _TileOp()
    assert "flipH" in do.candidates(gi, torch.flip(gi, (1,)))
    assert "rot180" in do.candidates(gi, torch.rot90(gi, 2, (0, 1)))
    assert bool((do.apply(gi, "flipH") == torch.flip(gi, (1,))).all())
    assert (1, 2) in to.candidates(gi, _shift(gi, 1, 2, _mode(gi)))
    assert (2, 3) in tl.candidates(gi, gi.repeat(2, 3))
    canvas = torch.zeros(GRID_SIDE, GRID_SIDE, dtype=torch.long)
    canvas[7:11, 15:20] = gi                                   # place off-corner like the real aug
    ext = _extract_grid(canvas.view(-1), GRID_SIDE)
    assert ext is not None and ext.shape == (4, 5) and bool((ext == gi).all())
    # a recolour (non-geometric) output must NOT trigger any geometric detector (exactness guard).
    recol = gi.clone(); recol[recol == gi[0, 0]] = COLOR_OFFSET + N_COLORS - 1
    assert not (do.candidates(gi, recol) or to.candidates(gi, recol) or tl.candidates(gi, recol)) \
        or bool((recol == gi).all())
    # size-change primitives: crop recovers a content block from a bg canvas; scale up/down round-trips.
    cr, sc = _CropOp(), _ScaleOp()
    gi2 = torch.randint(COLOR_OFFSET + 1, COLOR_OFFSET + N_COLORS, (4, 5))     # colours 1..9 (no bg)
    bgc = torch.full((8, 9), COLOR_OFFSET, dtype=torch.long); bgc[2:6, 3:8] = gi2
    assert "content" in cr.candidates(bgc, gi2) and bool((cr.apply(bgc, "content") == gi2).all()), "crop"
    up = gi2.repeat_interleave(3, 0).repeat_interleave(3, 1)
    assert ("up", 3) in sc.candidates(gi2, up) and bool((sc.apply(gi2, ("up", 3)) == up).all()), "scale up"
    assert ("down", 3) in sc.candidates(up, gi2) and bool((sc.apply(up, ("down", 3)) == gi2).all()), "scale down"
    assert not cr.candidates(gi2, gi2) and not sc.candidates(gi2, gi2), "crop/scale must NOT fire on identity"
    print("geometric self-check PASS (dihedral/translate/tile/crop/scale detect; canvas crop; no false-fire).")


def validate_router_support_only(batches, side, thresh=0.90):
    """The realistic inference test: per held-out demo, fit the cell-colour map on the SUPPORT demos
    ONLY (the test demo's output is unknown at inference) and measure SUPPORT self-consistency, then
    apply the map to the held-out demo. QUESTION: does support-only consistency PREDICT held-out
    coverage? If high-consistency holdouts are well-covered and low ones are not, the recolour-vs-other
    gate works as a pure inference-time signal -- no held-out label, no training."""
    bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]
    stat = {bn: {"c": 0.0, "t": 0.0, "n": 0} for bn in bins}
    acc = {"c": 0.0, "t": 0.0, "n": 0}
    rej = {"c": 0.0, "t": 0.0, "n": 0}
    modal = lambda lst: max(set(lst), key=lst.count)
    for ci, co, dv in batches:
        for b in range(ci.shape[0]):
            valid_m = [m for m in range(ci.shape[1]) if bool(dv[b, m])]
            if len(valid_m) < 2:
                continue
            for h in valid_m:
                votes, scells = {}, []
                for m in valid_m:
                    if m == h:
                        continue
                    xin, yout = ci[b, m], co[b, m]
                    chm = (xin != yout) & (xin >= COLOR_OFFSET) & (yout >= COLOR_OFFSET)
                    for i in torch.nonzero(chm).flatten().tolist():
                        a, d = int(xin[i]) - COLOR_OFFSET, int(yout[i]) - COLOR_OFFSET
                        votes.setdefault(a, []).append(d); scells.append((a, d))
                if not scells:
                    continue
                mp = {a: modal(v) for a, v in votes.items()}
                s_cons = sum(1 for a, d in scells if mp[a] == d) / len(scells)
                xin, yout = ci[b, h], co[b, h]
                chh = (xin != yout) & (xin >= COLOR_OFFSET) & (yout >= COLOR_OFFSET)
                idx = torch.nonzero(chh).flatten().tolist()
                if not idx:
                    continue
                cor = sum(int(mp.get(int(xin[i]) - COLOR_OFFSET, -1) == int(yout[i]) - COLOR_OFFSET)
                          for i in idx)
                for bn in bins:
                    if bn[0] <= s_cons < bn[1]:
                        stat[bn]["c"] += cor; stat[bn]["t"] += len(idx); stat[bn]["n"] += 1
                        break
                tgt = acc if s_cons >= thresh else rej
                tgt["c"] += cor; tgt["t"] += len(idx); tgt["n"] += 1
    cov = lambda d: 100.0 * d["c"] / max(d["t"], 1.0)
    print(f"\n[ROUTER VALIDATION] support-only consistency -> held-out coverage (the inference-time gate)")
    print(f"  {'support-consistency':<22}{'holdouts':>10}{'held-out coverage%':>20}")
    for bn in bins:
        d = stat[bn]
        if d["n"]:
            print(f"  [{bn[0]:.2f},{min(bn[1],1.0):.2f}){'':<11}{d['n']:>10}{cov(d):>20.1f}")
    print(f"  -- gate @ thresh {thresh:.2f}: "
          f"ACCEPT {acc['n']} holdouts -> {cov(acc):.1f}% | REJECT {rej['n']} -> {cov(rej):.1f}%")
    sep = cov(acc) - cov(rej)
    print(f"  => signal {'SEPARATES (gate works)' if sep > 20 else 'WEAK'}: "
          f"accepted coverage is {sep:+.1f}pts vs rejected.")
    return {"accept_n": acc["n"], "accept_cov": cov(acc), "reject_cov": cov(rej)}


def make_shape_conditional_tasks(B=8, M=4, seed=3):
    """colour a appears as TWO shapes -- a 2x2 SQUARE -> b and a 1x3 LINE -> c. Built so SHAPE is the
    ONLY discriminator: both shapes sit in the same 3x3 position band AND the same size bucket (small),
    are isolated (same neighbour/singleton), so cell/size/position/neighbour CANNOT separate them; only
    _shape_key can. The proof the shape key works before trusting it on real data."""
    S, L = GRID_SIDE, GRID_LEN
    g = torch.Generator().manual_seed(seed)
    ci = torch.full((B, M, L), NOISE + COLOR_OFFSET, dtype=torch.long)
    co = torch.full((B, M, L), NOISE + COLOR_OFFSET, dtype=torch.long)
    for b in range(B):
        a = int(torch.randint(1, N_COLORS, (1,), generator=g))
        bb = int(torch.randint(1, N_COLORS, (1,), generator=g))
        while bb == a:
            bb = int(torch.randint(1, N_COLORS, (1,), generator=g))
        cc = int(torch.randint(1, N_COLORS, (1,), generator=g))
        while cc in (a, bb):
            cc = int(torch.randint(1, N_COLORS, (1,), generator=g))
        for m in range(M):
            grid = torch.full((S, S), NOISE + COLOR_OFFSET, dtype=torch.long)
            grid[2:4, 2:4] = a + COLOR_OFFSET                  # 2x2 square (band(0,0), size 4=small)
            grid[6, 2:5] = a + COLOR_OFFSET                    # 1x3 line   (band(0,0), size 3=small)
            out = grid.clone()
            out[2:4, 2:4] = bb + COLOR_OFFSET                  # square -> b
            out[6, 2:5] = cc + COLOR_OFFSET                    # line   -> c
            ci[b, m] = grid.reshape(L); co[b, m] = out.reshape(L)
    return ci, co, torch.ones(B, M, dtype=torch.bool)


def make_scale_varying_shape_tasks(B=8, M=4, seed=4):
    """Each demo holds a SOLID SQUARE -> b and a LINE -> c, but their SIZE VARIES per demo (square
    (m+2)x(m+2), line length m+2). The exact-pixel _shape_key hash thus DIFFERS every demo (cannot
    transfer -> low resolved%), but the abstract _shape_class_key (square=2, line=1) is constant ->
    it transfers and wins. Proves the abstraction beats exact shape on scale variation, the realistic
    intra-task case."""
    S, L = GRID_SIDE, GRID_LEN
    g = torch.Generator().manual_seed(seed)
    ci = torch.full((B, M, L), NOISE + COLOR_OFFSET, dtype=torch.long)
    co = torch.full((B, M, L), NOISE + COLOR_OFFSET, dtype=torch.long)
    for b in range(B):
        a = int(torch.randint(1, N_COLORS, (1,), generator=g))
        bb = int(torch.randint(1, N_COLORS, (1,), generator=g))
        while bb == a:
            bb = int(torch.randint(1, N_COLORS, (1,), generator=g))
        cc = int(torch.randint(1, N_COLORS, (1,), generator=g))
        while cc in (a, bb):
            cc = int(torch.randint(1, N_COLORS, (1,), generator=g))
        for m in range(M):
            k = m + 2
            grid = torch.full((S, S), NOISE + COLOR_OFFSET, dtype=torch.long)
            grid[1:1 + k, 1:1 + k] = a + COLOR_OFFSET                # solid kxk square (top-left)
            grid[20, 2:2 + k] = a + COLOR_OFFSET                     # 1xk line (lower area)
            out = grid.clone()
            out[1:1 + k, 1:1 + k] = bb + COLOR_OFFSET                # square -> b
            out[20, 2:2 + k] = cc + COLOR_OFFSET                     # line   -> c
            ci[b, m] = grid.reshape(L); co[b, m] = out.reshape(L)
    return ci, co, torch.ones(B, M, dtype=torch.bool)


def relational_copy_probe(batches, side, thresh=0.90):
    """Test the 'NO INVENTION' hypothesis on the conditional-recolour bucket: is every changed cell's
    OUTPUT colour already present in that demo's INPUT grid (the rule re-uses existing colours by some
    RELATION), or are new colours conjured? Then attribute each genuine recolour to a relational SOURCE:
    background / adjacent-object colour / largest-object colour / other-present / novel. Done per demo
    (a within-task property -- the rule is the same across a task's demos, so no leave-one-out needed)."""
    def collect(predicate):
        out = []
        for ci, co, dv in batches:
            for b in range(ci.shape[0]):
                valid = [m for m in range(ci.shape[1]) if bool(dv[b, m])]
                if len(valid) < 2:
                    continue
                r = _task_recolour_consistency(ci[b], co[b], dv[b])
                if r is not None and predicate(r[0]):
                    out.append((ci[b], co[b], valid))
        return out

    cond = collect(lambda c: c < thresh)
    clean = collect(lambda c: c >= 0.97)

    def analyze(tasks):
        cell_in = cell_tot = grid_sub = grid_tot = 0
        src = {"background": 0, "adjacent-object": 0, "largest-object": 0, "other-present": 0, "novel": 0}
        for ci, co, valid in tasks:
            for m in valid:
                xin, yout = ci[m], co[m]
                in_pal = set((xin[xin >= COLOR_OFFSET] - COLOR_OFFSET).tolist())
                out_pal = set((yout[yout >= COLOR_OFFSET] - COLOR_OFFSET).tolist())
                if not out_pal:
                    continue
                grid_tot += 1
                grid_sub += int(out_pal <= in_pal)
                bg = _grid_bg(xin)
                tc = _touch_key(xin.unsqueeze(0), side)[0]
                rk = _rank_key(xin.unsqueeze(0), side)[0]
                m0 = (rk == 0) & (xin >= COLOR_OFFSET)
                largest = int((xin[m0][0]) - COLOR_OFFSET) if bool(m0.any()) else -1
                ch = (xin != yout) & (xin >= COLOR_OFFSET) & (yout >= COLOR_OFFSET)
                for i in torch.nonzero(ch).flatten().tolist():
                    d = int(yout[i]) - COLOR_OFFSET
                    cell_tot += 1
                    cell_in += int(d in in_pal)
                    if d not in in_pal:
                        src["novel"] += 1
                    elif d == bg:
                        src["background"] += 1
                    elif int(tc[i]) == d:
                        src["adjacent-object"] += 1
                    elif d == largest:
                        src["largest-object"] += 1
                    else:
                        src["other-present"] += 1
        return cell_in, cell_tot, grid_sub, grid_tot, src

    print(f"\n[RELATIONAL / NO-INVENTION PROBE]  does the output re-use INPUT colours? (per demo)")
    for nm, tasks in (("conditional", cond), ("clean-recolour", clean)):
        ci_, ct_, gs_, gt_, src = analyze(tasks)
        pc = lambda n, d: 100.0 * n / max(d, 1)
        print(f"  {nm:<16} tasks={len(tasks):>3}  "
              f"changed-cell out-colour IN input palette = {pc(ci_, ct_):.1f}%  "
              f"| whole-output palette within input = {pc(gs_, gt_):.1f}%")
        if nm == "conditional" and ct_:
            tot = sum(src.values())
            order = ["background", "adjacent-object", "largest-object", "other-present", "novel"]
            print("     source of the recolour (changed cells): " +
                  "  ".join(f"{k}={100.0*src[k]/tot:.0f}%" for k in order))
    return cond, clean


def _synthetic_ranker_check():
    """Validate the ranker on tasks with a KNOWN best key BEFORE trusting it on real data:
      * make_recolor_tasks       (pure cell-colour rule) -> NO key should beat cell.
      * make_object_conditioned  (size-conditional rule) -> size/singleton SHOULD win.
      * make_shape_conditional   (shape-conditional rule) -> shape SHOULD win (size/position cannot)."""
    ci, co, cm, _s, _d = make_recolor_tasks(B=8, M=4, seed=0)
    _r, win = rank_keys([(ci, co, cm)], GRID_SIDE, "SYNTHETIC pure cell-colour rule (expect NO winner)")
    assert win is None, f"ranker BROKEN: a key beat cell on a pure cell-colour rule ({win})"
    ci, co, cm, _m = make_object_conditioned_tasks(B=8, M=4, seed=1)
    _r, win = rank_keys([(ci, co, cm)], GRID_SIDE, "SYNTHETIC size-conditional rule (expect size/singleton win)")
    assert win is not None and win[0] in ("size", "singleton"), \
        f"ranker BROKEN: size/singleton must win the size-conditional task (got {win})"
    ci, co, cm = make_shape_conditional_tasks(B=8, M=4, seed=3)
    _r, win = rank_keys([(ci, co, cm)], GRID_SIDE, "SYNTHETIC shape-conditional rule (expect shape win)",
                        keys=KEYS_COND)
    assert win is not None and win[0] in ("shape", "shape_cls"), \
        f"ranker BROKEN: shape must win the shape-conditional task (got {win})"
    ci, co, cm = make_scale_varying_shape_tasks(B=8, M=4, seed=4)
    rows, win = rank_keys([(ci, co, cm)], GRID_SIDE,
                          "SYNTHETIC scale-varying shape rule (expect shape_cls win; exact shape cannot transfer)",
                          keys=KEYS_COND)
    res = {r[0]: r for r in rows}                                # r = (name, cov, resolved, tot)
    assert res["shape_cls"][1] - res["cell"][1] > 1.0 and res["shape_cls"][2] > 20.0, \
        f"ranker BROKEN: abstract shape_cls must transfer+win the scale-varying task (got {res['shape_cls']})"
    assert res["shape"][2] < 40.0, \
        f"exact _shape_key should NOT transfer across sizes (resolved {res['shape'][2]:.1f}% expected <40)"
    print("\nranker self-check PASS (cell-rule no winner; size-rule size; shape-rule shape; "
          "scale-varying shape_cls wins where exact shape cannot transfer).")


def load_real_eval_batches(n_batches, batch, dataset_path=None):
    """Load real LODO demos from a probe dataset. Returns a list of (ci, co, dv) CPU tuples with
    >=2 valid demos each. dv[b,m] = demo m of task b has a colour cell. dataset_path defaults to the
    training distribution (aug-1000); pass an aug-0 root for the fast un-augmented probe -- the LODO
    context structure is identical either way (intra-puzzle, not sibling-augmentation)."""
    import os
    os.environ.setdefault("DISABLE_COMPILE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    import yaml
    import pretrain
    import numpy as _np
    import random as _random
    torch.manual_seed(0)                                        # reproducible batch draw run-to-run
    _np.random.seed(0)                                          # context demos sampled with NUMPY rng
    _random.seed(0)
    raw = yaml.safe_load(Path(CONFIG_PATH).resolve().read_text(encoding="utf-8"))
    raw["load_checkpoint"] = CKPT_PATH
    raw["data_paths"] = [dataset_path or DATASET_PATH]
    raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0
    raw["global_batch_size"] = batch
    raw["run_name"] = "check_extraction"
    raw["checkpoint_path"] = "reports/check_extraction"
    config = pretrain.PretrainConfig(**raw)
    loader, _meta = pretrain.create_dataloader(
        config, "train", 0, 1, test_set_mode=False,
        epochs_per_iter=1, global_batch_size=config.global_batch_size)
    out, pids = [], []
    for _set, cb, _g in loader:
        if "context_inputs" not in cb or cb["context_inputs"].shape[1] < 2:
            continue
        ci = cb["context_inputs"].cpu().long()
        co = cb["context_outputs"].cpu().long()
        dv = (ci >= COLOR_OFFSET).any(dim=-1)                    # [B,M] demo has colour content
        out.append((ci, co, dv))
        pid = cb.get("puzzle_identifiers")                       # [B] task id -> identifiers.json -> family
        pids.append(pid.cpu().long() if hasattr(pid, "cpu") else
                    (torch.as_tensor(pid).long() if pid is not None else None))
        if len(out) >= n_batches:
            break
    return out, pids


def _load_family_map(probe_path):
    """(id2hash, hash2fam) for the per-family report, or (None,None) if unavailable.
    id2hash = the probe dataset's identifiers.json (index -> ARC task hash);
    hash2fam = codex family per task from the atlas CSV (the 140 priority set)."""
    import json
    import csv
    ids_path = Path(probe_path) / "identifiers.json"
    csv_path = (Path(__file__).resolve().parents[1] / "reports" / "arc_task_atlas"
                / "selected_task_categories.csv")
    if not ids_path.exists() or not csv_path.exists():
        return None, None
    id2hash = json.loads(ids_path.read_text(encoding="utf-8"))
    hash2fam = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:   # utf-8-sig strips the CSV BOM
        for row in csv.DictReader(f):
            hash2fam[row["task_id"].strip()] = row["category"].strip()
    return id2hash, hash2fam


def _demo_structure_stats(ti, to):
    """Per held-out demo (ti,to = input/output tokens [L]) -> (recolour_in_place, struct_match, n_colour).
    recolour_in_place = colour-cell POSITIONS identical in->out (only colours change) -> the family
    c2_copy_structure_prior can fix. struct_match = structure-class (pad/eos/colour) agreement on the
    ACTIVE canvas (colour in either) -> how close 'copy input structure' gets the output structure."""
    pin, pout = (ti >= COLOR_OFFSET), (to >= COLOR_OFFSET)        # colour-cell masks
    active = pin | pout
    na = int(active.sum())
    if na == 0:
        return None
    rip = float(bool((pin == pout).all()))                       # positions kept (recolour-in-place)
    cls_i = torch.where(pin, torch.full_like(ti, 2), ti)         # 0=PAD, 1=EOS, 2=colour
    cls_o = torch.where(pout, torch.full_like(to, 2), to)
    struct_match = float(((cls_i == cls_o) & active).sum()) / na
    return rip, struct_match, float(pin.sum())


def per_family_structure_report(batches, pids, probe_path, side):
    """PHASE-4 SIZING + codex-label cross-check. Per codex family (the priority 140), report the
    DETERMINISTIC structure ceiling on held-out demos: %recolour-in-place and active-cell structure
    match. Same-positions families (conditional_recolor) -> c2_copy_structure_prior can force output
    structure = input structure (the Phase-4 lever). size_change cannot (output canvas differs ->
    needs Phase-6 output-shape prediction); rearrangement moves cells (positions change)."""
    id2hash, hash2fam = _load_family_map(probe_path)
    if id2hash is None or pids is None or any(p is None for p in pids):
        print("\n[per-family] skipped (need --probe-dataset aug0 + the atlas CSV + pids).")
        return
    agg = {}
    for (ci, co, dv), pid in zip(batches, pids):
        B, M, _ = ci.shape
        for b in range(B):
            idx = int(pid[b])
            h_hash = id2hash[idx] if 0 <= idx < len(id2hash) else None
            fam = hash2fam.get(h_hash, "other(not-in-140)")
            a = agg.setdefault(fam, {"tasks": 0, "hold": 0, "rip": 0.0, "smatch": 0.0, "ncol": 0.0})
            a["tasks"] += 1
            for h in range(M):
                if not bool(dv[b, h]):
                    continue
                st = _demo_structure_stats(ci[b, h], co[b, h])
                if st is None:
                    continue
                a["hold"] += 1; a["rip"] += st[0]; a["smatch"] += st[1]; a["ncol"] += st[2]
    print("\n[PER-FAMILY STRUCTURE]  codex family (priority 140) -> deterministic structure ceiling (Phase-4 sizing)")
    print("  family                  tasks  holdouts  recolour-in-place%  active-struct-match%  avg-colour-cells")
    order = ["conditional_recolor", "size_change", "rearrangement", "other(not-in-140)"]
    for fam in order + [k for k in agg if k not in order]:
        a = agg.get(fam)
        if not a:
            continue
        h = max(a["hold"], 1)
        print(f"  {fam:22s} {a['tasks']:6d} {a['hold']:9d}   {100*a['rip']/h:16.1f}   "
              f"{100*a['smatch']/h:18.1f}   {a['ncol']/h:14.1f}")
    print("  => recolour-in-place ~100% (conditional_recolor): c2_copy_structure_prior can force output")
    print("     structure = input -> the Phase-4 fix for unchg. size_change low: output canvas differs")
    print("     -> needs output-shape prediction (Phase 6). rearrangement: cells move (positions change).")


def _per_family_self_check():
    """The structure metric must separate recolour-in-place from size-change/move."""
    L = GRID_LEN
    # recolour-in-place: same positions, only colour changes (3->7)
    ti = torch.zeros(L, dtype=torch.long); ti[:50] = 3 + COLOR_OFFSET
    to = torch.zeros(L, dtype=torch.long); to[:50] = 7 + COLOR_OFFSET
    rip, sm, _ = _demo_structure_stats(ti, to)
    assert rip == 1.0 and sm == 1.0, f"recolour-in-place must score rip=1 struct=1 (got {rip},{sm})"
    # size-change: output occupies DIFFERENT cells (canvas differs)
    to2 = torch.zeros(L, dtype=torch.long); to2[200:210] = 5 + COLOR_OFFSET
    rip2, sm2, _ = _demo_structure_stats(ti, to2)
    assert rip2 == 0.0 and sm2 < 0.2, f"size-change must score rip=0 struct<0.2 (got {rip2},{sm2})"
    print("per-family structure self-check PASS (recolour-in-place rip=1/struct=1; size-change rip=0/struct~0).")


def main():
    ap = argparse.ArgumentParser(description="cross-demo extraction check + real-data key ranking")
    ap.add_argument("--real", action="store_true",
                    help="rank conditioning keys (cell/size/singleton/position/neighbour) on the REAL "
                         "LODO held-out demos -- the diagnostic for which VALUE key to wire next.")
    ap.add_argument("--real-batches", type=int, default=8)
    ap.add_argument("--rule-lib", default=None,
                    help="A2: path to persist/load the discovered-macro RuleLibrary (JSON) across runs")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--probe-dataset", choices=sorted(PROBE_DATASETS), default="aug1000",
                    help="which dataset the --real diagnostics read. aug1000 = the training "
                         "distribution (default, ~913x augmented). aug0 = the un-augmented 960-task "
                         "seed: fast (~7MB), clean per-task distribution, but NOT held out -- a "
                         "measuring stick, not a generalization test. Use --real-batches 240 to "
                         "cover all 960 aug-0 tasks.")
    args = ap.parse_args()

    ci, co, cm, src, dst = make_recolor_tasks(B=8, M=4, seed=0)
    good = run_check(CTBankExtractor(), ci, co, cm, src, dst)
    bad = run_check(ShuffleInvariantPooler(), ci, co, cm, src, dst)
    print(f"[GOOD] {CTBankExtractor.name}\n        {_fmt(good)}")
    print(f"[BAD ] {ShuffleInvariantPooler.name}\n        {_fmt(bad)}")
    assert good["verdict"], "CHECK BROKEN: a task-specific extractor must PASS"
    assert not bad["verdict"], "CHECK BROKEN: a pairing-invariant extractor must FAIL"
    print("\ncheck_extraction self-validation PASS "
          "(separates task-specific from collapsed extractors).")
    print()
    _demo_object_conditioned()
    print()
    _synthetic_ranker_check()
    _geometric_self_check()
    _per_family_self_check()
    _s2_self_check()
    _core_knowledge_self_check()
    _compose_self_check()
    _parser_self_check()
    _symfill_self_check()
    _relational_self_check()
    _extract_panel_self_check()
    _relrecolor_self_check()
    _set_cover_self_check()
    _conditional_geo_self_check()
    _distance_recolor_self_check()
    _legend_match_self_check()
    _committed_self_check()
    _execute_recipe_self_check()
    _size_change_self_check()

    if args.real:
        probe_path = PROBE_DATASETS[args.probe_dataset]
        print(f"\n[real] probe-dataset={args.probe_dataset}  ({probe_path})")
        print(f"[real] loading {args.real_batches} batches (batch={args.batch}) ...")
        batches, pids = load_real_eval_batches(args.real_batches, args.batch, dataset_path=probe_path)
        ntasks = sum(ci.shape[0] for ci, _, _ in batches)
        print(f"[real] {len(batches)} batches, {ntasks} tasks with >=2 demos.")
        per_family_structure_report(batches, pids, probe_path, GRID_SIDE)
        s2_selector_probe(batches, pids, probe_path, GRID_SIDE, rule_lib_path=args.rule_lib)
        rank_keys(batches, GRID_SIDE, f"REAL LODO held-out demos ({ntasks} tasks)")
        task_type_split(batches, GRID_SIDE)
        validate_router_support_only(batches, GRID_SIDE)
        geometric_detector_diagnostic(batches, GRID_SIDE)
        characterize_rejected(batches, GRID_SIDE)
        sub = conditional_recolour_subset(batches, GRID_SIDE)
        print(f"\n[real] conditional-recolour subset: {len(sub)} tasks "
              f"(recolour-in-place but NOT cell-consistent -- the ~26% prize)")
        if sub:
            rank_keys(sub, GRID_SIDE, f"CONDITIONAL-recolour subset ({len(sub)} tasks)",
                      keys=dict(KEYS_COND, **{"nb+shp+pos": _combined_key,
                                              "count": _count_key, "topology": _topology_key}))
        relational_copy_probe(batches, GRID_SIDE)


if __name__ == "__main__":
    main()

