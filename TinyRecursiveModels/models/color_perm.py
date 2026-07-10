"""Phase 1: colour-permutation augmentation (the cheapest anti-memorisation lever).

Per example, permute the 10 colour tokens (2..11) by a random permutation applied CONSISTENTLY
across that example's inputs / labels / context_inputs / context_outputs. PAD=0, EOS=1, and any
ignore label (< COLOR_OFFSET, e.g. -100) are left untouched.

The RULE STRUCTURE is preserved: a recolour src->dst becomes perm(src)->perm(dst) in every demo
AND the test of that task, so cross-demo extraction (cond_inout/cond_changed) still recovers a
consistent rule -- but the model can no longer memorise "colour 3 -> 7"; it must read the map from
the demos. dpcc (extraction) is unaffected; MAIN (puzzle_emb recall) drops -- that drop IS the
success signal that memorisation is being broken.
"""
from __future__ import annotations

import torch

VOCAB = 12
COLOR_OFFSET = 2
N_COLORS = 10


def make_remap(B: int, device, gen: torch.Generator | None = None) -> torch.Tensor:
    """[B,12] token remap: identity for PAD(0)/EOS(1); a per-example random perm of colours 2..11.

    Vectorised (no per-example Python loop). Random draw happens on the GENERATOR's own device
    (torch requires generator.device == tensor.device), then moves to `device` -- so a CUDA or a
    CPU generator both work, closing the device-mismatch crash the old .to(device)-after-randperm
    only avoided for CPU generators."""
    remap = torch.arange(VOCAB, device=device).unsqueeze(0).repeat(B, 1)
    gdev = gen.device if gen is not None else torch.device("cpu")
    # argsort of uniform noise = a uniform random permutation per row (Fisher-Yates-equivalent).
    perm = torch.argsort(torch.rand(B, N_COLORS, generator=gen, device=gdev), dim=1)   # [B, N]
    remap[:, COLOR_OFFSET:VOCAB] = perm.to(device) + COLOR_OFFSET
    return remap


def _remap(t: torch.Tensor, remap: torch.Tensor) -> torch.Tensor:
    B = remap.shape[0]
    flat = t.reshape(B, -1).long()
    g = torch.gather(remap, 1, flat.clamp(0, VOCAB - 1))
    out = torch.where((flat >= COLOR_OFFSET) & (flat < VOCAB), g, flat)      # only permute colours
    return out.reshape(t.shape)


def apply_color_perm(batch: dict, gen: torch.Generator | None = None) -> dict:
    """In-place per-example colour permutation across all grid tensors of the batch."""
    inp = batch.get("inputs")
    if inp is None or not torch.is_tensor(inp):
        return batch
    remap = make_remap(inp.shape[0], inp.device, gen)
    for k in ("inputs", "labels", "context_inputs", "context_outputs"):
        v = batch.get(k)
        if torch.is_tensor(v):
            batch[k] = _remap(v, remap)
    return batch


def _self_test() -> None:
    B, M, L = 2, 3, 16
    batch = {
        "inputs": torch.zeros(B, L, dtype=torch.long),
        "labels": torch.zeros(B, L, dtype=torch.long),
        "context_inputs": torch.zeros(B, M, L, dtype=torch.long),
        "context_outputs": torch.zeros(B, M, L, dtype=torch.long),
    }
    # example 0: rule colour 3 -> 7, with EOS at index 1 and PAD at index 0.
    for d in (("inputs", 3), ("labels", 7)):
        batch[d[0]][0, 1] = 1                                  # EOS
        batch[d[0]][0, 2:6] = d[1] + COLOR_OFFSET
    batch["context_inputs"][0, :, 2:6] = 3 + COLOR_OFFSET
    batch["context_outputs"][0, :, 2:6] = 7 + COLOR_OFFSET
    # ignore-label survives untouched
    batch["labels"][1, 0] = -100

    gen = torch.Generator().manual_seed(42)
    out = apply_color_perm(batch, gen)

    # (1) PAD/EOS/ignore preserved
    assert int(out["inputs"][0, 0]) == 0 and int(out["inputs"][0, 1]) == 1, "PAD/EOS must survive"
    assert int(out["labels"][1, 0]) == -100, "ignore label must survive"

    # (2) the SAME perm applied to inputs AND context of an example (consistency)
    p_in = int(out["inputs"][0, 2]) - COLOR_OFFSET            # perm(3) seen in inputs
    p_ctx = int(out["context_inputs"][0, 0, 2]) - COLOR_OFFSET
    assert p_in == p_ctx, "perm must be consistent across inputs and context"

    # (3) rule preserved: perm(3) -> perm(7), same on labels and context_outputs
    p_out = int(out["labels"][0, 2]) - COLOR_OFFSET           # perm(7)
    p_cout = int(out["context_outputs"][0, 0, 2]) - COLOR_OFFSET
    assert p_out == p_cout, "rule target must be consistent"
    assert p_in != p_out, "3->7 must stay a genuine recolour (perm injective)"

    print(f"color_perm self-test PASS  (3->7 became {p_in}->{p_out}; PAD/EOS/ignore preserved; "
          f"consistent across demos+test)")


if __name__ == "__main__":
    _self_test()
