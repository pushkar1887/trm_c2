"""Isolated PAD diagnostic: WHY the outside-grid lever doesn't flip pad, checked on the exact candidate math.

Replicates _output_logits's factored candidate construction (trm_fvr_c2.py) on synthetic PAD and EOS cells
so we can see, without a full training run:
  1. how big the lever must be to flip a pad cell (the LSE_c - lm_pad gap), and
  2. whether outside_grid = (1 - valid_mask) [contaminates EOS] vs (input == PAD) [eos-clean] hurts EOS.

Run: py -3.11 scripts/diag_pad_gap.py   (torch only; no checkpoint/coolname needed)
"""
import torch
import torch.nn.functional as F

PAD, EOS = 0, 1


def candidate_argmax(lm_pad, lm_eos, colour, outside_val_pad_row):
    """Exact _output_logits factored candidate on ONE cell.

    structure_logits = [lm_pad + outside(pad row), lm_eos, logsumexp(colour)]
    candidate = [structure_logp[pad], structure_logp[eos], structure_logp[valid] + color_logp]
    Returns argmax over the 12 tokens (0=pad, 1=eos, 2..11=colour).
    """
    colour = torch.as_tensor(colour, dtype=torch.float32)
    lse_c = torch.logsumexp(colour, dim=-1)
    structure_logits = torch.stack([
        torch.as_tensor(lm_pad, dtype=torch.float32) + torch.as_tensor(outside_val_pad_row, dtype=torch.float32),
        torch.as_tensor(lm_eos, dtype=torch.float32),
        lse_c,
    ])
    structure_logp = F.log_softmax(structure_logits, dim=-1)
    color_logp = F.log_softmax(colour, dim=-1)
    candidate = torch.cat([structure_logp[0:1], structure_logp[1:2], structure_logp[2:3] + color_logp])
    return int(candidate.argmax().item()), float(structure_logits[0]), float(structure_logits[2])


def main():
    print("=" * 78)
    print("PAD CELL: floor confidently predicts COLOUR (blank-pid LODO doesn't know the shape).")
    print("gap = logsumexp(colour) - lm_pad  is what the lever must overcome.")
    print("=" * 78)
    # A pad cell: lm_pad low, lm_eos low, one colour dominates at `peak` (confident) -> LSE_c ~ peak.
    lm_pad, lm_eos = 0.0, 0.0
    for peak in (6.0, 10.0, 14.0):
        colour = [0.0] * 10
        colour[3] = peak                       # confident single colour
        lse_c = float(torch.logsumexp(torch.tensor(colour), 0))
        gap = lse_c - lm_pad
        # sweep the outside lever magnitude V added to the pad row on this (outside_grid=1) cell
        flip_V = None
        row = []
        for V in (0.0, 6.0, 7.7, 12.0, gap - 0.5, gap + 0.5, 30.0):
            am, sp, sv = candidate_argmax(lm_pad, lm_eos, colour, V)
            row.append((round(V, 1), "PAD" if am == PAD else ("EOS" if am == EOS else f"col{am-2}")))
            if am == PAD and flip_V is None:
                flip_V = V
        print(f"\ncolour_peak={peak:>5} -> LSE_c={lse_c:5.2f}  gap={gap:5.2f}")
        print("  V (pad-row add) -> candidate argmax:", row)
        print(f"  => pad first wins at V >= ~{gap:.1f}  (warm-init 6.0 and the grown 7.7 are BELOW this)")

    print("\n" + "=" * 78)
    print("EOS CONTAMINATION: outside_grid = (1 - valid_mask) is 1 on EOS cells too (valid_mask=0).")
    print("So a big pad-row lever RAISES pad on EOS cells -> converts EOS->PAD (hurts eos).")
    print("Contrast: outside_grid = (input == PAD) is 0 on EOS cells -> eos-clean.")
    print("=" * 78)
    # An EOS cell: floor confidently predicts EOS. lm_eos high, colour low.
    lm_pad_e, lm_eos_e = 0.0, 10.0
    colour_e = [0.0] * 10
    for V in (0.0, 6.0, 20.0, 30.0):
        # (1-valid_mask) => outside=1 on this eos cell => pad row gets +V
        am_contam, _, _ = candidate_argmax(lm_pad_e, lm_eos_e, colour_e, V)
        # (input==PAD) => outside=0 on this eos cell => pad row gets +0 (eos-clean)
        am_clean, _, _ = candidate_argmax(lm_pad_e, lm_eos_e, colour_e, 0.0)
        tag = lambda a: "PAD" if a == PAD else ("EOS" if a == EOS else f"col{a-2}")
        print(f"  V={V:>5}:  (1-valid_mask) eos-cell -> {tag(am_contam):3}   |   (input==PAD) eos-cell -> {tag(am_clean):3}")
    print("\nCONCLUSION: need (a) magnitude >= gap (warm-init ~ gap, not 6), AND (b) an EOS-CLEAN pad mask")
    print("            (input==PAD), else a big lever destroys EOS. Both are the fix.")


if __name__ == "__main__":
    main()
