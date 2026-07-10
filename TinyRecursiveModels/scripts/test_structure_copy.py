"""CPU-only logic check for the Stage-6 structure-copy prior (no model, instant).

Builds adversarial base logits that WRONGLY predict grey on PAD cells / PAD on black cells
(exactly the failure the visualiser found: PAD->grey, black<->PAD), applies the same bias math
used in trm_fvr_c2._output_logits, and asserts the output STRUCTURE class (PAD/EOS/VALID) now
matches the INPUT's. Confirms the fix is correct before spending a GPU run.
"""
import torch
import torch.nn.functional as F

PAD, EOS, OFF = 0, 1, 2


def struct_class(tok):
    return torch.where(tok >= OFF, torch.full_like(tok, 2),
                       torch.where(tok == EOS, torch.ones_like(tok), torch.zeros_like(tok)))


def apply_copy_structure(base_logits, si, shape_preserve, gate):
    """Mirror of the model's structure-copy prior."""
    sp = shape_preserve.view(-1, 1, 1)
    s = F.softplus(gate) * sp
    in_pad = (si == PAD).unsqueeze(-1).float()
    in_eos = (si == EOS).unsqueeze(-1).float()
    in_col = (si >= OFF).unsqueeze(-1).float()
    out = base_logits.clone()
    out[..., 0:1] = out[..., 0:1] + s * (in_pad - in_eos - in_col)
    out[..., 1:2] = out[..., 1:2] + s * (in_eos - in_pad - in_col)
    out[..., 2:12] = out[..., 2:12] + s * (in_col - in_pad - in_eos)
    return out


def main():
    torch.manual_seed(0)
    # input grid (tokens): PAD, EOS, color3(=tok5), black(=tok2), color7(=tok9), PAD, EOS, black
    si = torch.tensor([[PAD, EOS, 5, 2, 9, PAD, EOS, 2]])
    B, L, V = 1, si.shape[1], 12
    in_struct = struct_class(si)

    # ADVERSARIAL base: argmax deliberately wrong in the way the visualiser found --
    # PAD cells -> grey(color5=tok7); black cells -> PAD; others random-ish.
    base = torch.randn(B, L, V) * 0.1
    for j in range(L):
        t = int(si[0, j])
        if t == PAD:
            base[0, j, 7] += 8.0          # PAD -> grey (the #1 real error)
        elif t == EOS:
            base[0, j, 5] += 8.0          # EOS -> some colour
        else:                              # colour cell (incl. black) -> PAD
            base[0, j, PAD] += 8.0        # black<->PAD confusion

    before = base.argmax(-1)
    before_struct = struct_class(before)
    print("input tokens      :", si.tolist()[0])
    print("input struct       :", in_struct.tolist()[0], "(0=PAD 1=EOS 2=VALID)")
    print("BEFORE pred tokens :", before.tolist()[0])
    print("BEFORE pred struct :", before_struct.tolist()[0])
    base_match = int((before_struct == in_struct).sum())
    print(f"BEFORE structure match: {base_match}/{L}")

    fixed = apply_copy_structure(base, si, torch.tensor([1.0]), torch.tensor(3.0))
    after = fixed.argmax(-1)
    after_struct = struct_class(after)
    print("AFTER  pred struct :", after_struct.tolist()[0])
    after_match = int((after_struct == in_struct).sum())
    print(f"AFTER  structure match: {after_match}/{L}")

    assert after_match == L, f"structure-copy FAILED: only {after_match}/{L} cells match input structure"
    # shape-preserve=0 must be a no-op (don't touch shape-changing tasks)
    noop = apply_copy_structure(base, si, torch.tensor([0.0]), torch.tensor(3.0))
    assert torch.allclose(noop, base), "shape_preserve=0 should be a no-op but changed logits"
    print("\n[PASS] structure-copy forces output structure == input structure (PAD/EOS/VALID),")
    print("       and is a NO-OP when shape_preserve=0 (shape-changing tasks untouched).")


if __name__ == "__main__":
    main()
