"""S1 — Canvas Cleanup post-processor.

Given a 900-token TRM prediction and a predicted (h, w) canvas, force every
token outside the canvas to EOS. This removes the outside-canvas false
positives that pollute close-miss tasks without touching inside-canvas
predictions.

Token vocabulary (per fvr_structfuse_alpha_sweep.crop_shape):
  0  : padding
  1  : EOS
  2-11 : colours 0-9
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


EOS_TOKEN = 1
PAD_TOKEN = 0
COLOR_TOKEN_OFFSET = 2  # token id for colour c == c + 2


def crop_shape_from_seq(seq: np.ndarray) -> Tuple[int, int]:
    """Replicates fvr_structfuse_alpha_sweep.crop_shape — derives (h, w) from
    a 900-token flat sequence by finding the largest top-left rectangular
    region where every token is a colour (2..11)."""
    grid = seq.reshape(30, 30)
    num_c = 30
    max_area = 0
    max_shape = (0, 0)
    for num_r in range(1, 31):
        for c in range(1, num_c + 1):
            x = int(grid[num_r - 1, c - 1])
            if x < 2 or x > 11:
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_shape = (num_r, num_c)
    return max_shape


def canvas_cleanup(
    pred_seq: np.ndarray,
    pred_h: Optional[int] = None,
    pred_w: Optional[int] = None,
    fill_token: int = EOS_TOKEN,
) -> np.ndarray:
    """Force tokens outside (pred_h, pred_w) to `fill_token`.

    If pred_h / pred_w are not provided, derive them from the prediction
    itself using crop_shape_from_seq (matches the shape used by row_metrics).

    Returns a NEW 900-token sequence; does not mutate input.
    """
    if pred_h is None or pred_w is None:
        h, w = crop_shape_from_seq(pred_seq)
    else:
        h, w = int(pred_h), int(pred_w)
    if h <= 0 or w <= 0:
        return pred_seq.copy()
    grid = pred_seq.reshape(30, 30).copy()
    cleaned = np.full_like(grid, fill_token)
    cleaned[:h, :w] = grid[:h, :w]
    return cleaned.flatten()


def canvas_cleanup_batch(pred_seqs: np.ndarray, pred_hws: Optional[np.ndarray] = None) -> np.ndarray:
    """Batched version. pred_seqs: (B, 900). pred_hws optional (B, 2)."""
    out = np.empty_like(pred_seqs)
    for i in range(pred_seqs.shape[0]):
        h = w = None
        if pred_hws is not None:
            h, w = int(pred_hws[i, 0]), int(pred_hws[i, 1])
        out[i] = canvas_cleanup(pred_seqs[i], pred_h=h, pred_w=w)
    return out


def _selftest() -> None:
    # Test 1: crop_shape correctly identifies a clean 11x11 canvas
    grid = np.full((30, 30), EOS_TOKEN, dtype=np.int64)
    grid[:11, :11] = 7  # 11x11 of colour 5 (token 7)
    h, w = crop_shape_from_seq(grid.flatten())
    assert h == 11 and w == 11, f"crop_shape on clean canvas: ({h}, {w}); expected (11, 11)"

    # Test 2: canvas_cleanup with explicit (h, w) removes garbage outside.
    rng = np.random.default_rng(0)
    dirty = grid.copy()
    for r in range(11, 30):
        for c in range(30):
            dirty[r, c] = int(rng.integers(2, 12))
    for r in range(30):
        for c in range(11, 30):
            dirty[r, c] = int(rng.integers(2, 12))
    cleaned = canvas_cleanup(dirty.flatten(), pred_h=11, pred_w=11)
    cleaned_grid = cleaned.reshape(30, 30)
    assert (cleaned_grid[11:, :] == EOS_TOKEN).all(), "bottom region not cleaned"
    assert (cleaned_grid[:11, 11:] == EOS_TOKEN).all(), "right region not cleaned"
    assert (cleaned_grid[:11, :11] == 7).all(), "inside canvas corrupted"

    # Test 3: batched version
    batch = np.stack([dirty.flatten(), dirty.flatten()])
    hws = np.array([[11, 11], [11, 11]])
    out = canvas_cleanup_batch(batch, hws)
    assert out.shape == (2, 900)
    assert (out[0] == cleaned).all()
    print("[canvas_cleanup self-test] PASS")


if __name__ == "__main__":
    _selftest()
