"""Shared V3 model-candidate dump format.

The dump is intentionally limited to what Verify/Select needs:
  task_ids[N]        string task ids
  test_indices[N]   integer test/fold index inside each task
  record_kinds[N]   optional string kind: "fold" for LOOCV, "test" for target inputs
  candidates[N,K,900] flat TRM tokens: PAD=0, EOS=1, colour=raw+2
  vote_counts[N,K]  TTC/augmentation vote counts or any non-peeking model score

No target outputs, labels, or verifier decisions belong in this file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Any

import numpy as np


DumpRecord = Mapping[str, Any]


def _as_candidate_array(candidates: Any, side: int) -> np.ndarray:
    arr = np.asarray(candidates, dtype=np.int64)
    if arr.ndim == 2 and arr.shape[1] == side * side:
        return arr
    if arr.ndim == 3 and arr.shape[1:] == (side, side):
        return arr.reshape(arr.shape[0], side * side)
    raise ValueError(f"candidates must be [K,{side * side}] or [K,{side},{side}], got {arr.shape}")


def write_model_dump(path: str | Path, records: Iterable[DumpRecord], side: int = 30) -> None:
    """Write records to the plan-approved top-K candidate npz format."""
    recs = list(records)
    if not recs:
        raise ValueError("model dump requires at least one record")

    task_ids: list[str] = []
    test_indices: list[int] = []
    record_kinds: list[str | None] = []
    cand_arrays: list[np.ndarray] = []
    vote_arrays: list[np.ndarray] = []
    max_k = 0
    has_explicit_kind = any(("record_kind" in rec or "kind" in rec) for rec in recs)
    if has_explicit_kind and not all(("record_kind" in rec or "kind" in rec) for rec in recs):
        raise ValueError("record_kind must be provided on every record in a typed model dump")

    for rec in recs:
        task_id = str(rec["task_id"])
        test_index = int(rec["test_index"])
        kind = rec.get("record_kind", rec.get("kind", None))
        if ("record_kind" in rec or "kind" in rec) and kind is None:
            raise ValueError("record_kind must be a non-null 'fold' or 'test' value")
        if kind is not None:
            kind = str(kind)
            if kind not in {"fold", "test"}:
                raise ValueError(f"record_kind must be 'fold' or 'test', got {kind!r}")
        cand = _as_candidate_array(rec["candidates"], side)
        votes = np.asarray(rec["vote_counts"], dtype=np.float32)
        if votes.ndim != 1 or votes.shape[0] != cand.shape[0]:
            raise ValueError(f"vote_counts must be [K] matching candidates; got {votes.shape} vs {cand.shape}")
        if cand.min(initial=0) < 0 or cand.max(initial=0) > 11:
            raise ValueError("candidate tokens must be in [0,11]")
        task_ids.append(task_id)
        test_indices.append(test_index)
        record_kinds.append(kind)
        cand_arrays.append(cand)
        vote_arrays.append(votes)
        max_k = max(max_k, cand.shape[0])

    n = len(recs)
    candidates = np.zeros((n, max_k, side * side), dtype=np.int64)
    vote_counts = np.full((n, max_k), -np.inf, dtype=np.float32)
    candidate_mask = np.zeros((n, max_k), dtype=bool)
    for i, (cand, votes) in enumerate(zip(cand_arrays, vote_arrays)):
        k = cand.shape[0]
        candidates[i, :k] = cand
        vote_counts[i, :k] = votes
        candidate_mask[i, :k] = True

    payload = {
        "task_ids": np.asarray(task_ids, dtype=object),
        "test_indices": np.asarray(test_indices, dtype=np.int64),
        "candidates": candidates,
        "vote_counts": vote_counts,
        "candidate_mask": candidate_mask,
        "side": np.asarray(side, dtype=np.int64),
    }
    if has_explicit_kind:
        payload["record_kinds"] = np.asarray([k or "test" for k in record_kinds], dtype=object)
    np.savez_compressed(Path(path), **payload)


def load_model_dump(
    path: str | Path,
    side: int = 30,
    kind: str | None = None,
    require_kind: bool = False,
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Load and validate a model dump, keyed by (task_id, test_index).

    Older dumps have no record kind and are returned as-is. New complete
    pipeline dumps can contain both LOOCV fold rows and target-test rows with
    the same `(task_id, index)`. For those typed dumps callers must request
    `kind="fold"` or `kind="test"` so the selector cannot silently collide the
    two namespaces.
    """
    if kind is not None and kind not in {"fold", "test"}:
        raise ValueError(f"kind must be 'fold', 'test', or None, got {kind!r}")
    with np.load(Path(path), allow_pickle=True) as data:
        required = {"task_ids", "test_indices", "candidates", "vote_counts"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"model dump missing keys: {sorted(missing)}")

        task_ids = data["task_ids"].copy()
        test_indices = data["test_indices"].copy()
        candidates = data["candidates"].copy()
        vote_counts = data["vote_counts"].copy()
        candidate_mask = (
            data["candidate_mask"].copy()
            if "candidate_mask" in data.files
            else np.ones_like(vote_counts, dtype=bool)
        )
        record_kinds = data["record_kinds"].copy() if "record_kinds" in data.files else None
        dump_side = int(data["side"]) if "side" in data.files else side

    if record_kinds is None and require_kind and kind is not None:
        raise ValueError(f"model dump has no record_kind namespace; required kind={kind!r}")
    if record_kinds is not None and kind is None:
        raise ValueError("typed model dump contains record_kind namespaces; call load_model_dump(..., kind='fold'|'test')")
    if dump_side != side:
        raise ValueError(f"model dump side={dump_side}, expected {side}")
    if candidates.ndim != 3 or candidates.shape[2] != side * side:
        raise ValueError(f"candidates must be [N,K,{side * side}], got {candidates.shape}")
    if vote_counts.shape != candidates.shape[:2]:
        raise ValueError(f"vote_counts shape {vote_counts.shape} must match candidates first dims {candidates.shape[:2]}")
    if candidate_mask.shape != candidates.shape[:2]:
        raise ValueError(f"candidate_mask shape {candidate_mask.shape} must match candidates first dims {candidates.shape[:2]}")
    if len(task_ids) != candidates.shape[0] or len(test_indices) != candidates.shape[0]:
        raise ValueError("task_ids/test_indices length must match candidate records")
    if record_kinds is not None and len(record_kinds) != candidates.shape[0]:
        raise ValueError("record_kinds length must match candidate records")
    if candidates.min(initial=0) < 0 or candidates.max(initial=0) > 11:
        raise ValueError("candidate tokens must be in [0,11]")

    out: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for i in range(candidates.shape[0]):
        if record_kinds is not None and str(record_kinds[i]) != kind:
            continue
        key = (str(task_ids[i]), int(test_indices[i]))
        mask = candidate_mask[i].astype(bool)
        cand = candidates[i][mask].astype(np.int64, copy=False)
        votes = vote_counts[i][mask].astype(np.float32, copy=False)
        if cand.shape[0] == 0:
            raise ValueError(f"record {key} has no valid candidates")
        if key in out:
            raise ValueError(f"duplicate model dump record for {key}")
        out[key] = {"candidates": cand, "vote_counts": votes}
    return out
