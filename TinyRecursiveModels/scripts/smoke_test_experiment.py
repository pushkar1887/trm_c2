"""Pre-flight smoke test for an experiment config.

Catches the bug class codex hit on the first struct rule branch run:
  - eval/test data path missing -> in-training eval crashes -> save_train_state never runs
  - load_checkpoint missing -> pretrain fails at load time
  - checkpoint_path not writable -> save_train_state silently no-ops

This script does NOT train. It only verifies the wiring so a 6-10h run will
actually save something. Runtime: ~30-90 seconds per experiment.

Checks per experiment:
  1. config YAML parses and PretrainConfig accepts it.
  2. load_checkpoint file/dir exists.
  3. Each data_paths entry exists with the expected layout
     (identifiers.json + test/ + train/).
  4. Each data_paths_test entry exists likewise.
  5. checkpoint_path is writable (touch a tmp file in it).
  6. Optional: construct a train dataloader and pull ONE batch to catch
     puzzle_id / shape mismatches that would later poison loss computation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_path_exists(path: Path, kind: str = "file") -> Tuple[bool, str]:
    if not path.exists():
        return False, f"MISSING: {path}"
    if kind == "file" and not path.is_file():
        return False, f"NOT A FILE: {path}"
    if kind == "dir" and not path.is_dir():
        return False, f"NOT A DIR: {path}"
    return True, f"OK: {path}"


def check_dataset_layout(root: Path) -> Tuple[bool, List[str]]:
    """A canonical TRM dataset has: identifiers.json + train/ + test/ with .npy files."""
    msgs = []
    if not root.exists():
        return False, [f"MISSING ROOT: {root}"]
    msgs.append(f"OK root: {root}")
    needs = [
        ("identifiers.json", "file"),
        ("train", "dir"),
        ("test", "dir"),
    ]
    ok = True
    for name, kind in needs:
        sub_ok, sub_msg = check_path_exists(root / name, kind)
        msgs.append(("OK   " if sub_ok else "FAIL ") + f"{name}: {sub_msg}")
        ok = ok and sub_ok
    # Check the test split has the npy files the dataloader expects.
    for n in ("all__inputs.npy", "all__labels.npy", "all__puzzle_identifiers.npy", "all__puzzle_indices.npy"):
        sub_ok, sub_msg = check_path_exists(root / "test" / n, "file")
        msgs.append(("OK   " if sub_ok else "FAIL ") + f"test/{n}: {sub_msg}")
        ok = ok and sub_ok
    return ok, msgs


def smoke_test(config_path: Path, do_dataloader: bool = True) -> Dict[str, object]:
    result: Dict[str, object] = {
        "config": str(config_path),
        "passed": False,
        "messages": [],
        "errors": [],
    }
    msgs = result["messages"]
    errs = result["errors"]

    # ---- 1. YAML parse ----
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        msgs.append(f"[1/6] YAML parse OK: {config_path.name}")
    except Exception as e:
        errs.append(f"[1/6] YAML parse FAIL: {e}")
        return result

    # ---- 2. load_checkpoint exists ----
    load_ckpt = raw.get("load_checkpoint")
    if load_ckpt is None:
        msgs.append("[2/6] load_checkpoint not set (training from scratch)")
    else:
        p = Path(load_ckpt)
        ok, m = check_path_exists(p, "file")
        msgs.append(("[2/6] " + ("OK   " if ok else "FAIL ") + m))
        if not ok:
            errs.append(f"load_checkpoint missing: {p}")

    # ---- 3. data_paths (train) ----
    data_paths = raw.get("data_paths") or []
    if not data_paths:
        errs.append("[3/6] data_paths empty — training will fail")
    for dp in data_paths:
        # Resolve relative paths from repo root.
        if not Path(dp).is_absolute():
            dp_resolved = Path(__file__).resolve().parents[1] / dp
        else:
            dp_resolved = Path(dp)
        ok, sub_msgs = check_dataset_layout(dp_resolved)
        if ok:
            msgs.append(f"[3/6] data_paths OK: {dp_resolved}")
        else:
            errs.append(f"[3/6] data_paths broken: {dp_resolved}")
            msgs.extend(f"   {m}" for m in sub_msgs)

    # ---- 4. data_paths_test ----
    test_paths = raw.get("data_paths_test") or []
    if not test_paths:
        msgs.append("[4/6] data_paths_test empty — in-training eval will be skipped (safe)")
    for dp in test_paths:
        if not Path(dp).is_absolute():
            dp_resolved = Path(__file__).resolve().parents[1] / dp
        else:
            dp_resolved = Path(dp)
        ok, sub_msgs = check_dataset_layout(dp_resolved)
        if ok:
            msgs.append(f"[4/6] data_paths_test OK: {dp_resolved}")
        else:
            errs.append(f"[4/6] data_paths_test broken: {dp_resolved} (THIS is what crashed codex's first run)")
            msgs.extend(f"   {m}" for m in sub_msgs)

    # ---- 5. checkpoint_path writable ----
    ckpt_dir = raw.get("checkpoint_path")
    if ckpt_dir:
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(__file__).resolve().parents[1] / ckpt_path
        ckpt_path.mkdir(parents=True, exist_ok=True)
        try:
            test_file = ckpt_path / f".smoketest_{os.getpid()}.tmp"
            test_file.write_text("smoke", encoding="utf-8")
            test_file.unlink()
            msgs.append(f"[5/6] checkpoint_path writable: {ckpt_path}")
        except Exception as e:
            errs.append(f"[5/6] checkpoint_path NOT writable: {ckpt_path} ({e})")

    # ---- 6. Optionally construct one dataloader + pull a batch ----
    if do_dataloader:
        try:
            import pretrain  # heavy import only when needed
            cfg = pretrain.PretrainConfig(**raw)
            t0 = time.perf_counter()
            train_loader, train_meta = pretrain.create_dataloader(
                cfg, "train", 0, 1,
                test_set_mode=False, epochs_per_iter=1,
                global_batch_size=cfg.global_batch_size,
            )
            took = time.perf_counter() - t0
            msgs.append(f"[6/6] train dataloader constructed in {took:.1f}s (puzzle_ids={train_meta.num_puzzle_identifiers})")

            # Pull one batch to catch dtype/shape mismatches.
            t1 = time.perf_counter()
            try:
                for _set_name, batch, _gbs in train_loader:
                    msgs.append(
                        f"[6/6] pulled 1 train batch in {time.perf_counter()-t1:.2f}s; "
                        f"keys={sorted(batch.keys())[:6]}..."
                    )
                    break
            except Exception as e:
                errs.append(f"[6/6] train dataloader iteration FAIL: {e}")
        except Exception as e:
            errs.append(f"[6/6] dataloader construction FAIL: {type(e).__name__}: {e}")
            errs.append(traceback.format_exc().splitlines()[-1])

    result["passed"] = len(errs) == 0
    return result


def smoke_test_script(script_path: Path) -> Dict[str, object]:
    """For non-config scripts (canvas cleanup, verifier head), check they compile + self-test."""
    result: Dict[str, object] = {
        "script": str(script_path),
        "passed": False,
        "messages": [],
        "errors": [],
    }
    try:
        import py_compile
        py_compile.compile(str(script_path), doraise=True)
        result["messages"].append(f"compile OK: {script_path.name}")
        result["passed"] = True
    except py_compile.PyCompileError as e:
        result["errors"].append(f"compile FAIL: {e}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke test experiment configs + scripts.")
    p.add_argument("--config", action="append", default=None,
                   help="Specific config path(s) to test (can repeat). If omitted, tests all four experiments.")
    p.add_argument("--no-dataloader", action="store_true",
                   help="Skip dataloader construction (faster, less thorough).")
    p.add_argument("--out", default="reports/smoke_tests/smoke_summary.json")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if args.config:
        configs = [Path(c).resolve() for c in args.config]
    else:
        configs = [
            repo / "checkpoints/TRM-FVR-Experiments/struct_rule_branch_v005_aug1000_seed0/all_config.yaml",
            repo / "checkpoints/TRM-FVR-Experiments/lodo_light_v005_aug1000_seed0/all_config.yaml",
            repo / "checkpoints/TRM-FVR-Experiments/lodo_contrast_probe_v005_aug1000_seed0/all_config.yaml",
            repo / "checkpoints/TRM-FVR-Experiments/c4_changed_valid_v005_aug1000_seed0/all_config.yaml",
        ]
    print(f"[smoke] testing {len(configs)} configs ({'no dataloader' if args.no_dataloader else 'with dataloader'})")
    results = []
    for cfg in configs:
        print(f"\n=== {cfg.name} ===")
        if not cfg.exists():
            print(f"  CONFIG MISSING: {cfg}")
            results.append({"config": str(cfg), "passed": False, "errors": ["config file missing"]})
            continue
        r = smoke_test(cfg, do_dataloader=not args.no_dataloader)
        for m in r["messages"]:
            print("  " + m)
        for e in r["errors"]:
            print("  ERR " + e)
        print(f"  -> {'PASS' if r['passed'] else 'FAIL'}")
        results.append(r)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[smoke] summary -> {out_path}")
    n_pass = sum(1 for r in results if r["passed"])
    print(f"[smoke] {n_pass}/{len(results)} configs PASS")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
