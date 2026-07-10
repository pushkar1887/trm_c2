from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_value_v2_rich_ctx_cli_flag_is_exposed_and_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_stage1_local.py"
    help_run = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    assert "--value-v2-rich-ctx" in help_run.stdout

    source = script.read_text(encoding="utf-8")
    assert "c2_value_v2_rich_ctx" in source
    assert "args.value_v2_rich_ctx" in source
    assert "V2TAIL" in source
    assert "color_head.weight.grad" in source
    assert "VALUE_EVIDENCE_V2_DIM" in source


def test_rule_hypothesis_hint_cli_flag_is_exposed_and_wired() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_stage1_local.py"
    help_run = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    assert "--rule-hypothesis-hint" in help_run.stdout

    source = script.read_text(encoding="utf-8")
    assert "c2_rule_hypothesis_hint" in source
    assert "args.rule_hypothesis_hint" in source
    assert "rule-hypothesis" in source
    assert "RULEHYP" in source
    assert "c2_rule_hyp_norm" in source
    assert '"frame_embed",' in source
    assert '"rule_hyp_embed",' in source


if __name__ == "__main__":
    test_value_v2_rich_ctx_cli_flag_is_exposed_and_wired()
    test_rule_hypothesis_hint_cli_flag_is_exposed_and_wired()
    print("test_run_stage1_local_cli: PASS")
