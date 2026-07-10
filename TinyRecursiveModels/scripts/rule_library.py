"""Stage 5 — the RULE MEMORY (the discovered half of the two-sided library).

Architecture (plan: Parse -> Propose -> Verify -> Execute -> ABSTRACT): when the program search finds a
program that reconstructs a task's demos EXACTLY (LODO-verified), we FACTOR IT OUT and store it as a reusable
composite primitive (a "macro"). Future tasks try the stored macros as 1-step candidates BEFORE re-searching
the full composition space -- so each solved task makes later tasks cheaper/reachable (DreamCoder-style
abstraction). The library's OTHER half is the core-knowledge base primitives (recolor, geometric, neighbour,
object ops); those live in check_extraction.py. This module is JUST the store: it holds RECIPES (ordered op
names), not fitted parameters -- the parameters are always re-fit cross-demo on the new task (the cross-demo
pillar). Persists to JSON so it carries across runs (train OR inference time).

Self-contained + self-tested (build-the-check-before-the-feature). The executor that runs a recipe is supplied
by the caller (check_extraction) so this module has no heavy deps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _canon(recipe: list[str]) -> str:
    """Canonical dedup key for a recipe = the ordered op-name sequence joined."""
    return ">".join(recipe)


class RuleLibrary:
    """A growing set of discovered macros. A MACRO = an ordered list of base-op names (the program
    STRUCTURE), e.g. ["dihedral", "recolor"] meaning 'apply a (demo-agreed) dihedral op, then fit a
    per-colour recolor map'. Parameters are NOT stored -- they are re-fit on each new task's demos.
    Each macro carries lightweight provenance: how many tasks it has solved, and which families."""

    def __init__(self) -> None:
        self._macros: dict[str, dict] = {}     # canon-key -> {recipe, solved, families}

    # ---- write side (ABSTRACT) -------------------------------------------------------------------
    def add(self, recipe: list[str], family: str | None = None) -> bool:
        """Store a recipe discovered to be LODO-exact. Returns True if it is NEW (first sighting).
        A length-1 recipe is a base primitive (already in the DSL) -- we only abstract length>=2
        (genuine compositions), matching DreamCoder's 'factor out useful new combinations'."""
        if len(recipe) < 2:
            return False                       # not a macro -- a base primitive, don't store
        k = _canon(recipe)
        new = k not in self._macros
        m = self._macros.setdefault(k, {"recipe": list(recipe), "solved": 0, "families": {}})
        m["solved"] += 1
        if family is not None:
            m["families"][family] = m["families"].get(family, 0) + 1
        return new

    # ---- read side (reuse in PROPOSE) ------------------------------------------------------------
    def recipes(self) -> list[list[str]]:
        """The stored macro recipes, most-solved first (try the most useful macros earliest)."""
        return [m["recipe"] for m in sorted(self._macros.values(), key=lambda x: -x["solved"])]

    def __contains__(self, recipe: list[str]) -> bool:
        return _canon(recipe) in self._macros

    def __len__(self) -> int:
        return len(self._macros)

    def __iter__(self) -> Iterable[list[str]]:
        return iter(self.recipes())

    # ---- persistence (carries across runs) -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(list(self._macros.values()), indent=2))

    def load(self, path: str | Path) -> "RuleLibrary":
        p = Path(path)
        if p.exists():
            for m in json.loads(p.read_text()):
                k = _canon(m["recipe"])
                self._macros[k] = {"recipe": m["recipe"], "solved": int(m.get("solved", 1)),
                                   "families": dict(m.get("families", {}))}
        return self

    def summary(self) -> str:
        if not self._macros:
            return "RuleLibrary: empty"
        rows = sorted(self._macros.values(), key=lambda x: -x["solved"])
        head = ", ".join(f"[{'>'.join(m['recipe'])}]x{m['solved']}" for m in rows[:8])
        return f"RuleLibrary: {len(self._macros)} macros | top: {head}"


def _self_test() -> None:
    lib = RuleLibrary()
    # length-1 recipe is a base primitive -> NOT abstracted
    assert lib.add(["recolor"]) is False and len(lib) == 0, "length-1 must not be stored as a macro"
    # a genuine 2-op composition is new the first time, not new the second; solved-count increments
    assert lib.add(["dihedral", "recolor"], family="size_change") is True, "first 2-op macro is NEW"
    assert lib.add(["dihedral", "recolor"], family="size_change") is False, "duplicate is not NEW"
    assert len(lib) == 1, "dedup: identical recipe stored once"
    assert ["dihedral", "recolor"] in lib and ["recolor", "dihedral"] not in lib, "order matters"
    lib.add(["crop", "recolor"], family="conditional_recolor")
    assert len(lib) == 2
    # most-solved-first ordering: dihedral>recolor solved twice, ranks before crop>recolor
    assert lib.recipes()[0] == ["dihedral", "recolor"], "most-solved macro ranks first"
    # save/load roundtrip carries the macros + counts across runs
    import tempfile, os
    p = os.path.join(tempfile.gettempdir(), "_rule_lib_selftest.json")
    lib.save(p)
    lib2 = RuleLibrary().load(p)
    assert len(lib2) == 2 and ["dihedral", "recolor"] in lib2, "load must restore macros"
    assert lib2.recipes()[0] == ["dihedral", "recolor"], "load must restore solved-order"
    os.remove(p)
    print(f"rule_library self-test PASS  ({lib.summary()})")


if __name__ == "__main__":
    _self_test()
