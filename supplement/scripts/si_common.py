#!/usr/bin/env python3
"""
si_common.py
============
Shared paths and helpers for the Supplementary Information build scripts.

All numeric values in the SI must come from the Result CSV files in ``results/``.
This module centralises CSV loading and row selection so no script transcribes
numbers by hand.  It performs no analysis and executes no model.
"""
from __future__ import annotations

import csv
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "results")
FIGURES = os.path.join(REPO, "figures")
NB_NAME = "Beyond_Arbitrariness_BetaBinomial_Diagnostics_final.ipynb"
NB_PATH = os.path.join(REPO, NB_NAME)

SUP = os.path.join(REPO, "supplement")
TABLES = os.path.join(SUP, "tables")
SUPFIG = os.path.join(SUP, "figures")
GEN = os.path.join(SUP, "generated")
SCRIPTS = os.path.join(SUP, "scripts")

for _d in (SUP, TABLES, SUPFIG, GEN, SCRIPTS):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
def load_csv(name: str) -> list[dict]:
    """Load a CSV from results/ as a list of ordered dicts (strings)."""
    path = name if os.path.isabs(name) else os.path.join(RESULTS, name)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def csv_path(name: str) -> str:
    return os.path.join(RESULTS, name)


def find_row(rows: list[dict], **eq) -> dict:
    """Return the single row whose columns equal all key=value pairs; error if not exactly one."""
    hits = [r for r in rows if all(str(r.get(k, "")).strip() == str(v) for k, v in eq.items())]
    if len(hits) != 1:
        raise ValueError(f"find_row expected 1 match for {eq}, got {len(hits)}")
    return hits[0]


def fnum(x, nd: int = 3) -> str:
    """Format a numeric string/float to nd decimals; passthrough for blanks."""
    if x is None or x == "":
        return ""
    return f"{float(x):.{nd}f}"


def to_f(x):
    if x is None or x == "":
        return None
    return float(x)


def to_i(x):
    if x is None or x == "":
        return None
    return int(float(x))


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


# LaTeX-escape helper for table cell text
_LATEX = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
          "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def latex_escape(s: str) -> str:
    s = "" if s is None else str(s)
    return "".join(_LATEX.get(c, c) for c in s)
