#!/usr/bin/env python3
"""
inventory_notebook.py
=====================
Parse the executed master notebook and emit a per-cell evidence inventory.

Authoritative source: the executed notebook itself
(``Beyond_Arbitrariness_BetaBinomial_Diagnostics_final.ipynb``).

Output: ``supplement/generated/notebook_inventory.csv`` with one row per cell.

Objective fields (index, id, type, execution count, markdown heading, defined
variables, referenced input files, written output files, displayed output type)
are extracted by parsing cell source and stored outputs.  The two interpretive
fields (``model_or_analysis_name`` and ``inferred_status``) are assigned by the
transparent keyword rules in :func:`classify_status`.

No model is executed.  This only *reads* the notebook.
"""
from __future__ import annotations

import ast
import csv
import os
import re

import nbformat


def executes_fit(src: str) -> bool:
    """True iff the cell *executes* a model fit at module scope.

    We parse the cell and look for a call to a fitter (``fit_*`` / ``run_subset_primary``)
    or an ``mcmc.run(...)`` that is NOT inside a ``def`` body.  This distinguishes cells
    that merely *define* fitting helpers/models (infrastructure) from cells that actually
    run NUTS and save a posterior (an inferential fit).
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # notebooks sometimes contain non-python magics; fall back to a conservative regex
        return bool(re.search(r"^(?!\s*def\b)\s*[A-Za-z_][^\n]*\bfit_[A-Za-z_]+\s*\(", src, re.M))

    found = [False]

    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # skip definition bodies — these are not executions
            if isinstance(child, ast.Call):
                f = child.func
                name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
                if name.startswith("fit_") or name in {"run", "run_subset_primary"}:
                    found[0] = True
            walk(child)

    walk(tree)
    return found[0]

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NB_PATH = os.path.join(REPO, "Beyond_Arbitrariness_BetaBinomial_Diagnostics_final.ipynb")
OUT_PATH = os.path.join(REPO, "supplement", "generated", "notebook_inventory.csv")

# ---------------------------------------------------------------------------
# Extraction helpers (objective, regex based)
# ---------------------------------------------------------------------------
_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)", re.MULTILINE)
_DEF = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

_READ = re.compile(
    r"(?:read_csv|read_parquet|read_json|np\.load|xr\.open_dataset|az\.from_netcdf)\s*\(\s*([^)\n,]+)"
)
_OPEN_R = re.compile(r"open\s*\(\s*([^,\n)]+)\s*,\s*['\"][rR]")
_CLDF = re.compile(
    r"['\"]([^'\"]*(?:cldf|forms\.csv|languages\.csv|parameters\.csv|values\.csv)[^'\"]*)['\"]",
    re.IGNORECASE,
)
_WRITE = re.compile(
    r"(?:to_csv|to_parquet|to_json|savefig|save_paper_figure|to_netcdf)\s*\(\s*([^)\n,]+)"
)
_OPEN_W = re.compile(r"open\s*\(\s*([^,\n)]+)\s*,\s*['\"][wax]")
_FNAME = re.compile(r"['\"]([\w./-]+\.(?:csv|parquet|json|nc|pdf|png|svg))['\"]")


def _clean(expr: str) -> str:
    return expr.strip().strip("(),")


def defined_variables(src: str) -> str:
    names = []
    for m in _ASSIGN.finditer(src):
        names.append(m.group(1))
    for m in _DEF.finditer(src):
        names.append(m.group(1))
    seen = []
    for n in names:
        if n not in seen and not n.startswith("__"):
            seen.append(n)
    return "; ".join(seen[:40])


def referenced_input_files(src: str) -> str:
    hits = []
    for rx in (_READ, _OPEN_R, _CLDF):
        for m in rx.finditer(src):
            hits.append(_clean(m.group(1)))
    seen = []
    for h in hits:
        if h and h not in seen:
            seen.append(h)
    return "; ".join(seen[:20])


def written_output_files(src: str) -> str:
    hits = []
    for rx in (_WRITE, _OPEN_W):
        for m in rx.finditer(src):
            hits.append(_clean(m.group(1)))
    if re.search(r"to_csv|savefig|to_parquet|to_json|save_paper_figure|to_netcdf", src):
        for m in _FNAME.finditer(src):
            hits.append(m.group(1))
    seen = []
    for h in hits:
        h = h.strip()
        if h and h not in seen:
            seen.append(h)
    return "; ".join(seen[:25])


def displayed_output_type(cell) -> str:
    outs = cell.get("outputs", []) or []
    if not outs:
        return "none"
    types = []
    for o in outs:
        ot = o.get("output_type", "")
        if ot == "display_data":
            data = o.get("data", {})
            ot = "image" if any(k.startswith("image/") for k in data) else "display_data"
        types.append(ot)
    seen = []
    for t in types:
        if t not in seen:
            seen.append(t)
    return "; ".join(seen)


def markdown_heading(src: str) -> str:
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()
    for line in src.splitlines():
        if line.strip():
            return line.strip()[:120]
    return ""


# ---------------------------------------------------------------------------
# Interpretive classification (transparent keyword rules)
# ---------------------------------------------------------------------------
def model_or_analysis_name(src: str, outputs: str) -> str:
    defs = re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*model[A-Za-z0-9_]*)\s*\(", src)
    if defs:
        return "; ".join(sorted(set(defs)))
    stems = re.findall(r"([a-z0-9_]+)\.(?:csv|pdf|png)", outputs)
    if stems:
        return stems[0]
    if re.search(r"fit_dataset|fit_numpyro_model|run_subset_primary", src):
        return "numpyro fit"
    return ""


def classify_status(cell_type: str, src: str, outputs: str, heading: str) -> tuple[str, str]:
    """Return (inferred_status, note).  Allowed statuses:
    primary, replication, robustness, diagnostic, exploratory, superseded, infrastructure.
    """
    low = (src + " " + outputs + " " + heading).lower()
    if cell_type == "markdown":
        return "infrastructure", "narrative/section heading"

    # --- infrastructure: setup, io, figures, export ---
    if re.search(r"pip install|import numpyro|clone_or_pull|jax\.config|reproducibility_metadata", src):
        return "infrastructure", "environment / setup / provenance"
    if re.search(r"save_paper_figure|def build_asjp|def build_lexibank|def aggregate_language|"
                 r"ASJP_VOWELS|IPA_VOWELS|files\.download|zipfile|shutil\.make_archive", src):
        return "infrastructure", "preprocessing definitions / figure IO / export"
    if re.search(r"figure[1-4]|figures/figure", low) and "savefig" in low:
        return "infrastructure", "main-figure rendering"

    # "beta_binomial" appears in the source (model name), outputs, or heading of the BB layer
    is_bb = "beta_binomial" in low or "betabinomial" in low
    # A model is actually fit in this cell iff a fitter executes at module scope.
    runs_fit = executes_fit(src)

    # --- leave-family-out / leave-top-two influence re-fits (all Beta-Binomial) ---
    if re.search(r"exclusion_sensitivity|excluding_top_two", low) and runs_fit:
        return "robustness", "leave-family-out / leave-top-two influence re-fit (Beta-Binomial)"

    # --- a model was actually fit and saved in this cell ---
    if runs_fit:
        if is_bb:
            olow = outputs.lower()
            if "primary" in olow and "within" not in olow and "matched" not in olow:
                return "primary", "Beta-Binomial primary reported model"
            if "replication" in olow:
                return "replication", "Beta-Binomial Lexibank replication"
            return "robustness", "Beta-Binomial conditional / secondary reported analysis"
        # non-Beta-Binomial inferential fit = the earlier, superseded layer
        note = "Binomial-likelihood inferential fit (superseded by Beta-Binomial)"
        if "ppc_summary" in outputs.lower():
            note += "; PPC output reused in Figure S2 dispersion comparison"
        return "superseded", note

    # --- non-fit cells: diagnostics / PPC / audits / descriptive / reporting tables ---
    if "prior_predictive" in low:
        return "exploratory", "prior predictive check"
    if re.search(r"ppc|posterior_predictive|boundary|family_boundary|diagnostics|rhat|r_hat|"
                 r"\bess\b|divergen|family_audit|sample_flow|descriptive|band_summary|audit|"
                 r"decision_table|statistical_table|final_effects|final_diagnostics", low):
        return "diagnostic", "diagnostic / PPC / audit / descriptive / reporting table"

    return "infrastructure", "supporting code / definitions / figure IO / export"


# ---------------------------------------------------------------------------
def main() -> int:
    nb = nbformat.read(NB_PATH, as_version=4)
    rows = []
    for idx, cell in enumerate(nb.cells):
        ctype = cell.cell_type
        src = cell.get("source", "") or ""
        cid = cell.get("id", "") or ""
        ec = cell.get("execution_count", None)
        heading = markdown_heading(src) if ctype == "markdown" else ""
        outs = written_output_files(src) if ctype == "code" else ""
        ins = referenced_input_files(src) if ctype == "code" else ""
        dvars = defined_variables(src) if ctype == "code" else ""
        dtype = displayed_output_type(cell) if ctype == "code" else "none"
        mname = model_or_analysis_name(src, outs) if ctype == "code" else ""
        status, note = classify_status(ctype, src, outs, heading)
        rows.append({
            "cell_index": idx,
            "cell_id": cid,
            "cell_type": ctype,
            "markdown_heading": heading,
            "execution_count": "" if ec is None else ec,
            "defined_variables": dvars,
            "referenced_input_files": ins,
            "written_output_files": outs,
            "displayed_output_type": dtype,
            "model_or_analysis_name": mname,
            "inferred_status": status,
            "notes": note,
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cols = ["cell_index", "cell_id", "cell_type", "markdown_heading", "execution_count",
            "defined_variables", "referenced_input_files", "written_output_files",
            "displayed_output_type", "model_or_analysis_name", "inferred_status", "notes"]
    with open(OUT_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # brief stdout summary
    from collections import Counter
    code = [r for r in rows if r["cell_type"] == "code"]
    status_counts = Counter(r["inferred_status"] for r in code)
    print(f"Wrote {OUT_PATH}")
    print(f"Total cells: {len(rows)} ({len(code)} code, {len(rows) - len(code)} markdown)")
    print("Code-cell status counts:")
    for k, v in sorted(status_counts.items()):
        print(f"  {k:14s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
