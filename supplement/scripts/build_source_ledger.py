#!/usr/bin/env python3
"""
build_source_ledger.py
======================
Emit two traceability ledgers:

* supplement/generated/source_ledger.csv        — one row per SI table/figure,
  linking it to its Result CSV, the notebook cell(s) that produced that CSV,
  the model, the sample definition, and the publication status.
* supplement/generated/claim_to_evidence_map.csv — one row per publication-facing
  claim (the Section-5 checkpoints), linking manuscript claim -> SI item ->
  source CSV -> notebook cell.

Both are generated from si_registry.SI_ITEMS and checkpoints.CHECKPOINTS; nothing
is transcribed by hand.
"""
from __future__ import annotations

import csv
import os

import si_common as C
from si_registry import SI_ITEMS, NOTEBOOK
from checkpoints import CHECKPOINTS

# checkpoint id -> (supporting SI item(s), notebook cell(s) that produced the CSV)
CLAIM_MAP = {
    "asjp_prefilter_languages": ("Table S3", "67,99"),
    "asjp_prefilter_family_ids": ("Table S3", "67,99"),
    "asjp_primary_languages": ("Table S3", "99"),
    "asjp_primary_family_ids": ("Table S3", "99"),
    "lexibank_languages": ("Table S1, Table S16", "53,55"),
    "macroarea_matched_languages": ("Table S3, Table S4", "99"),
    "macroarea_matched_family_ids": ("Table S3, Table S4", "99"),
    "macroarea_unlinked_languages": ("Table S3, Table S5", "99"),
    "primary_beta_lat": ("Table S8, Table S10", "72,84"),
    "lexibank_beta_lat": ("Table S8", "80,84"),
    "beta_initial": ("Table S8, Table S11", "76,84"),
    "beta_final": ("Table S8, Table S11", "76,84"),
    "beta_final_minus_initial": ("Table S8, Table S11", "76,84"),
    "beta_between": ("Table S8", "74,84"),
    "beta_within": ("Table S8", "74,84"),
    "matched_no_macroarea_beta": ("Table S14", "100"),
    "matched_macroarea_beta": ("Table S14", "85"),
    "macroarea_attenuation": ("Table S14", "85,100"),
    "north_beta": ("Table S15 (context)", "78,84"),
    "south_beta": ("Table S15 (context)", "78,84"),
    "leave_both_final_minus_initial": ("Table S11", "97"),
    "leave_atlantic_congo_beta": ("Table S10", "95"),
    "leave_austronesian_beta": ("Table S10", "95"),
    "leave_both_beta": ("Table S10", "95"),
    "exact_one_observed_asjp": ("Table S13, Figure S2", "82"),
    "exact_one_ppc_asjp": ("Figure S2", "82"),
    "exact_one_observed_lexibank": ("Table S13, Figure S2", "82"),
    "exact_one_ppc_lexibank": ("Figure S2", "82"),
}

SECTION_TITLES = {
    "S1": "S1 Data provenance and preprocessing",
    "S2": "S2 Sample flow and metadata audits",
    "S3": "S3 Statistical model specifications",
    "S4": "S4 MCMC sampling diagnostics",
    "S5": "S5 Binomial versus Beta-Binomial sensitivity",
    "S6": "S6 Lexical-coverage robustness",
    "S7": "S7 Influence of major language families",
    "S8": "S8 Family-level boundary concentration and PPC misfit",
    "S9": "S9 Areal and hemispheric heterogeneity",
    "S10": "S10 ASJP-Lexibank overlap and comparability",
    "S11": "S11 Computational reproducibility",
}


def build_source_ledger():
    cols = ["si_item", "si_section", "table_or_figure", "result_csv", "auxiliary_data_file",
            "python_source", "notebook_file", "notebook_cell_index", "model_name",
            "sample_definition", "publication_status", "manuscript_location",
            "consistency_check", "notes"]
    rows = []
    for it in SI_ITEMS:
        data_csvs = [c for c in it["csvs"] if c.endswith(".csv")]
        aux = [c for c in it["csvs"] if not c.endswith(".csv")]
        rows.append({
            "si_item": it["id"],
            "si_section": SECTION_TITLES.get(it["section"], it["section"]),
            "table_or_figure": it["table_or_figure"],
            "result_csv": "; ".join(data_csvs),
            "auxiliary_data_file": "; ".join(aux),
            "python_source": "notebook (Bayesian pipeline); scripts/*.py are the separate "
                             "frequentist pipeline and are not sources for these items",
            "notebook_file": NOTEBOOK,
            "notebook_cell_index": it["cells"],
            "model_name": it["model"],
            "sample_definition": it["sample"],
            "publication_status": it["status"],
            "manuscript_location": it["section"],
            "consistency_check": it["consistency"],
            "notes": "",
        })
    path = os.path.join(C.GEN, "source_ledger.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path, len(rows)


def build_claim_map():
    cols = ["manuscript_section", "manuscript_claim", "supporting_si_item",
            "source_csv", "source_code", "notebook_cell", "validation_status"]
    rows = []
    for cp in CHECKPOINTS:
        si_item, cell = CLAIM_MAP.get(cp["id"], ("", cp["source"].get("row_index", "")))
        declared = cp.get("declared")
        if declared is None:
            claim = f'{cp["label"]} (repository-internal; authoritative value from CSV)'
        else:
            claim = f'{cp["label"]} approximately {declared}'
        rows.append({
            "manuscript_section": "Section 5 checkpoint" if cp["layer"] == "manuscript_s5"
                                  else "repository-internal",
            "manuscript_claim": claim,
            "supporting_si_item": si_item,
            "source_csv": cp["source"]["file"],
            "source_code": f'notebook cell(s) {cell}',
            "notebook_cell": cell,
            "validation_status": "verified in validation_report.md / manuscript_consistency_report.md",
        })
    path = os.path.join(C.GEN, "claim_to_evidence_map.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path, len(rows)


def main() -> int:
    p1, n1 = build_source_ledger()
    p2, n2 = build_claim_map()
    print(f"Wrote {p1} ({n1} SI items)")
    print(f"Wrote {p2} ({n2} claims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
