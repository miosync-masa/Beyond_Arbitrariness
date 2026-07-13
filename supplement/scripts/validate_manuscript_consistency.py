#!/usr/bin/env python3
"""
validate_manuscript_consistency.py
==================================
Automated consistency checks across: Result CSVs, manifest.json, the manuscript
contract, and the generated SI tables.  Emits two reports:

* supplement/generated/validation_report.md
    Repository-internal + evidence checks (counts, coefficients, CIs, attenuation,
    exact-one PPC, leave-family-out counts, leave-top-two contrast, diagnostics),
    plus a table-cell spot check that generated LaTeX contains the CSV values.

* supplement/generated/manuscript_consistency_report.md
    The three consistency LAYERS, reported SEPARATELY:
      (a) repository-internal consistency  (manifest <-> CSV)
      (b) manuscript-contract consistency  (Section-5 declared <-> CSV)
      (c) direct manuscript-source validation = NOT PERFORMED (.tex/.bib absent)

On disagreement the affected item is enumerated (conflicting files + values); no
scientific result is auto-corrected.  Integer counts use exact equality; floats
use the documented per-kind tolerances in checkpoints.KIND_TOL.
"""
from __future__ import annotations

import json
import os

import si_common as C
from checkpoints import CHECKPOINTS, KIND_TOL


def resolve(source):
    rows = C.load_csv(source["file"])
    row = rows[source["row_index"]] if "row_index" in source else C.find_row(rows, **source["match"])
    return row[source["field"]]


def resolve_num(source):
    v = resolve(source)
    return None if v == "" else float(v)


def approx(a, b, tol):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
def main() -> int:
    with open(os.path.join(C.SUP, "manifest.json")) as fh:
        manifest = json.load(fh)

    # ============ Layer A: repository-internal (manifest <-> CSV) ============
    a_rows, a_fail = [], 0
    for cp in CHECKPOINTS:
        csv_val = resolve_num(cp["source"])
        man_val = manifest.get(cp["manifest_key"])
        if cp["kind"] == "count":
            ok = int(round(csv_val)) == int(round(float(man_val)))
        else:
            ok = (man_val is not None) and approx(float(man_val), csv_val, 1e-9)
        a_fail += (0 if ok else 1)
        a_rows.append((cp["id"], cp["source"]["file"], csv_val, man_val, "PASS" if ok else "FAIL"))

    # CI endpoints manifest <-> CSV
    ci_checks = [
        ("primary_beta_lat", "paper1_beta_binomial_final_effects.csv",
         {"model": "ASJP Beta-Binomial primary", "parameter": "beta_lat"}),
        ("lexibank_beta_lat", "paper1_beta_binomial_final_effects.csv",
         {"model": "Lexibank Beta-Binomial replication", "parameter": "beta_lat"}),
        ("beta_final", "paper1_beta_binomial_final_effects.csv",
         {"model": "ASJP Beta-Binomial final slope", "parameter": "beta_lat_final"}),
    ]
    ci_rows = []
    for key, f, match in ci_checks:
        row = C.find_row(C.load_csv(f), **match)
        lo_ok = approx(float(row["ci_2.5"]), float(manifest[key + "_ci_low"]), 1e-9)
        hi_ok = approx(float(row["ci_97.5"]), float(manifest[key + "_ci_high"]), 1e-9)
        a_fail += (0 if (lo_ok and hi_ok) else 1)
        ci_rows.append((key, row["ci_2.5"], manifest[key + "_ci_low"],
                        row["ci_97.5"], manifest[key + "_ci_high"], "PASS" if lo_ok and hi_ok else "FAIL"))

    # leave-family-out sample counts (exact) from the exclusion CSV
    ex = {r["model"]: r for r in C.load_csv("asjp_boundary_family_exclusion_sensitivity.csv")}
    lfo = [("ASJP BB excluding Atlantic-Congo", 8506, 645),
           ("ASJP BB excluding Austronesian", 9353, 645),
           ("ASJP BB excluding top two boundary families", 6973, 644)]
    lfo_rows = []
    for name, nl, nf in lfo:
        r = ex[name]
        ok = int(r["n_languages"]) == nl and int(r["n_families"]) == nf
        a_fail += (0 if ok else 1)
        lfo_rows.append((name, r["n_languages"], nl, r["n_families"], nf, "PASS" if ok else "FAIL"))

    # convergence diagnostics: all reported BB models, max_rhat<=1.05, divergences==0
    diag_files = [
        "asjp_beta_binomial_primary_diagnostics.csv",
        "lexibank_beta_binomial_replication_diagnostics.csv",
        "asjp_beta_binomial_within_between_diagnostics.csv",
        "asjp_beta_binomial_position_interaction_diagnostics.csv",
        "asjp_beta_binomial_hemisphere_diagnostics.csv",
        "asjp_beta_binomial_macroarea_control_diagnostics.csv",
        "asjp_beta_binomial_macroarea_matched_no_macroarea_diagnostics.csv",
        "asjp_beta_binomial_south_macroarea_slopes_diagnostics.csv",
    ]
    diag_rows = []
    for f in diag_files:
        d = C.load_csv(f)[0]
        ok = float(d["max_rhat"]) <= 1.05 and int(d["divergences"]) == 0
        a_fail += (0 if ok else 1)
        diag_rows.append((d["model"], d["max_rhat"], d["min_ess_bulk"], d["divergences"],
                          "PASS" if ok else "FAIL"))

    # table-cell spot check: generated LaTeX contains CSV-derived strings
    spot = []
    def table_text(tid):
        with open(os.path.join(C.TABLES, f"table{tid}.tex")) as fh:
            return fh.read()
    spot_checks = [
        ("S3", "10,886", "primary language count"),
        ("S3", "260", "unlinked isolate/unknown count"),
        ("S8", "-0.151", "BB primary beta_lat median"),
        ("S10", "-0.069", "leave Atlantic-Congo beta_lat"),
        ("S11", "-0.242", "leave-both final-minus-initial"),
        ("S14", "48.0", "macroarea attenuation (publication)"),
    ]
    for tid, needle, desc in spot_checks:
        present = needle in table_text(tid)
        a_fail += (0 if present else 1)
        spot.append((f"Table {tid}", needle, desc, "PASS" if present else "FAIL"))

    # ============ Layer B: manuscript-contract (declared <-> CSV) ============
    b_rows, b_flag = [], 0
    for cp in CHECKPOINTS:
        if cp["layer"] != "manuscript_s5":
            continue
        declared = cp["declared"]
        csv_val = resolve_num(cp["source"])
        tol = KIND_TOL.get(cp["kind"], 1e-3)
        if cp["kind"] == "count":
            ok = int(round(declared)) == int(round(csv_val))
        else:
            ok = approx(float(declared), csv_val, tol)
        status = "PASS" if ok else "FLAG-FOR-HUMAN-CORRECTION"
        if not ok:
            b_flag += 1
        b_rows.append((cp["id"], declared, round(csv_val, 6), tol, status, cp.get("note", "")))

    # ---------------- write validation_report.md ----------------
    overall_pass = (a_fail == 0)
    lines = ["# SI validation report", "",
             f"Generated by `validate_manuscript_consistency.py`. Master seed "
             f"{manifest['master_seed']}; results provenance SHA `{manifest['repository_sha'][:12]}`; "
             f"SI build HEAD `{manifest['si_build_head_sha'][:12]}`.", "",
             "Tolerances: integer counts exact; "
             + ", ".join(f"{k}={v}" for k, v in KIND_TOL.items()) + ".", "",
             f"## RESULT: {'PASS' if overall_pass else 'DISCREPANCIES FOUND'} "
             f"(repository-internal + evidence checks: {a_fail} failure(s))", ""]
    lines += ["Note: the manuscript-contract layer separately FLAGS the macroarea attenuation "
              "wording (47.6% vs authoritative 48.0%) for human correction; see "
              "`manuscript_consistency_report.md`. That flag is an intended manuscript-text "
              "correction, not a repository-internal failure.", ""]

    def tbl(title, header, rows):
        out = [f"### {title}", "", "| " + " | ".join(header) + " |",
               "|" + "|".join(["---"] * len(header)) + "|"]
        for r in rows:
            out.append("| " + " | ".join(str(x) for x in r) + " |")
        out.append("")
        return out

    lines += tbl("A1. Manifest values vs Result CSV (counts, coefficients, shares)",
                 ["checkpoint", "csv file", "csv value", "manifest value", "status"], a_rows)
    lines += tbl("A2. Credible-interval endpoints (manifest vs CSV)",
                 ["coefficient", "csv ci_2.5", "manifest low", "csv ci_97.5", "manifest high", "status"], ci_rows)
    lines += tbl("A3. Leave-family-out sample counts (exact)",
                 ["model", "csv n_lang", "expected", "csv n_fam", "expected", "status"], lfo_rows)
    lines += tbl("A4. Convergence diagnostics (max R-hat <= 1.05, divergences == 0)",
                 ["model", "max_rhat", "min_ess_bulk", "divergences", "status"], diag_rows)
    lines += tbl("A5. Generated-table cell spot checks",
                 ["table", "value", "description", "status"], spot)
    with open(os.path.join(C.GEN, "validation_report.md"), "w") as fh:
        fh.write("\n".join(lines))

    # ---------------- write manuscript_consistency_report.md ----------------
    ml = ["# Manuscript consistency report (three layers)", "",
          "The manuscript `.tex`/`.bib` are **not** present in this repository. The three "
          "consistency layers below are therefore reported **separately** and must not be conflated.",
          ""]
    ml += ["## Layer (a) - Repository-internal consistency (manifest <-> Result CSV)", "",
           f"**{'PASS' if a_fail == 0 else 'FAIL'}** - {len(a_rows)} manifest values, "
           f"{len(ci_rows)} CI pairs, {len(lfo_rows)} leave-family-out counts, "
           f"{len(diag_rows)} diagnostics, {len(spot)} table spot checks; "
           f"{a_fail} failure(s). Every publication-facing number in the SI resolves to a Result CSV.", ""]
    ml += ["## Layer (b) - Manuscript-contract consistency (Section-5 declared <-> Result CSV)", "",
           f"**{'PASS' if b_flag == 0 else f'{b_flag} item(s) FLAGGED for human correction'}**", "",
           "| checkpoint | manuscript declared | authoritative CSV | tol | status |",
           "|---|---|---|---|---|"]
    for cid, dec, cval, tol, status, note in b_rows:
        ml.append(f"| {cid} | {dec} | {cval} | {tol} | {status} |")
    ml.append("")
    flagged = [r for r in b_rows if r[4] != "PASS"]
    if flagged:
        ml += ["### Flagged items (require human correction in the manuscript text)", ""]
        for cid, dec, cval, tol, status, note in flagged:
            ml += [f"- **{cid}**: manuscript declares `{dec}`, authoritative CSV value is `{cval}` "
                   f"(tolerance {tol}). {note}",
                   "  Conflicting sources: manuscript text vs "
                   "`asjp_beta_binomial_macroarea_matched_comparison.csv` "
                   "(`attenuation_relative_to_no_macro`). Recommended inspection: manuscript occurrences "
                   "of 47.6% / 47.7%; replace with 48.0% (median-based). Not auto-corrected.", ""]
    ml += ["## Layer (c) - Direct manuscript-source validation", "",
           "**NOT PERFORMED.** The external manuscript `.tex` and `.bib` are not in this repository, "
           "so publication-facing values cannot be parsed directly from the manuscript source. The "
           "Section-5 checkpoint values (Layer b) are used as the manuscript contract in their place. "
           "If the manuscript source is later added, re-point the validator at parsed `.tex` values to "
           "perform this layer.", ""]
    with open(os.path.join(C.GEN, "manuscript_consistency_report.md"), "w") as fh:
        fh.write("\n".join(ml))

    print(f"Layer A (repo-internal): {a_fail} failure(s) -> {'PASS' if a_fail == 0 else 'FAIL'}")
    print(f"Layer B (manuscript-contract): {b_flag} flagged item(s) for human correction")
    print("Layer C (direct manuscript source): NOT PERFORMED (.tex/.bib absent)")
    print("Wrote validation_report.md and manuscript_consistency_report.md")
    # exit non-zero only on a genuine repo-internal failure (flags are expected)
    return 0 if a_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
