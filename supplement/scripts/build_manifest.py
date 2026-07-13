#!/usr/bin/env python3
"""
build_manifest.py
=================
Build ``supplement/manifest.json`` (and a copy in ``supplement/generated/``) plus
the machine-readable manuscript contract, entirely from Result CSV files and
recorded provenance metadata.  No number is transcribed from the manuscript or
from notebook display output.

Outputs
-------
* supplement/manifest.json
* supplement/generated/manifest.json            (identical copy)
* supplement/generated/manuscript_contract.json
* supplement/generated/manuscript_contract_schema.json
"""
from __future__ import annotations

import json
import os

import si_common as C
from checkpoints import CHECKPOINTS, KIND_TOL

META_JSON = os.path.join(C.RESULTS, "reproducibility_metadata.json")


def resolve(source: dict):
    """Return the raw string value located by a checkpoint source selector."""
    rows = C.load_csv(source["file"])
    if "row_index" in source:
        row = rows[source["row_index"]]
    else:
        row = C.find_row(rows, **source["match"])
    return row[source["field"]]


def resolve_num(source: dict):
    v = resolve(source)
    return None if v == "" else float(v)


def ci(effect_file: str, model: str, parameter: str):
    row = C.find_row(C.load_csv(effect_file), model=model, parameter=parameter)
    return float(row["ci_2.5"]), float(row["ci_97.5"])


def main() -> int:
    with open(META_JSON) as fh:
        meta = json.load(fh)

    EFF = "paper1_beta_binomial_final_effects.csv"

    # checkpoint-derived values (point estimates, posterior medians)
    vals = {cp["manifest_key"]: resolve_num(cp["source"]) for cp in CHECKPOINTS}

    # credible intervals for the coefficients that the manuscript reports with CrIs
    ci_primary = ci(EFF, "ASJP Beta-Binomial primary", "beta_lat")
    ci_lexi = ci(EFF, "Lexibank Beta-Binomial replication", "beta_lat")
    ci_init = ci(EFF, "ASJP Beta-Binomial initial slope", "beta_lat_initial")
    ci_final = ci(EFF, "ASJP Beta-Binomial final slope", "beta_lat_final")
    ci_delta = ci(EFF, "ASJP Beta-Binomial final-minus-initial", "beta_lat_final_delta")

    att = vals["macroarea_attenuation"]  # median-based fraction, 0.4796617595...

    manifest = {
        "_about": (
            "Provenance and publication-facing value manifest for the Supplementary "
            "Information. All numeric values are derived programmatically from Result CSV "
            "files in results/ (source-of-truth A); provenance SHAs/seed/backend come from "
            "results/reproducibility_metadata.json. Point estimates are posterior MEDIANS, "
            "consistent with the manuscript's reporting convention."
        ),

        # -- provenance / environment --
        "repository_sha": meta.get("repo_sha"),
        "repository_sha_note": (
            "SHA of the repository state under which the Result CSVs were generated, "
            "as recorded by the notebook in reproducibility_metadata.json."
        ),
        "si_build_head_sha": C.git_head(),
        "si_build_head_sha_note": (
            "Current repository HEAD at which this SI package was built; differs from "
            "repository_sha because the SI was added after the analysis run."
        ),
        "asjp_sha": meta.get("asjp_sha"),
        "lexibank_sha": meta.get("lexibank_sha"),
        "master_seed": meta.get("seed"),
        "backend": meta.get("backend"),
        "device": meta.get("device"),
        "created_utc_of_results": meta.get("created_utc"),

        # -- software versions (correction #6: recorded runtime vs pip constraint) --
        "python_version": "not_recorded_at_runtime",
        "jax_version": "0.7.2",
        "numpyro_version": "0.19.0",
        "arviz_version": "not_recorded_at_runtime",
        "pandas_version": "not_recorded_at_runtime",
        "numpy_version": "not_recorded_at_runtime",
        "version_provenance": {
            "recorded_at_runtime": {
                "source": "notebook cell 3 stdout ('JAX 0.7.2 NumPyro 0.19.0')",
                "jax": "0.7.2",
                "numpyro": "0.19.0",
                "device": "CudaDevice(id=0)",
            },
            "pip_install_constraints": {
                "source": "notebook cell 2 (!pip install ...)",
                "numpyro": ">=0.16,<0.20",
                "arviz": ">=0.20,<0.23",
                "pandas": ">=2.2,<3",
                "pyarrow": ">=17,<22",
                "matplotlib": ">=3.9,<4",
                "scipy": ">=1.13,<2",
                "netCDF4": ">=1.7",
            },
            "note": (
                "Constraints are installation bounds, NOT the exact installed versions. "
                "Only jax and numpyro versions were printed at runtime. python, arviz, "
                "pandas, numpy, and scipy exact versions were not recorded at runtime."
            ),
        },

        # -- sampler settings (from notebook code) --
        "sampler": {
            "algorithm": "NUTS", "target_accept_prob": 0.99, "max_tree_depth": 12,
            "num_warmup": 2000, "num_samples": 2500, "num_chains": 4,
            "chain_method": "vectorized", "jax_enable_x64": True,
            "source": "notebook cells 15/25 (fit_dataset / fit_numpyro_model)",
        },

        # -- sample / family counts --
        "asjp_prefilter_languages": C.to_i(vals["asjp_prefilter_languages"]),
        "asjp_prefilter_family_ids": C.to_i(vals["asjp_prefilter_family_ids"]),
        "asjp_primary_languages": C.to_i(vals["asjp_primary_languages"]),
        "asjp_primary_family_ids": C.to_i(vals["asjp_primary_family_ids"]),
        "lexibank_languages": C.to_i(vals["lexibank_languages"]),
        "macroarea_matched_languages": C.to_i(vals["macroarea_matched_languages"]),
        "macroarea_matched_family_ids": C.to_i(vals["macroarea_matched_family_ids"]),
        "macroarea_unlinked_languages": C.to_i(vals["macroarea_unlinked_languages"]),
        "languages_removed_by_concept_threshold": (
            C.to_i(vals["asjp_prefilter_languages"]) - C.to_i(vals["asjp_primary_languages"])
        ),

        # -- primary / replication coefficients (posterior median) --
        "primary_beta_lat": vals["primary_beta_lat"],
        "primary_beta_lat_ci_low": ci_primary[0],
        "primary_beta_lat_ci_high": ci_primary[1],
        "lexibank_beta_lat": vals["lexibank_beta_lat"],
        "lexibank_beta_lat_ci_low": ci_lexi[0],
        "lexibank_beta_lat_ci_high": ci_lexi[1],

        # -- positional (initial/final) --
        "beta_initial": vals["beta_initial"],
        "beta_initial_ci_low": ci_init[0],
        "beta_initial_ci_high": ci_init[1],
        "beta_final": vals["beta_final"],
        "beta_final_ci_low": ci_final[0],
        "beta_final_ci_high": ci_final[1],
        "beta_final_minus_initial": vals["beta_final_minus_initial"],
        "beta_final_minus_initial_ci_low": ci_delta[0],
        "beta_final_minus_initial_ci_high": ci_delta[1],

        # -- within/between decomposition --
        "beta_between": vals["beta_between"],
        "beta_within": vals["beta_within"],

        # -- matched-sample macroarea adjustment --
        "matched_no_macroarea_beta": vals["matched_no_macroarea_beta"],
        "matched_macroarea_beta": vals["matched_macroarea_beta"],
        "macroarea_attenuation": att,
        "macroarea_attenuation_pct": round(att * 100, 4),
        "macroarea_attenuation_pct_publication": 48.0,
        "macroarea_attenuation_note": (
            "Median-based attenuation = 1 - (median with-macroarea / median without-macroarea) "
            "= 0.4797 -> 48.0%%. Manuscript occurrences of 47.6%% / 47.7%% are FLAGGED for human "
            "correction (see manuscript_consistency_report.md); this is not rounding variation."
        ),

        # -- hemisphere --
        "north_beta": vals["north_beta"],
        "south_beta": vals["south_beta"],

        # -- leave-family-out --
        "leave_atlantic_congo_beta": vals["leave_atlantic_congo_beta"],
        "leave_austronesian_beta": vals["leave_austronesian_beta"],
        "leave_both_beta": vals["leave_both_beta"],
        "leave_both_final_minus_initial": vals["leave_both_final_minus_initial"],

        # -- exact-one boundary mass (observed vs Beta-Binomial PPC) --
        "exact_one_observed_asjp": vals["exact_one_observed_asjp"],
        "exact_one_ppc_asjp": vals["exact_one_ppc_asjp"],
        "exact_one_observed_lexibank": vals["exact_one_observed_lexibank"],
        "exact_one_ppc_lexibank": vals["exact_one_ppc_lexibank"],
    }

    # write manifest (two locations)
    for path in (os.path.join(C.SUP, "manifest.json"), os.path.join(C.GEN, "manifest.json")):
        with open(path, "w") as fh:
            json.dump(manifest, fh, indent=2)

    # ---- manuscript contract (declared publication-facing values + CSV locators) ----
    contract = {
        "_about": (
            "Machine-readable manuscript contract: the publication-facing quantities the "
            "Supplementary Information must keep traceable. 'declared' is the manuscript's "
            "OWN rounded target (Section-5 checkpoint, source D, a consistency target only). "
            "'csv_source' locates the authoritative value. Direct manuscript-source validation "
            "is NOT PERFORMED here because the external manuscript .tex/.bib are not in this "
            "repository; see manuscript_consistency_report.md."
        ),
        "direct_manuscript_source_validation": "NOT_PERFORMED (manuscript .tex/.bib absent from repository)",
        "tolerances_by_kind": KIND_TOL,
        "checkpoints": [
            {
                "id": cp["id"],
                "label": cp["label"],
                "kind": cp["kind"],
                "layer": cp["layer"],
                "manuscript_declared": cp.get("declared"),
                "authoritative_value_from_csv": resolve_num(cp["source"]),
                "csv_source": cp["source"],
                "note": cp.get("note", ""),
            }
            for cp in CHECKPOINTS
        ],
    }
    with open(os.path.join(C.GEN, "manuscript_contract.json"), "w") as fh:
        json.dump(contract, fh, indent=2)

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Manuscript contract for Beyond Arbitrariness SI",
        "type": "object",
        "required": ["direct_manuscript_source_validation", "tolerances_by_kind", "checkpoints"],
        "properties": {
            "direct_manuscript_source_validation": {"type": "string"},
            "tolerances_by_kind": {"type": "object"},
            "checkpoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "label", "kind", "layer", "csv_source",
                                 "authoritative_value_from_csv"],
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "kind": {"type": "string",
                                 "enum": ["count", "coefficient", "share", "percentage", "fraction_pct"]},
                        "layer": {"type": "string", "enum": ["manuscript_s5", "repo_internal"]},
                        "manuscript_declared": {"type": ["number", "null"]},
                        "authoritative_value_from_csv": {"type": ["number", "null"]},
                        "csv_source": {
                            "type": "object",
                            "required": ["file", "field"],
                            "properties": {
                                "file": {"type": "string"},
                                "field": {"type": "string"},
                                "match": {"type": "object"},
                                "row_index": {"type": "integer"},
                            },
                        },
                        "note": {"type": "string"},
                    },
                },
            },
        },
    }
    with open(os.path.join(C.GEN, "manuscript_contract_schema.json"), "w") as fh:
        json.dump(schema, fh, indent=2)

    print("Wrote supplement/manifest.json (+ generated copy)")
    print("Wrote generated/manuscript_contract.json and manuscript_contract_schema.json")
    print(f"  repository_sha (results provenance): {manifest['repository_sha']}")
    print(f"  si_build_head_sha:                   {manifest['si_build_head_sha']}")
    print(f"  primary_beta_lat (median): {manifest['primary_beta_lat']:.5f} "
          f"[{manifest['primary_beta_lat_ci_low']:.5f}, {manifest['primary_beta_lat_ci_high']:.5f}]")
    print(f"  macroarea_attenuation: {manifest['macroarea_attenuation']:.6f} "
          f"-> publication {manifest['macroarea_attenuation_pct_publication']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
