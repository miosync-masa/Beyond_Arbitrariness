#!/usr/bin/env python3
"""
checkpoints.py
==============
Declarative registry mapping every publication-facing quantity to its exact
Result-CSV source (file + row selector + column).  Used by:

  * build_manifest.py                 -> derive manifest values from CSVs
  * build_manifest.py                 -> emit manuscript_contract.json
  * validate_manuscript_consistency.py -> the three consistency layers

Each entry carries the manuscript-declared value from the Section-5 checkpoint
list where one exists (``declared``), the numeric ``kind`` (which sets the
comparison tolerance), and the CSV ``source`` locator.  ``layer`` marks whether
the checkpoint participates in the manuscript-contract comparison
(``manuscript_s5``) or only the repository-internal comparison (``repo_internal``).

NOTHING here is transcribed from the manuscript PDF or notebook display: the
``declared`` fields are the *manuscript's own rounded targets* (source D, a
consistency target only); the authoritative value is always recomputed from the
CSV ``source`` at validation time.
"""

# kind -> default absolute tolerance for float comparison against the rounded
# manuscript-declared value.  Integer counts use exact equality (tol 0).
KIND_TOL = {
    "count": 0.0,           # exact equality for integer counts
    "coefficient": 1.5e-3,  # 3-dp display rounding, allowing manuscript median/mean rounding
    "share": 1e-3,
    "percentage": 0.2,      # percentage points
    "fraction_pct": 2e-3,   # attenuation fraction; tight enough to FLAG 0.476 (47.6%) vs 0.4797 (48.0%)
}


def sel(file, field, match=None, row_index=None):
    d = {"file": file, "field": field}
    if match is not None:
        d["match"] = match
    if row_index is not None:
        d["row_index"] = row_index
    return d


# ---------------------------------------------------------------------------
# The registry.  manifest_key = key written into manifest.json.
# ---------------------------------------------------------------------------
CHECKPOINTS = [
    # ---- sample / family counts (exact) ----
    dict(id="asjp_prefilter_languages", manifest_key="asjp_prefilter_languages",
         label="ASJP pre-filter language summaries", kind="count", declared=11393,
         layer="manuscript_s5",
         source=sel("asjp_macroarea_sample_flow.csv", "n_languages", row_index=0)),
    dict(id="asjp_prefilter_family_ids", manifest_key="asjp_prefilter_family_ids",
         label="ASJP pre-filter family identifiers", kind="count", declared=667,
         layer="manuscript_s5",
         source=sel("asjp_macroarea_sample_flow.csv", "n_families", row_index=0)),
    dict(id="asjp_primary_languages", manifest_key="asjp_primary_languages",
         label="ASJP primary inferential languages", kind="count", declared=10886,
         layer="manuscript_s5",
         source=sel("asjp_macroarea_sample_flow.csv", "n_languages", row_index=1)),
    dict(id="asjp_primary_family_ids", manifest_key="asjp_primary_family_ids",
         label="ASJP primary family identifiers", kind="count", declared=646,
         layer="manuscript_s5",
         source=sel("asjp_macroarea_sample_flow.csv", "n_families", row_index=1)),
    dict(id="lexibank_languages", manifest_key="lexibank_languages",
         label="Lexibank languages", kind="count", declared=5501,
         layer="manuscript_s5",
         source=sel("lexibank_family_audit.csv", "n_languages", row_index=0)),
    dict(id="macroarea_matched_languages", manifest_key="macroarea_matched_languages",
         label="Macroarea matched languages", kind="count", declared=10626,
         layer="manuscript_s5",
         source=sel("asjp_macroarea_sample_flow.csv", "n_languages", row_index=2)),
    dict(id="macroarea_matched_family_ids", manifest_key="macroarea_matched_family_ids",
         label="Macroarea matched family identifiers", kind="count", declared=386,
         layer="manuscript_s5",
         source=sel("asjp_macroarea_sample_flow.csv", "n_families", row_index=2)),
    dict(id="macroarea_unlinked_languages", manifest_key="macroarea_unlinked_languages",
         label="Macroarea-unlinked languages", kind="count", declared=260,
         layer="manuscript_s5",
         source=sel("asjp_macroarea_sample_difference.csv", "languages_excluded", row_index=0)),

    # ---- primary / replication coefficients (posterior median) ----
    dict(id="primary_beta_lat", manifest_key="primary_beta_lat",
         label="ASJP primary beta_lat", kind="coefficient", declared=-0.151,
         layer="manuscript_s5",
         source=sel("paper1_beta_binomial_final_effects.csv", "median",
                    match={"model": "ASJP Beta-Binomial primary", "parameter": "beta_lat"})),
    dict(id="lexibank_beta_lat", manifest_key="lexibank_beta_lat",
         label="Lexibank beta_lat", kind="coefficient", declared=-0.138,
         layer="manuscript_s5",
         source=sel("paper1_beta_binomial_final_effects.csv", "median",
                    match={"model": "Lexibank Beta-Binomial replication", "parameter": "beta_lat"})),

    # ---- positional / decomposition coefficients ----
    dict(id="beta_initial", manifest_key="beta_initial",
         label="beta_initial", kind="coefficient", declared=0.028, layer="manuscript_s5",
         source=sel("paper1_beta_binomial_final_effects.csv", "median",
                    match={"model": "ASJP Beta-Binomial initial slope", "parameter": "beta_lat_initial"})),
    dict(id="beta_final", manifest_key="beta_final",
         label="beta_final", kind="coefficient", declared=-0.285, layer="manuscript_s5",
         source=sel("paper1_beta_binomial_final_effects.csv", "median",
                    match={"model": "ASJP Beta-Binomial final slope", "parameter": "beta_lat_final"})),
    dict(id="beta_final_minus_initial", manifest_key="beta_final_minus_initial",
         label="beta_final_minus_initial", kind="coefficient", declared=-0.313, layer="manuscript_s5",
         source=sel("paper1_beta_binomial_final_effects.csv", "median",
                    match={"model": "ASJP Beta-Binomial final-minus-initial", "parameter": "beta_lat_final_delta"})),
    dict(id="beta_between", manifest_key="beta_between",
         label="beta_between", kind="coefficient", declared=-0.348, layer="manuscript_s5",
         source=sel("paper1_beta_binomial_final_effects.csv", "median",
                    match={"model": "ASJP Beta-Binomial between-family", "parameter": "beta_between"})),
    dict(id="beta_within", manifest_key="beta_within",
         label="beta_within", kind="coefficient", declared=-0.033, layer="manuscript_s5",
         source=sel("paper1_beta_binomial_final_effects.csv", "median",
                    match={"model": "ASJP Beta-Binomial within-family", "parameter": "beta_within"})),

    # ---- matched-sample macroarea adjustment ----
    dict(id="matched_no_macroarea_beta", manifest_key="matched_no_macroarea_beta",
         label="matched no-macroarea beta", kind="coefficient", declared=-0.138, layer="manuscript_s5",
         source=sel("asjp_beta_binomial_macroarea_matched_comparison.csv", "median",
                    match={"model": "ASJP BB matched sample without macroarea", "parameter": "beta_lat"})),
    dict(id="matched_macroarea_beta", manifest_key="matched_macroarea_beta",
         label="matched macroarea beta", kind="coefficient", declared=-0.072, layer="manuscript_s5",
         source=sel("asjp_beta_binomial_macroarea_matched_comparison.csv", "median",
                    match={"model": "ASJP BB matched sample with macroarea", "parameter": "beta_lat"})),
    dict(id="macroarea_attenuation", manifest_key="macroarea_attenuation",
         label="macroarea attenuation (median-based, fraction)", kind="fraction_pct",
         declared=0.476, layer="manuscript_s5",
         note="Manuscript text states 47.6%; authoritative median-based CSV value is 0.4797 (48.0%). "
              "Publication-facing value per approved correction = 48.0%. FLAG 47.6/47.7% for human correction.",
         source=sel("asjp_beta_binomial_macroarea_matched_comparison.csv", "attenuation_relative_to_no_macro",
                    match={"model": "ASJP BB matched sample with macroarea", "parameter": "beta_lat"})),

    # ---- hemisphere ----
    dict(id="north_beta", manifest_key="north_beta",
         label="north beta", kind="coefficient", declared=-0.230, layer="manuscript_s5",
         source=sel("paper1_beta_binomial_final_effects.csv", "median",
                    match={"model": "ASJP BB North slope", "parameter": "beta_lat_north"})),
    dict(id="south_beta", manifest_key="south_beta",
         label="south beta", kind="coefficient", declared=0.527, layer="manuscript_s5",
         source=sel("paper1_beta_binomial_final_effects.csv", "median",
                    match={"model": "ASJP BB South slope", "parameter": "beta_lat_south"})),

    # ---- leave-family-out ----
    dict(id="leave_both_final_minus_initial", manifest_key="leave_both_final_minus_initial",
         label="leave-both final-minus-initial", kind="coefficient", declared=-0.242, layer="manuscript_s5",
         source=sel("asjp_position_excluding_top_two_boundary_families.csv", "median",
                    match={"model": "ASJP BB final-minus-initial excluding top two boundary families",
                           "parameter": "beta_lat_final_delta"})),
    dict(id="leave_atlantic_congo_beta", manifest_key="leave_atlantic_congo_beta",
         label="leave Atlantic-Congo beta_lat", kind="coefficient", declared=None, layer="repo_internal",
         source=sel("asjp_boundary_family_exclusion_sensitivity.csv", "median",
                    match={"model": "ASJP BB excluding Atlantic-Congo", "parameter": "beta_lat"})),
    dict(id="leave_austronesian_beta", manifest_key="leave_austronesian_beta",
         label="leave Austronesian beta_lat", kind="coefficient", declared=None, layer="repo_internal",
         source=sel("asjp_boundary_family_exclusion_sensitivity.csv", "median",
                    match={"model": "ASJP BB excluding Austronesian", "parameter": "beta_lat"})),
    dict(id="leave_both_beta", manifest_key="leave_both_beta",
         label="leave both beta_lat", kind="coefficient", declared=None, layer="repo_internal",
         source=sel("asjp_boundary_family_exclusion_sensitivity.csv", "median",
                    match={"model": "ASJP BB excluding top two boundary families", "parameter": "beta_lat"})),

    # ---- exact-one boundary mass (observed vs Beta-Binomial PPC) ----
    dict(id="exact_one_observed_asjp", manifest_key="exact_one_observed_asjp",
         label="ASJP observed exact-one share", kind="share", declared=None, layer="repo_internal",
         source=sel("asjp_beta_binomial_ppc_summary.csv", "observed",
                    match={"statistic": "share_rate_one"})),
    dict(id="exact_one_ppc_asjp", manifest_key="exact_one_ppc_asjp",
         label="ASJP Beta-Binomial PPC exact-one median", kind="share", declared=None, layer="repo_internal",
         source=sel("asjp_beta_binomial_ppc_summary.csv", "ppc_median",
                    match={"statistic": "share_rate_one"})),
    dict(id="exact_one_observed_lexibank", manifest_key="exact_one_observed_lexibank",
         label="Lexibank observed exact-one share", kind="share", declared=None, layer="repo_internal",
         source=sel("lexibank_beta_binomial_ppc_summary.csv", "observed",
                    match={"statistic": "share_rate_one"})),
    dict(id="exact_one_ppc_lexibank", manifest_key="exact_one_ppc_lexibank",
         label="Lexibank Beta-Binomial PPC exact-one median", kind="share", declared=None, layer="repo_internal",
         source=sel("lexibank_beta_binomial_ppc_summary.csv", "ppc_median",
                    match={"statistic": "share_rate_one"})),
]
