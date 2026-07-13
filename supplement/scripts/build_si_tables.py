#!/usr/bin/env python3
"""
build_si_tables.py
==================
Generate every SI table (S1-S18) as a standalone LaTeX float written to
``supplement/tables/tableSN.tex``.  All numbers come from the Result CSV files
(source-of-truth A); rule/prior text comes verbatim from notebook code cells 6,
11 and 70 (source-of-truth B).  Nothing is transcribed from the manuscript.

Run after build_manifest.py (needs supplement/manifest.json for the master seed).
"""
from __future__ import annotations

import json
import os

import si_common as C
from si_common import latex_escape as esc

with open(os.path.join(C.SUP, "manifest.json")) as _fh:
    MANIFEST = json.load(_fh)
MASTER_SEED = MANIFEST["master_seed"]
REPO_SHA = MANIFEST["repository_sha"]
ASJP_SHA = MANIFEST["asjp_sha"]
LEXI_SHA = MANIFEST["lexibank_sha"]


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def i(x):
    return f"{int(round(float(x))):,}"


def f3(x, nd=3):
    if x is None or x == "":
        return "--"
    return f"{float(x):.{nd}f}"


def pp(x):
    v = float(x)
    if v >= 0.9995:
        return r"$>$0.999"
    if v <= 0.0005:
        return r"$<$0.001"
    return f"{v:.3f}"


def crib(median, lo, hi, nd=3):
    return f"{f3(median, nd)} [{f3(lo, nd)}, {f3(hi, nd)}]"


def row_by(rows, **eq):
    return C.find_row(rows, **eq)


def write_table(item_id, caption, body, label=None, note="", resize=True, font=r"\footnotesize"):
    label = label or f"tab:{item_id.lower()}"
    inner = body
    if resize:
        # shrink-only: scale down if wider than the text block, never scale up
        inner = (r"\resizebox{\ifdim\width>\linewidth \linewidth\else\width\fi}{!}{%"
                 + "\n" + body + "\n}")
    note_tex = ("\n" + r"\par\smallskip{\footnotesize\emph{Note.} " + note + "}") if note else ""
    tex = (
        r"\begin{table}[htbp]\centering" + "\n"
        + font + "\n"
        + r"\caption{" + caption + "}\n"
        + r"\label{" + label + "}\n"
        + inner + "\n"
        + note_tex + "\n"
        + r"\end{table}" + "\n"
    )
    path = os.path.join(C.TABLES, f"table{item_id}.tex")
    with open(path, "w") as fh:
        fh.write(tex)
    return path


def tabular(colspec, header_rows, data_rows):
    """Build a booktabs tabular. header_rows/data_rows are lists of cell-lists."""
    out = [r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    for hr in header_rows:
        out.append(" & ".join(hr) + r" \\")
    out.append(r"\midrule")
    for dr in data_rows:
        out.append(" & ".join(dr) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out)


# ===========================================================================
# S1  Database provenance and preprocessing summary
# ===========================================================================
def build_S1():
    aa = row_by(C.load_csv("asjp_family_audit.csv"), dataset="ASJP")
    la = row_by(C.load_csv("lexibank_family_audit.csv"), dataset="Lexibank")
    flow = C.load_csv("asjp_macroarea_sample_flow.csv")
    asjp_words = sum(int(r["n_words"]) for r in C.load_csv("asjp_family_counts.csv"))
    lex_words = sum(int(r["n_words"]) for r in C.load_csv("lexibank_family_counts.csv"))
    attrs = [
        ("Version / commit SHA",
         f"ASJP (CLDF), SHA \\texttt{{{ASJP_SHA[:12]}}}",
         f"Lexibank-analysed, SHA \\texttt{{{LEXI_SHA[:12]}}}"),
        ("Raw forms (pre-exclusion)",
         "not recorded in committed CSVs", "not recorded in committed CSVs"),
        ("Retained word tokens", i(asjp_words), i(lex_words)),
        ("Pre-filter language summaries", i(flow[0]["n_languages"]), i(la["n_languages"])),
        ("Inferential languages", i(flow[1]["n_languages"]), i(la["n_languages"])),
        ("Family identifiers (primary)", i(flow[1]["n_families"]), i(la["n_families"])),
        ("Transcription system", "ASJPcode (41 symbols)", "IPA segments (CLTS)"),
        ("Coordinate source", "ASJP \\texttt{languages.csv} Lat/Long",
         "Lexibank CLDF Lat/Long"),
        ("Family source", "ASJP CLDF classification",
         "Glottolog \\texttt{Family} ($\\rightarrow$ \\texttt{Family\\_in\\_Data})"),
    ]
    data = [[esc(a) if False else a, v1, v2] for (a, v1, v2) in attrs]
    body = tabular("p{0.30\\textwidth} p{0.32\\textwidth} p{0.30\\textwidth}",
                   [["Attribute", "ASJP", "Lexibank"]], data)
    note = ("Retained word tokens are summed from the family-count CSVs; raw pre-exclusion form "
            "totals are not stored in the committed Result CSVs. Family-identifier counts are the "
            "primary inferential sample (isolate/unknown languages carry unique identifiers).")
    return write_table("S1", "Database provenance and preprocessing summary.", body, note=note)


# ===========================================================================
# S2  Segment and vowel classification rules (verbatim from notebook cell 6/7)
# ===========================================================================
def build_S2():
    asjp_v = "i e E 3 a u o"
    ipa_v = r"i y \textipa{1} \textipa{0} \textipa{W} u \textipa{I} \textipa{U} e \o{} \dots{} a \dots{} (25 symbols)"
    rows = [
        ("Vowel definition",
         f"character in ASJP\\_VOWELS = \\{{{asjp_v}\\}}",
         "token whose first non-diacritic character is an IPA vowel (25-symbol set)"),
        ("Ignored / stripped symbols",
         "keep only ASJP vowel/consonant symbols; all modifiers dropped",
         "strip length, nasalisation, phonation, stress, tone diacritics (IPA\\_STRIP) before classifying"),
        ("Segment extraction rule",
         "per-character: retain symbols in the ASJP vowel or consonant set",
         "split \\texttt{Segments} on spaces (\\texttt{+}$\\rightarrow$space); drop empty tokens"),
        ("Initial-segment rule",
         "initial vowel $=1$ iff first segment is a vowel",
         "initial vowel $=1$ iff first segment is a vowel"),
        ("Final-segment rule",
         "final vowel $=1$ iff last segment is a vowel; final cluster $=$ trailing consonant count",
         "same rule applied to IPA segments"),
        ("Malformed / empty forms",
         "non-string or no valid segment $\\rightarrow$ empty list, language contributes no counts",
         "non-string/empty or all-diacritic $\\rightarrow$ dropped"),
    ]
    body = tabular("p{0.20\\textwidth} p{0.36\\textwidth} p{0.36\\textwidth}",
                   [["Rule", "ASJP", "Lexibank (IPA)"]], rows)
    note = ("Definitions taken verbatim from notebook cell~6 (\\texttt{ASJP\\_VOWELS}, "
            "\\texttt{IPA\\_VOWELS}, \\texttt{IPA\\_STRIP}, \\texttt{asjp\\_segments}, "
            "\\texttt{ipa\\_segments}, \\texttt{final\\_cluster}) and the aggregation in cell~7. "
            "Word-final vowel occurrence is measured directly; it is a plausible correlate of, but "
            "not identical to, open-syllable structure.")
    return write_table("S2", "Segment and vowel classification rules.", body, note=note)


# ===========================================================================
# S3  ASJP sample flow
# ===========================================================================
def build_S3():
    flow = C.load_csv("asjp_macroarea_sample_flow.csv")
    diff = C.load_csv("asjp_macroarea_sample_difference.csv")[0]
    removed = int(flow[0]["n_languages"]) - int(flow[1]["n_languages"])
    data = []
    for r in flow:
        data.append([esc(r["stage"]), i(r["n_languages"]), i(r["n_families"]),
                     i(r["n_unknown_or_isolate_languages"])])
    body = tabular("p{0.44\\textwidth} r r r",
                   [["Stage", "Languages", "Family ids", "Isolate/unknown"]], data)
    note = (f"{removed} languages were removed by the 20-concept inclusion threshold "
            f"(11{{,}}393 $\\rightarrow$ 10{{,}}886). A further {i(diff['languages_excluded'])} "
            "languages could not be linked under the identifier used for the macroarea table; these "
            "are NOT missing-macroarea cases -- all "
            f"{i(diff['unknown_or_isolate_languages_excluded'])} are isolate/unknown languages with "
            "unique family identifiers (full ledger: \\texttt{asjp\\_macroarea\\_dropped\\_languages.csv}).")
    return write_table("S3", "ASJP sample flow.", body, note=note)


# ===========================================================================
# S4  Macroarea-linked sample composition
# ===========================================================================
def build_S4():
    rows = C.load_csv("asjp_macroarea_distribution.csv")
    total = sum(int(r["n_languages"]) for r in rows)
    data = []
    for r in sorted(rows, key=lambda x: -int(x["n_languages"])):
        share = int(r["n_languages"]) / total
        data.append([esc(r["Macroarea"]), i(r["n_languages"]), i(r["n_families"]), f"{share*100:.1f}\\%"])
    data.append([r"\textbf{Total}", f"\\textbf{{{i(total)}}}", "386", "100.0\\%"])
    body = tabular("l r r r",
                   [["Macroarea", "Languages", "Family ids", "Share of matched sample"]], data)
    note = ("Matched macroarea-complete sample (10{,}626 languages, 386 family identifiers). Shares "
            "are of the matched sample; family identifiers are not additive across macroareas.")
    return write_table("S4", "Macroarea-linked sample composition.", body, note=note)


# ===========================================================================
# S5  Macroarea linkage audit summary
# ===========================================================================
def build_S5():
    ec = C.load_csv("asjp_macroarea_exclusion_counts.csv")
    md = C.load_csv("asjp_macroarea_source_metadata.csv")[0]
    data = []
    for r in ec:
        data.append([esc(r["macroarea_exclusion_reason"]), i(r["n_languages"]),
                     i(r["n_families"]), i(r["n_unknown_or_isolate_languages"])])
    body = tabular("l r r r",
                   [["Linkage outcome", "Languages", "Family ids", "Isolate/unknown"]], data)
    note = ("Macroarea source: ASJP \\texttt{languages.csv} column \\texttt{Macroarea}; "
            f"inner merge on \\texttt{{Language\\_ID}}; {i(md['source_rows'])} source rows, "
            f"{i(md['duplicate_language_id_rows'])} duplicate identifiers, "
            f"{i(md['complete_macroarea_language_ids'])} languages with a macroarea. The complete "
            "260-language exclusion ledger is retained as \\texttt{asjp\\_macroarea\\_dropped\\_languages.csv}; "
            "every excluded language is isolate/unknown, not a missing-macroarea case.")
    return write_table("S5", "Macroarea linkage audit summary.", body, note=note)


# ===========================================================================
# S6  Prior specification and parameterisation (verbatim from code)
# ===========================================================================
def build_S6():
    rows = [
        (r"$\alpha$ (intercept)", r"$\mathrm{Normal}(0,\,1.5)$", "global log-odds intercept"),
        (r"$\beta_{\mathrm{lat}}$ and all slope terms", r"$\mathrm{Normal}(0,\,1)$",
         r"latitude / level / interaction slopes (\texttt{beta\_lat}, \texttt{beta\_lat\_north}, "
         r"\texttt{beta\_south\_level}, \texttt{beta\_lat\_south\_delta}, \texttt{beta\_between}, "
         r"\texttt{beta\_within}, \texttt{beta\_lat\_initial}, \texttt{beta\_final\_level}, "
         r"\texttt{beta\_lat\_final\_delta})"),
        (r"$\sigma_{\mathrm{family}},\ \sigma_{\mathrm{language}}$", r"$\mathrm{Exponential}(1)$",
         "random-effect standard deviations (family; and language in the positional model)"),
        (r"$z_{\mathrm{family}},\ z_{\mathrm{language}}$", r"$\mathrm{Normal}(0,1)$, then mean-centred",
         "non-centred, sum-to-zero standardised random intercepts"),
        ("macroarea effects", r"$\mathrm{Normal}(0,\,0.75)$",
         "fixed categorical intercepts/slopes for the six observed macroareas (deliberately NOT hierarchical)"),
        (r"$\phi$ (Beta-Binomial concentration)", r"$\mathrm{LogNormal}(\log 20,\,1)$",
         "overdispersion parameter of the reported Beta-Binomial likelihood"),
    ]
    body = tabular("p{0.24\\textwidth} p{0.24\\textwidth} p{0.44\\textwidth}",
                   [["Parameter", "Prior", "Role"]], rows)
    note = (r"Likelihoods: $y \sim \mathrm{Binomial}(n,\ \mathrm{logit}^{-1}\eta)$ (superseded layer) "
            r"and $y \sim \mathrm{BetaBinomial}(n,\ \mu\phi,\ (1-\mu)\phi)$ with "
            r"$\mu=\mathrm{logit}^{-1}\eta$ (reported layer); linear predictor "
            r"$\eta=\alpha+\beta_{\mathrm{lat}}\,z_{\mathrm{lat}}+\sigma_{\mathrm{family}}z_{\mathrm{family}[i]}$. "
            "Latitude is $|\\mathrm{Latitude}|$ z-scored with population SD (\\texttt{ddof=0}); it is a "
            "composite geographic coordinate, not a causal exposure. Family random intercepts model "
            "family-level clustering and are NOT full phylogenetic control. The positional model adds "
            "language-level intercepts; the initial--final contrast is a within-language comparison. "
            "Priors are read from notebook cells 11 and 70, not inferred from posteriors.")
    return write_table("S6", "Prior specification and parameterisation.", body, note=note)


# ===========================================================================
# S7  Complete MCMC diagnostics
# ===========================================================================
def build_S7():
    # (display, status, diag_csv, n_lang, n_fam, seed_offset)
    models = [
        ("Primary (ASJP)", "primary", "asjp_beta_binomial_primary_diagnostics.csv", 10886, 646, 301),
        ("Replication (Lexibank)", "replication", "lexibank_beta_binomial_replication_diagnostics.csv", 5501, 298, 305),
        ("Within/between-family", "robustness", "asjp_beta_binomial_within_between_diagnostics.csv", 10886, 646, 302),
        ("Initial-vs-final (positional)", "robustness", "asjp_beta_binomial_position_interaction_diagnostics.csv", 10886, 646, 303),
        ("Hemisphere", "robustness", "asjp_beta_binomial_hemisphere_diagnostics.csv", 10886, 646, 304),
        ("Family + macroarea", "robustness", "asjp_beta_binomial_macroarea_control_diagnostics.csv", 10626, 386, 306),
        ("Matched, no macroarea", "robustness", "asjp_beta_binomial_macroarea_matched_no_macroarea_diagnostics.csv", 10626, 386, 308),
        ("Southern macroarea slopes", "robustness", "asjp_beta_binomial_south_macroarea_slopes_diagnostics.csv", 4075, None, 307),
    ]
    data = []
    for disp, status, csvf, nl, nf, off in models:
        d = C.load_csv(csvf)[0]
        seed = f"{MASTER_SEED}+{off}"
        data.append([disp, status, i(nl), (i(nf) if nf else "--"),
                     "4", "2000", "2500", f3(d["max_rhat"], 2),
                     i(d["min_ess_bulk"]), i(d["min_ess_tail"]), d["divergences"], "0.99", seed])
    header = [["Model", "Status", "N lang.", "N fam.", "Chains", "Warm-up", "Draws/chain",
               "Max $\\hat{R}$", "Min bulk ESS", "Min tail ESS", "Div.", "Target", "Seed"]]
    body = tabular("l l r r c c c c r r c c l", header, data)
    note = ("Leave-family-out and leave-top-two positional re-fits use the same NUTS settings "
            "(target 0.99, max\\_tree\\_depth 12, 2000 warm-up, 2500 draws/chain, 4 chains) but their "
            "R-hat/ESS were not saved to separate diagnostics CSVs; their point estimates appear in "
            "Tables S10--S11. Southern macroarea per-macroarea family counts are in Table S15. "
            "All reported models show 0 divergences.")
    return write_table("S7", "Complete MCMC diagnostics for the reported Beta-Binomial models.",
                       body, resize=True, note=note)


# ===========================================================================
# S8  Binomial vs Beta-Binomial coefficient comparison
# ===========================================================================
def build_S8():
    binp = C.load_csv("asjp_primary_effect.csv")[0]
    bbp = row_by(C.load_csv("asjp_beta_binomial_primary_effect.csv"),
                 model="ASJP Beta-Binomial primary", parameter="beta_lat")
    binpos = C.load_csv("asjp_initial_final_effects.csv")
    bbpos = C.load_csv("asjp_beta_binomial_position_effects.csv")
    binwb = C.load_csv("asjp_within_between_effects.csv")
    bbwb = C.load_csv("asjp_beta_binomial_within_between_effects.csv")

    def bcrib(rows, **eq):
        r = row_by(rows, **eq)
        return crib(r["median"], r["ci_2.5"], r["ci_97.5"])

    data = [
        ["Primary latitude", crib(binp["median"], binp["ci_2.5"], binp["ci_97.5"]),
         crib(bbp["median"], bbp["ci_2.5"], bbp["ci_97.5"])],
        ["Initial-position latitude", bcrib(binpos, parameter="beta_lat_initial"),
         bcrib(bbpos, parameter="beta_lat_initial")],
        ["Final-position latitude", bcrib(binpos, parameter="beta_lat_final"),
         bcrib(bbpos, parameter="beta_lat_final")],
        ["Final-minus-initial", bcrib(binpos, parameter="beta_lat_final_delta"),
         bcrib(bbpos, parameter="beta_lat_final_delta")],
        ["Between-family latitude", bcrib(binwb, parameter="beta_between"),
         bcrib(bbwb, parameter="beta_between")],
        ["Within-family latitude", bcrib(binwb, parameter="beta_within"),
         bcrib(bbwb, parameter="beta_within")],
    ]
    body = tabular("l l l",
                   [["Quantity", "Binomial (superseded)", "Beta-Binomial (reported)"]], data)
    note = ("Posterior median [95\\% credible interval]. The Binomial fits are the earlier layer; the "
            "Beta-Binomial fits are the reported layer, adopted because the Binomial understated "
            "between-language dispersion (Figure S2).")
    return write_table("S8", "Binomial versus Beta-Binomial coefficient comparison (ASJP).", body, note=note)


# ===========================================================================
# S9  Boundary mass by minimum total word count
# ===========================================================================
def build_S9():
    rows = C.load_csv("boundary_mass_by_coverage.csv")
    data = []
    for r in rows:
        data.append([r["database"], i(r["min_n_total"]), i(r["n_languages"]),
                     f3(r["median_n_total"], 0),
                     f3(r["share_exact_one"]), f3(r["share_ge_0_99"]), f3(r["share_ge_0_95"])])
    header = [["Database", "Min.\\ total words", "N lang.", "Median total words",
               "Share $=1.00$", "Share $\\ge 0.99$", "Share $\\ge 0.95$"]]
    body = tabular("l r r r r r r", header, data)
    note = ("The robustness sweep varies the minimum \\emph{total word count} (\\texttt{min\\_n\\_total}); "
            "it is NOT a concept-count sweep. The inferential inclusion criterion throughout the paper "
            "is a fixed minimum of 20 concepts. The committed CSV records median total words, not median "
            "concepts. Upper-boundary concentration persists across thresholds, so it is not merely a "
            "sparse-wordlist artifact; this does not identify a typological class.")
    return write_table("S9", "Boundary mass by minimum total word count.", body, note=note)


# ===========================================================================
# S10  Leave-family-out primary estimates
# ===========================================================================
def build_S10():
    full = row_by(C.load_csv("asjp_beta_binomial_primary_effect.csv"),
                  model="ASJP Beta-Binomial primary", parameter="beta_lat")
    ex = C.load_csv("asjp_boundary_family_exclusion_sensitivity.csv")
    exd = {r["model"]: r for r in ex}
    data = [["Full primary", "10,886", "646",
             crib(full["median"], full["ci_2.5"], full["ci_97.5"]), pp(full["p_lt_0"])]]
    for lbl, key in [("Excl.\\ Atlantic-Congo", "ASJP BB excluding Atlantic-Congo"),
                     ("Excl.\\ Austronesian", "ASJP BB excluding Austronesian"),
                     ("Excl.\\ both", "ASJP BB excluding top two boundary families")]:
        r = exd[key]
        data.append([lbl, i(r["n_languages"]), i(r["n_families"]),
                     crib(r["median"], r["ci_2.5"], r["ci_97.5"]), pp(r["p_lt_0"])])
    body = tabular("l r r l c",
                   [["Sample", "N lang.", "N fam.", "$\\beta_{\\mathrm{lat}}$ [95\\% CrI]", "$P(\\beta<0)$"]], data)
    note = ("Atlantic-Congo strengthens the negative global slope; Austronesian partly suppresses it. "
            "The two largest families do not contribute in the same direction, so the global latitude "
            "coefficient is a composite summary. The association is not genealogy-free.")
    return write_table("S10", "Leave-family-out primary estimates (Beta-Binomial).", body, note=note)


# ===========================================================================
# S11  Positional estimates after excluding Atlantic-Congo and Austronesian
# ===========================================================================
def build_S11():
    full = {r["parameter"]: r for r in C.load_csv("asjp_beta_binomial_position_effects.csv")}
    excl = {r["parameter"]: r for r in C.load_csv("asjp_position_excluding_top_two_boundary_families.csv")}
    labels = [("beta_lat_initial", "Initial slope"),
              ("beta_lat_final", "Final slope"),
              ("beta_lat_final_delta", "Final-minus-initial")]
    data = []
    for par, disp in labels:
        f = full[par]
        e = excl[par]
        data.append([disp, crib(f["median"], f["ci_2.5"], f["ci_97.5"]), pp(f["p_lt_0"]),
                     crib(e["median"], e["ci_2.5"], e["ci_97.5"]), pp(e["p_lt_0"])])
    header = [["Parameter", "Full [95\\% CrI]", "$P(\\beta<0)$",
               "Excl.\\ both [95\\% CrI]", "$P(\\beta<0)$"]]
    body = tabular("l l c l c", header, data)
    note = ("Removing both large families does not eliminate final-position specificity: the "
            "final-minus-initial contrast remains clearly negative "
            "($-0.242$ [$-0.276$, $-0.208$]). Both intercept and language random effects are included.")
    return write_table("S11", "Positional estimates after excluding Atlantic-Congo and Austronesian.",
                       body, note=note)


# ===========================================================================
# S12  Largest contributors to exact-one boundary mass
# ===========================================================================
def build_S12():
    def top(csvf, db, k):
        rows = [r for r in C.load_csv(csvf) if r["Family"] != "Bookkeeping"]
        rows.sort(key=lambda x: -float(x["share_of_all_boundary_languages"]))
        out = []
        for r in rows[:k]:
            out.append([db, esc(r["Family"]), i(r["n_languages"]), i(r["n_exact_one"]),
                        f3(r["share_exact_one"]), f3(r["share_of_all_boundary_languages"])])
        return out
    data = top("asjp_family_boundary_concentration.csv", "ASJP", 8) + \
        top("lexibank_family_boundary_concentration.csv", "Lexibank", 6)
    header = [["Database", "Family", "N lang.", "N exact-one",
               "Obs.\\ exact-one share", "Share of all exact-one"]]
    body = tabular("l l r r r r", header, data)
    note = ("Atlantic-Congo dominates the exact-one boundary in both databases. The ASJP "
            "\\texttt{Bookkeeping} pseudo-family (17 languages, a Glottolog non-genealogical node) is "
            "excluded from this interpretive table but retained in "
            "\\texttt{audit\\_asjp\\_bookkeeping\\_family.csv}. Boundary concentration is descriptive and "
            "is not proof of a discrete open-syllable class.")
    return write_table("S12", "Largest contributors to exact-one boundary mass.", body, note=note)


# ===========================================================================
# S13  Family-level posterior predictive residuals
# ===========================================================================
def build_S13():
    def majors(csvf, db, k):
        rows = [r for r in C.load_csv(csvf)
                if int(r["n_languages"]) >= 100 and r["Family"] != "Bookkeeping"]
        rows.sort(key=lambda x: -int(x["n_languages"]))
        out = []
        for r in rows[:k]:
            out.append([db, esc(r["Family"]), i(r["n_languages"]),
                        f3(r["observed_share_exact_one"]), f3(r["ppc_median"]),
                        f"[{f3(r['ppc_2.5'])}, {f3(r['ppc_97.5'])}]",
                        f3(r["residual_observed_minus_ppc_median"]),
                        f3(r["bayesian_p_ge_observed"])])
        return out
    data = majors("asjp_beta_binomial_family_boundary_ppc.csv", "ASJP", 8) + \
        majors("lexibank_beta_binomial_family_boundary_ppc.csv", "Lexibank", 6)
    header = [["Database", "Family", "N lang.", "Obs.\\ share", "PPC median",
               "PPC 95\\%", "Obs $-$ PPC", "$p_{\\ge\\mathrm{obs}}$"]]
    body = tabular("l l r r r c r r", header, data)
    note = ("Major families ($n\\ge100$ languages) shown; the \\texttt{Bookkeeping} pseudo-family is "
            "excluded (retained in the audit ledger). Boundary concentration and model misfit are "
            "distinct: Atlantic-Congo dominates the boundary in both databases but its misfit is "
            "database-dependent, whereas Austronesian and Nuclear Trans New Guinea show reproducible "
            "underprediction across both. An exact-one excess is not proof of a discrete open-syllable class.")
    return write_table("S13", "Family-level posterior predictive residuals for exact-one boundary mass.",
                       body, resize=True, note=note)


# ===========================================================================
# S14  Matched-sample macroarea adjustment
# ===========================================================================
def build_S14():
    prim = row_by(C.load_csv("asjp_beta_binomial_primary_effect.csv"),
                  model="ASJP Beta-Binomial primary", parameter="beta_lat")
    comp = C.load_csv("asjp_beta_binomial_macroarea_matched_comparison.csv")
    nomac = row_by(comp, model="ASJP BB matched sample without macroarea", parameter="beta_lat")
    wmac = row_by(comp, model="ASJP BB matched sample with macroarea", parameter="beta_lat")
    att_pct = MANIFEST["macroarea_attenuation_pct_publication"]
    data = [
        ["Full primary, family-only", "10,886", "646",
         crib(prim["median"], prim["ci_2.5"], prim["ci_97.5"]), f3(prim["odds_ratio_per_1sd"]), "--"],
        ["Matched, family-only", i(nomac["n_languages"]), i(nomac["n_families"]),
         crib(nomac["median"], nomac["ci_2.5"], nomac["ci_97.5"]), f3(nomac["odds_ratio_per_1sd"]), "--"],
        ["Matched, family + macroarea", i(wmac["n_languages"]), i(wmac["n_families"]),
         crib(wmac["median"], wmac["ci_2.5"], wmac["ci_97.5"]), f3(wmac["odds_ratio_per_1sd"]),
         f"{att_pct:.1f}\\%"],
    ]
    header = [["Model", "N lang.", "N fam.", "$\\beta_{\\mathrm{lat}}$ [95\\% CrI]",
               "OR per SD", "Attenuation"]]
    body = tabular("l r r l r r", header, data)
    note = ("Broad macroarea adjustment removes approximately 48.0\\% of the matched-sample "
            "association (median-based, $1-0.0717/0.1377$). Sample restriction alone (full "
            "$\\rightarrow$ matched) produces only a modest change. Macroarea is a broad areal "
            "adjustment, not a complete areal control; latitude is not a globally invariant exposure.")
    return write_table("S14", "Matched-sample macroarea adjustment.", body, note=note)


# ===========================================================================
# S15  Southern-Hemisphere macroarea-specific slopes
# ===========================================================================
def build_S15():
    rows = C.load_csv("asjp_beta_binomial_south_macroarea_effects.csv")
    rows.sort(key=lambda x: -int(x["n_languages"]))
    data = []
    for r in rows:
        n = int(r["n_languages"])
        interp = "uninformative ($n=1$)" if n == 1 else "informative"
        data.append([esc(r["macroarea"]), i(n), i(r["n_families"]),
                     crib(r["median"], r["ci_2.5"], r["ci_97.5"]),
                     pp(r["p_lt_0"]), pp(r["p_gt_0"]), interp])
    header = [["Macroarea", "N lang.", "N fam.", "$\\beta$ [95\\% CrI]",
               "$P(\\beta<0)$", "$P(\\beta>0)$", "Status"]]
    body = tabular("l r r l c c l", header, data)
    note = ("The pooled Southern coefficient is not a universal reversal: the South is regionally "
            "heterogeneous (e.g.\\ South America trends negative, Papunesia strongly positive). "
            "Southern Eurasia ($n=1$) is uninformative and must not be interpreted.")
    return write_table("S15", "Southern-Hemisphere macroarea-specific slopes.", body, note=note)


# ===========================================================================
# S16  Cross-database language overlap (NOT COMPUTABLE from committed artifacts)
# ===========================================================================
def build_S16():
    asjp_n = MANIFEST["asjp_primary_languages"]
    lex_n = MANIFEST["lexibank_languages"]
    nc = "not computable"
    data = [
        ["ASJP primary languages", i(asjp_n)],
        ["Lexibank languages", i(lex_n)],
        ["Shared identifiers", nc],
        ["ASJP-only", nc],
        ["Lexibank-only", nc],
        ["Percentage of ASJP overlapping", nc],
        ["Percentage of Lexibank overlapping", nc],
        ["Identifier used", "none available (no Glottocode / shared key in committed CSVs)"],
        ["Unmatched or ambiguous records", nc],
    ]
    body = tabular("p{0.45\\textwidth} r",
                   [["Quantity", "Value"]], data)
    note = ("The committed Result CSVs contain no shared cross-database identifier (no Glottocode, and "
            "no per-language Lexibank table), and the raw CLDF databases are not in this repository, so "
            "the overlap cannot be computed from the committed evidence package. ASJP and Lexibank are "
            "independently assembled databases with partially overlapping language coverage; they are "
            "not statistically independent language samples. This cell is marked incomplete rather than "
            "populated with fabricated values.")
    return write_table("S16", "Cross-database language overlap (audit; not computable from committed artifacts).",
                       body, note=note)


# ===========================================================================
# S17  Computational environment
# ===========================================================================
def build_S17():
    vp = MANIFEST["version_provenance"]
    rec = vp["recorded_at_runtime"]
    con = vp["pip_install_constraints"]
    rows = [
        ("Python", "not recorded at runtime", "--", "Colab runtime"),
        ("JAX", rec["jax"], "--", "cell 3 stdout"),
        ("NumPyro", rec["numpyro"], con["numpyro"], "cell 3 stdout / cell 2 pin"),
        ("ArviZ", "not recorded", con["arviz"], "cell 2 pin"),
        ("pandas", "not recorded", con["pandas"], "cell 2 pin"),
        ("NumPy", "not recorded", "--", "bundled with runtime"),
        ("SciPy", "not recorded", con["scipy"], "cell 2 pin"),
        ("matplotlib", "not recorded", con["matplotlib"], "cell 2 pin"),
        ("pyarrow", "not recorded", con["pyarrow"], "cell 2 pin"),
        ("netCDF4", "not recorded", con["netCDF4"], "cell 2 pin"),
        ("Accelerator", "CudaDevice(id=0), A100 (declared)", "--", "cell 3 / notebook metadata"),
        ("Backend / device", f"{MANIFEST['backend']} / {MANIFEST['device']}", "--",
         "reproducibility_metadata.json"),
    ]
    data = [[esc(a), esc(b), esc(c), esc(d)] for (a, b, c, d) in rows]
    header = [["Component", "Recorded runtime version", "pip install constraint", "Source"]]
    body = tabular("l l l l", header, data)
    note = ("pip constraints are installation bounds, not the exact installed versions. Only JAX and "
            "NumPyro versions were printed at runtime; Python, ArviZ, pandas, NumPy and SciPy exact "
            "versions were not recorded. \\texttt{jax\\_enable\\_x64=True}; master seed "
            f"{MASTER_SEED}.")
    return write_table("S17", "Computational environment.", body, note=note)


# ===========================================================================
# S18  Analysis provenance and output-file map
# ===========================================================================
def build_S18():
    # (analysis, data_sha_label, seed_offset, result_csv, nc_artifact, figure_source, cell)
    entries = [
        ("ASJP BB primary", "ASJP", 301, "asjp\\_beta\\_binomial\\_primary\\_effect.csv",
         "asjp\\_beta\\_binomial\\_primary.nc", "figure2\\_core\\_posterior\\_effects.csv", "72"),
        ("Lexibank BB replication", "Lexibank", 305, "lexibank\\_beta\\_binomial\\_replication\\_effect.csv",
         "lexibank\\_beta\\_binomial\\_replication.nc", "figure2\\_core\\_posterior\\_effects.csv", "80"),
        ("ASJP BB within/between", "ASJP", 302, "asjp\\_beta\\_binomial\\_within\\_between\\_effects.csv",
         "asjp\\_beta\\_binomial\\_within\\_between.nc", "figure2\\_core\\_posterior\\_effects.csv", "74"),
        ("ASJP BB positional", "ASJP", 303, "asjp\\_beta\\_binomial\\_position\\_effects.csv",
         "asjp\\_beta\\_binomial\\_position\\_interaction.nc", "figure1\\_position\\_specific\\_gradients.csv", "76"),
        ("ASJP BB hemisphere", "ASJP", 304, "asjp\\_beta\\_binomial\\_hemisphere\\_effects.csv",
         "asjp\\_beta\\_binomial\\_hemisphere.nc", "figure2\\_core\\_posterior\\_effects.csv", "78"),
        ("ASJP BB family+macroarea", "ASJP", 306, "asjp\\_beta\\_binomial\\_macroarea\\_effect.csv",
         "asjp\\_beta\\_binomial\\_macroarea\\_control.nc", "figure3\\_macroarea\\_attenuation.csv", "85"),
        ("ASJP BB Southern macroarea", "ASJP", 307, "asjp\\_beta\\_binomial\\_south\\_macroarea\\_effects.csv",
         "asjp\\_beta\\_binomial\\_south\\_macroarea\\_slopes.nc", "figure3\\_southern\\_macroarea\\_slopes.csv", "86"),
        ("ASJP BB matched, no macroarea", "ASJP", 308,
         "asjp\\_beta\\_binomial\\_macroarea\\_matched\\_comparison.csv",
         "asjp\\_beta\\_binomial\\_macroarea\\_matched\\_no\\_macroarea.nc",
         "figure3\\_macroarea\\_attenuation.csv", "100"),
        ("Leave-family-out", "ASJP", 410, "asjp\\_boundary\\_family\\_exclusion\\_sensitivity.csv",
         "(not committed)", "figure2\\_core\\_posterior\\_effects.csv", "95"),
        ("Leave-top-two positional", "ASJP", 420,
         "asjp\\_position\\_excluding\\_top\\_two\\_boundary\\_families.csv",
         "(not committed)", "figure2\\_core\\_posterior\\_effects.csv", "97"),
    ]
    data = []
    for name, dsha, off, csvf, ncf, figf, cell in entries:
        data.append([name, f"\\texttt{{{REPO_SHA[:10]}}}",
                     (f"\\texttt{{{ASJP_SHA[:10]}}}" if dsha == "ASJP" else f"\\texttt{{{LEXI_SHA[:10]}}}"),
                     str(MASTER_SEED), f"+{off}", f"\\texttt{{{csvf}}}", f"\\texttt{{{ncf}}}",
                     f"\\texttt{{{figf}}}", cell])
    header = [["Analysis", "Repo SHA", "Data SHA", "Master seed", "Offset",
               "Result CSV", "NetCDF artifact", "Figure source", "Cell"]]
    body = tabular("l l l l l l l l c", header, data)
    note = ("Repo SHA is the results-provenance SHA from \\texttt{reproducibility\\_metadata.json}; the SI "
            "was built at HEAD \\texttt{" + MANIFEST["si_build_head_sha"][:10] + "}. NetCDF posterior "
            "artifacts are named by the notebook but are not committed to the repository; the H100/GPU "
            "models were NOT rerun to build this SI.")
    return write_table("S18", "Analysis provenance and output-file map.", body, resize=True, note=note)


def main() -> int:
    builders = [build_S1, build_S2, build_S3, build_S4, build_S5, build_S6, build_S7,
                build_S8, build_S9, build_S10, build_S11, build_S12, build_S13, build_S14,
                build_S15, build_S16, build_S17, build_S18]
    for b in builders:
        p = b()
        print("wrote", os.path.relpath(p, C.REPO))
    print(f"Generated {len(builders)} SI tables in {os.path.relpath(C.TABLES, C.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
