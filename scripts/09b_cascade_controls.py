"""
09b — Genealogical & Areal Controls for the Phonotactic–Morphological Cascade
===============================================================================
Tests whether the V-final → morphological strategy associations (§3.13) survive
controls for language-family non-independence and continental areal clustering.

Four complementary controls:
  B1: Mixed-effects model with family random intercepts
      feature ~ Vfinal + abs_lat + (1 | family)
  B2: + Macroarea fixed effects
      feature ~ Vfinal + abs_lat + macroarea + (1 | family)
  B3: Family-level aggregation (Spearman on family means)
  B4: Within-family permutation test (5,000 iterations)

Dependencies:
  - pandas, statsmodels  (for B1/B2 mixed-effects models)
  - Standard library only  (for B3/B4)

If statsmodels is unavailable, B1/B2 are skipped with a warning.
B3/B4 always run (no external dependencies).

Data paths (same as 09_cascade_grambank_wals.py):
  ASJP:     /content/asjp/cldf/
  Grambank: /content/grambank/cldf/
  WALS:     /content/wals/cldf/  (optional)

Output: Console summary tables matching §3.13 / §6.8 / Table S1 format.

Usage:
  python 09b_cascade_controls.py [--skip-perm] [--perm-n 5000]
"""

import csv
import sys
import random
from collections import defaultdict
from math import sqrt, erf, comb

# ================================================================
# OPTIONAL DEPENDENCY CHECK
# ================================================================

HAS_MIXED = False
try:
    import pandas as pd
    import statsmodels.formula.api as smf
    HAS_MIXED = True
except ImportError:
    print("=" * 80)
    print("WARNING: pandas/statsmodels not installed.")
    print("  B1 (family random intercepts) and B2 (+ macroarea) will be SKIPPED.")
    print("  B3 (family aggregation) and B4 (permutation) run with stdlib only.")
    print("  Install with: pip install pandas statsmodels")
    print("=" * 80)
    print()

# ================================================================
# CONFIGURATION
# ================================================================

SKIP_PERM = '--skip-perm' in sys.argv
PERM_N = 5000
for i, arg in enumerate(sys.argv):
    if arg == '--perm-n' and i + 1 < len(sys.argv):
        PERM_N = int(sys.argv[i + 1])

# The 8 Grambank features that survived FDR correction in §3.13
GB_FEATURES_CORE = {
    'GB080': 'Verb suffixes/enclitics (non-person)',
    'GB079': 'Verb prefixes/proclitics (non-person)',
    'GB089': 'S-argument suffix on verb',
    'GB090': 'S-argument prefix on verb',
    'GB091': 'A-argument suffix on verb',
    'GB092': 'A-argument prefix on verb',
    'GB093': 'P-argument suffix on verb',
    'GB094': 'P-argument prefix on verb',
}

GB_FEATURES_EXTENDED = {
    'GB070': 'Morphological case (non-pronominal)',
    'GB071': 'Morphological case (pronominal)',
    'GB082': 'Present tense marking on verbs',
    'GB083': 'Past tense marking on verbs',
    'GB084': 'Future tense marking on verbs',
    'GB086': 'Perfective/imperfective distinction',
    'GB133': 'Verb-final order',
    'GB136': 'Fixed constituent order',
    'GB044': 'Productive plural marking on nouns',
}

ALL_GB_FEATURES = {**GB_FEATURES_CORE, **GB_FEATURES_EXTENDED}

MIN_FAMILY_SIZE = 3
MIN_LANG_PER_FEAT = 30

MACROAREAS = ['Africa', 'Eurasia', 'Papunesia', 'Australia',
              'North America', 'South America']

# ================================================================
# STATISTICS (stdlib only)
# ================================================================

def _normal_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def _t_to_p(t, df):
    if df < 1:
        return 1.0
    if df > 30:
        return 2 * (1 - _normal_cdf(abs(t)))
    z = abs(t) * (1 - 1 / (4 * df))
    return 2 * (1 - _normal_cdf(z))


def pearson_r(x, y):
    n = len(x)
    if n < 5:
        return 0.0, 1.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = sqrt(sum((xi - mx) ** 2 for xi in x) / (n - 1))
    sy = sqrt(sum((yi - my) ** 2 for yi in y) / (n - 1))
    if sx == 0 or sy == 0:
        return 0.0, 1.0
    cov_val = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    r = max(-1.0, min(1.0, cov_val / (sx * sy)))
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * sqrt((n - 2) / (1 - r ** 2))
    p = _t_to_p(t, n - 2)
    return r, p


def spearman_r(x, y):
    def _rank(data):
        indexed = sorted(enumerate(data), key=lambda t: t[1])
        ranks = [0.0] * len(data)
        i = 0
        while i < len(indexed):
            j = i + 1
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j + 1) / 2
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks
    rx = _rank(x)
    ry = _rank(y)
    return pearson_r(rx, ry)


def _sign_test_p(n_positive, n_total):
    if n_total == 0:
        return 1.0
    k = max(n_positive, n_total - n_positive)
    p = 0.0
    for i in range(k, n_total + 1):
        p += comb(n_total, i) * (0.5 ** n_total)
    return min(1.0, 2 * p)


# ================================================================
# DATA LOADING
# ================================================================

VOWELS = set('ieE3auo')
CONSONANTS = set('pbfvmw8tdszclnrSZCjT5ykgxNqXh7L4G!')


def load_asjp_wordfinal():
    """Load ASJP: compute word-final V% per language, indexed by Glottocode."""
    languages = {}
    with open('/content/asjp/cldf/languages.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            languages[row['ID']] = {
                'glottocode': row.get('Glottocode', ''),
                'name': row.get('Name', ''),
                'family': row.get('Family', ''),
                'lat': float(row['Latitude']) if row.get('Latitude') else None,
            }

    lang_finals = defaultdict(lambda: {'v': 0, 'c': 0})
    with open('/content/asjp/cldf/forms.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('Loan') == 'true':
                continue
            form = row['Form']
            phonemes = [ch for ch in form if ch in VOWELS or ch in CONSONANTS]
            if not phonemes:
                continue
            final = phonemes[-1]
            lang_id = row['Language_ID']
            if final in VOWELS:
                lang_finals[lang_id]['v'] += 1
            else:
                lang_finals[lang_id]['c'] += 1

    lang_vfinal = {}
    for lang_id, counts in lang_finals.items():
        total = counts['v'] + counts['c']
        if total >= 20:
            gc = languages.get(lang_id, {}).get('glottocode', '')
            if gc:
                lat = languages.get(lang_id, {}).get('lat')
                lang_vfinal[gc] = {
                    'vfinal_pct': counts['v'] / total * 100,
                    'vfinal_prop': counts['v'] / total,
                    'name': languages.get(lang_id, {}).get('name', ''),
                    'family': languages.get(lang_id, {}).get('family', ''),
                    'lat': lat,
                    'abs_lat': abs(lat) if lat is not None else None,
                    'n_words': total,
                }
    return lang_vfinal


def load_grambank_with_macroarea():
    """Load Grambank features (binary) + Macroarea from languages.csv."""
    gb_macroarea = {}
    gb_langs_path = '/content/grambank/cldf/languages.csv'
    try:
        with open(gb_langs_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gc = row.get('Glottocode', row.get('ID', ''))
                ma = row.get('Macroarea', '')
                if gc and ma:
                    gb_macroarea[gc] = ma
                elif gc:
                    gb_macroarea[gc] = 'Unknown'
    except FileNotFoundError:
        print(f"  WARNING: {gb_langs_path} not found; Macroarea unavailable.")

    gb_data = defaultdict(dict)
    with open('/content/grambank/cldf/values.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            gc = row['Language_ID']
            feat = row['Parameter_ID']
            val = row['Value']
            if val in ('0', '1'):
                gb_data[gc][feat] = int(val)

    return gb_data, gb_macroarea


# ================================================================
# JOIN: Build analysis dataset
# ================================================================

def build_analysis_data(lang_vfinal, gb_data, gb_macroarea):
    """Join ASJP V-final data with Grambank features on Glottocode."""
    overlap = set(lang_vfinal.keys()) & set(gb_data.keys())
    print(f"\n  ASJP languages with Glottocode: {len(lang_vfinal)}")
    print(f"  Grambank languages: {len(gb_data)}")
    print(f"  Overlap (joined on Glottocode): {len(overlap)}")

    data = {}
    n_no_lat = 0
    for gc in overlap:
        vf = lang_vfinal[gc]
        if vf['abs_lat'] is None:
            n_no_lat += 1
            continue
        entry = {
            'vfinal_prop': vf['vfinal_prop'],
            'abs_lat': vf['abs_lat'],
            'family': vf['family'] if vf['family'] else 'Isolate',
            'macroarea': gb_macroarea.get(gc, 'Unknown'),
        }
        for feat_id in ALL_GB_FEATURES:
            if feat_id in gb_data[gc]:
                entry[feat_id] = gb_data[gc][feat_id]
        data[gc] = entry

    if n_no_lat:
        print(f"  Excluded (no latitude): {n_no_lat}")
    print(f"  Final analysis set: {len(data)} languages")

    families = set(d['family'] for d in data.values())
    macroareas = defaultdict(int)
    for d in data.values():
        macroareas[d['macroarea']] += 1
    print(f"  Language families: {len(families)}")
    print(f"  Macroarea distribution:")
    for ma in sorted(macroareas, key=macroareas.get, reverse=True):
        print(f"    {ma}: {macroareas[ma]}")

    return data


# ================================================================
# B1: MIXED-EFFECTS MODEL (family random intercept)
# B2: + MACROAREA FIXED EFFECT
# ================================================================

def run_mixed_effects(data, features):
    """
    B1: feature ~ vfinal_prop + abs_lat + (1 | family)
    B2: feature ~ vfinal_prop + abs_lat + macroarea + (1 | family)
    Linear probability model (LPM) via statsmodels MixedLM.
    """
    if not HAS_MIXED:
        print("\n  [SKIPPED] B1/B2: statsmodels not available.\n")
        return []

    results = []

    for feat_id, feat_name in sorted(features.items()):
        rows = []
        for gc, d in data.items():
            if feat_id in d:
                rows.append({
                    'feature': d[feat_id],
                    'vfinal': d['vfinal_prop'],
                    'abs_lat': d['abs_lat'],
                    'family': d['family'],
                    'macroarea': d['macroarea'],
                })

        if len(rows) < MIN_LANG_PER_FEAT:
            continue

        df = pd.DataFrame(rows)

        # Merge singleton families for model stability
        fam_counts = df['family'].value_counts()
        singletons = set(fam_counts[fam_counts == 1].index)
        df['family_re'] = df['family'].apply(
            lambda f: f if f not in singletons else '_Singleton'
        )

        n_families = df['family_re'].nunique()
        n_lang = len(df)

        # --- B1 ---
        b1_coef, b1_se, b1_p, b1_converged = None, None, None, False
        try:
            model_b1 = smf.mixedlm(
                "feature ~ vfinal + abs_lat",
                data=df,
                groups=df["family_re"],
            )
            result_b1 = model_b1.fit(reml=True, method='lbfgs', maxiter=500)
            b1_coef = result_b1.params.get('vfinal', None)
            b1_se = result_b1.bse.get('vfinal', None)
            b1_p = result_b1.pvalues.get('vfinal', None)
            b1_converged = getattr(result_b1, 'converged', True)
        except Exception as e:
            print(f"  B1 failed for {feat_id}: {e}")

        # --- B2 ---
        b2_coef, b2_se, b2_p, b2_converged = None, None, None, False
        try:
            if df['macroarea'].nunique() > 1:
                model_b2 = smf.mixedlm(
                    "feature ~ vfinal + abs_lat + C(macroarea)",
                    data=df,
                    groups=df["family_re"],
                )
                result_b2 = model_b2.fit(reml=True, method='lbfgs', maxiter=500)
                b2_coef = result_b2.params.get('vfinal', None)
                b2_se = result_b2.bse.get('vfinal', None)
                b2_p = result_b2.pvalues.get('vfinal', None)
                b2_converged = getattr(result_b2, 'converged', True)
        except Exception as e:
            print(f"  B2 failed for {feat_id}: {e}")

        # Raw correlation for comparison
        x_raw = [r['vfinal'] for r in rows]
        y_raw = [r['feature'] for r in rows]
        r_raw, p_raw = pearson_r(x_raw, y_raw)

        results.append({
            'feat_id': feat_id,
            'feat_name': feat_name,
            'n': n_lang,
            'n_families': n_families,
            'r_raw': r_raw,
            'p_raw': p_raw,
            'b1_coef': b1_coef,
            'b1_se': b1_se,
            'b1_p': b1_p,
            'b1_converged': b1_converged,
            'b2_coef': b2_coef,
            'b2_se': b2_se,
            'b2_p': b2_p,
            'b2_converged': b2_converged,
        })

    return results


# ================================================================
# B3: FAMILY-LEVEL AGGREGATION
# ================================================================

def run_family_aggregation(data, features):
    """
    Aggregate to family level: mean V-final and feature rate.
    Spearman correlation on family means.
    """
    results = []

    for feat_id, feat_name in sorted(features.items()):
        family_data = defaultdict(lambda: {'vfinals': [], 'features': []})
        for gc, d in data.items():
            if feat_id in d:
                fam = d['family']
                family_data[fam]['vfinals'].append(d['vfinal_prop'])
                family_data[fam]['features'].append(d[feat_id])

        fam_vfinal, fam_feat_rate, fam_sizes = [], [], []
        for fam, fd in family_data.items():
            if len(fd['vfinals']) >= MIN_FAMILY_SIZE:
                fam_vfinal.append(sum(fd['vfinals']) / len(fd['vfinals']))
                fam_feat_rate.append(sum(fd['features']) / len(fd['features']))
                fam_sizes.append(len(fd['vfinals']))

        if len(fam_vfinal) < 5:
            continue

        rs, ps = spearman_r(fam_vfinal, fam_feat_rate)
        rp, pp = pearson_r(fam_vfinal, fam_feat_rate)

        # Raw language-level correlation for sign comparison
        x_raw, y_raw = [], []
        for gc, d in data.items():
            if feat_id in d:
                x_raw.append(d['vfinal_prop'])
                y_raw.append(d[feat_id])
        r_raw, _ = pearson_r(x_raw, y_raw)

        sign_match = (rs > 0) == (r_raw > 0) if (rs != 0 and r_raw != 0) else None

        results.append({
            'feat_id': feat_id,
            'feat_name': feat_name,
            'n_families': len(fam_vfinal),
            'total_langs': sum(fam_sizes),
            'spearman_r': rs,
            'spearman_p': ps,
            'pearson_r': rp,
            'pearson_p': pp,
            'r_raw': r_raw,
            'sign_match': sign_match,
        })

    return results


# ================================================================
# B4: WITHIN-FAMILY PERMUTATION TEST
# ================================================================

def run_family_permutation(data, features, n_perm=5000):
    """
    Shuffle V-final within each family, recompute correlation.
    Empirical p = proportion of |r_perm| >= |r_obs|.
    """
    results = []

    for feat_id, feat_name in sorted(features.items()):
        family_groups = defaultdict(list)
        for gc, d in data.items():
            if feat_id in d:
                family_groups[d['family']].append((d['vfinal_prop'], d[feat_id]))

        all_pairs = []
        for pairs in family_groups.values():
            all_pairs.extend(pairs)

        if len(all_pairs) < MIN_LANG_PER_FEAT:
            continue

        x_obs = [p[0] for p in all_pairs]
        y_obs = [p[1] for p in all_pairs]
        r_obs, _ = pearson_r(x_obs, y_obs)

        n_extreme = 0
        for _ in range(n_perm):
            x_perm, y_perm = [], []
            for fam, pairs in family_groups.items():
                vf_vals = [p[0] for p in pairs]
                feat_vals = [p[1] for p in pairs]
                random.shuffle(vf_vals)
                x_perm.extend(vf_vals)
                y_perm.extend(feat_vals)
            r_perm, _ = pearson_r(x_perm, y_perm)
            if abs(r_perm) >= abs(r_obs):
                n_extreme += 1

        p_perm = (n_extreme + 1) / (n_perm + 1)

        results.append({
            'feat_id': feat_id,
            'feat_name': feat_name,
            'n': len(all_pairs),
            'n_families': len(family_groups),
            'r_obs': r_obs,
            'p_perm': p_perm,
            'n_perm': n_perm,
        })

    return results


# ================================================================
# SIGN CONSISTENCY: Binomial test across features
# ================================================================

def sign_consistency_test(mixed_results, family_results):
    print("\n" + "=" * 100)
    print("SIGN CONSISTENCY ACROSS CONTROLS")
    print("=" * 100)

    if mixed_results:
        n_match = sum(1 for r in mixed_results
                      if r['b1_coef'] is not None and r['r_raw'] != 0
                      and (r['b1_coef'] > 0) == (r['r_raw'] > 0))
        n_total = sum(1 for r in mixed_results
                      if r['b1_coef'] is not None and r['r_raw'] != 0)
        if n_total > 0:
            p = _sign_test_p(n_match, n_total)
            print(f"\n  B1 (family RI): {n_match}/{n_total} features preserve sign "
                  f"(binomial p = {p:.4f})")

        n_match = sum(1 for r in mixed_results
                      if r['b2_coef'] is not None and r['r_raw'] != 0
                      and (r['b2_coef'] > 0) == (r['r_raw'] > 0))
        n_total = sum(1 for r in mixed_results
                      if r['b2_coef'] is not None and r['r_raw'] != 0)
        if n_total > 0:
            p = _sign_test_p(n_match, n_total)
            print(f"  B2 (+ macroarea): {n_match}/{n_total} features preserve sign "
                  f"(binomial p = {p:.4f})")

    if family_results:
        n_match = sum(1 for r in family_results if r['sign_match'] is True)
        n_total = sum(1 for r in family_results if r['sign_match'] is not None)
        if n_total > 0:
            p = _sign_test_p(n_match, n_total)
            print(f"  B3 (family means): {n_match}/{n_total} features preserve sign "
                  f"(binomial p = {p:.4f})")

    print("\n  --- Core prefix/suffix features only ---")
    core_ids = set(GB_FEATURES_CORE.keys())
    if mixed_results:
        core_b1 = [r for r in mixed_results if r['feat_id'] in core_ids]
        n_match = sum(1 for r in core_b1
                      if r['b1_coef'] is not None and r['r_raw'] != 0
                      and (r['b1_coef'] > 0) == (r['r_raw'] > 0))
        n_total = sum(1 for r in core_b1
                      if r['b1_coef'] is not None and r['r_raw'] != 0)
        if n_total > 0:
            p = _sign_test_p(n_match, n_total)
            print(f"  B1 core: {n_match}/{n_total} preserve sign "
                  f"(binomial p = {p:.4f})")
            print(f"  Chance probability of {n_match}/{n_total}: "
                  f"2^-{n_total} = {2**(-n_total):.6f}")


# ================================================================
# DISPLAY FUNCTIONS
# ================================================================

def print_mixed_results(results):
    if not results:
        return
    print("\n" + "=" * 100)
    print("B1: MIXED-EFFECTS MODEL — feature ~ vfinal + abs_lat + (1 | family)")
    print("B2: MIXED-EFFECTS MODEL — feature ~ vfinal + abs_lat + macroarea + (1 | family)")
    print("=" * 100)

    header = (f"  {'Feature':<45s} {'n':>5s} {'r_raw':>7s} "
              f"{'B1_coef':>8s} {'B1_SE':>7s} {'B1_p':>10s} "
              f"{'B2_coef':>8s} {'B2_SE':>7s} {'B2_p':>10s} {'Sign':>5s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        sign = ''
        if r['b1_coef'] is not None and r['r_raw'] != 0:
            sign = 'Y' if (r['b1_coef'] > 0) == (r['r_raw'] > 0) else 'N'

        b1c = f"{r['b1_coef']:>+7.4f}" if r['b1_coef'] is not None else '    N/A'
        b1s = f"{r['b1_se']:>6.4f}" if r['b1_se'] is not None else '   N/A'
        b1p = f"{r['b1_p']:>9.2e}" if r['b1_p'] is not None else '      N/A'
        b2c = f"{r['b2_coef']:>+7.4f}" if r['b2_coef'] is not None else '    N/A'
        b2s = f"{r['b2_se']:>6.4f}" if r['b2_se'] is not None else '   N/A'
        b2p = f"{r['b2_p']:>9.2e}" if r['b2_p'] is not None else '      N/A'

        sig = ''
        if r['b1_p'] is not None:
            if r['b1_p'] < 0.001: sig = '***'
            elif r['b1_p'] < 0.01: sig = '**'
            elif r['b1_p'] < 0.05: sig = '*'

        print(f"  {r['feat_name']:<45s} {r['n']:>5d} {r['r_raw']:>+6.4f} "
              f"{b1c} {b1s} {b1p} {b2c} {b2s} {b2p} {sign:>5s} {sig}")

    print()
    print("  Note: LPM coefficients. Sign column: Y = B1 preserves raw correlation sign.")
    print("  Attenuation relative to r_raw is expected (Vfinal-latitude collinearity).")


def print_family_results(results):
    if not results:
        return
    print("\n" + "=" * 100)
    print(f"B3: FAMILY-LEVEL AGGREGATION (Spearman, min family size = {MIN_FAMILY_SIZE})")
    print("=" * 100)

    header = (f"  {'Feature':<45s} {'n_fam':>6s} {'n_lang':>7s} "
              f"{'rho_fam':>8s} {'p_fam':>10s} {'r_raw':>7s} {'Sign':>5s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        sign = 'Y' if r['sign_match'] is True else ('N' if r['sign_match'] is False else '')
        sig = ''
        if r['spearman_p'] < 0.001: sig = '***'
        elif r['spearman_p'] < 0.01: sig = '**'
        elif r['spearman_p'] < 0.05: sig = '*'

        print(f"  {r['feat_name']:<45s} {r['n_families']:>6d} {r['total_langs']:>7d} "
              f"{r['spearman_r']:>+7.4f} {r['spearman_p']:>9.2e} {r['r_raw']:>+6.4f} "
              f"{sign:>5s} {sig}")


def print_permutation_results(results):
    if not results:
        return
    print("\n" + "=" * 100)
    print(f"B4: WITHIN-FAMILY PERMUTATION TEST ({results[0]['n_perm']} iterations)")
    print("=" * 100)

    header = (f"  {'Feature':<45s} {'n':>5s} {'n_fam':>6s} "
              f"{'r_obs':>8s} {'p_perm':>10s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        sig = ''
        if r['p_perm'] < 0.001: sig = '***'
        elif r['p_perm'] < 0.01: sig = '**'
        elif r['p_perm'] < 0.05: sig = '*'

        print(f"  {r['feat_name']:<45s} {r['n']:>5d} {r['n_families']:>6d} "
              f"{r['r_obs']:>+7.4f} {r['p_perm']:>9.4f} {sig}")

    print()
    print("  p_perm = proportion of permuted |r| >= observed |r| (two-tailed).")


# ================================================================
# COMBINED SUMMARY TABLE (Table S1)
# ================================================================

def print_combined_summary(mixed_res, family_res, perm_res, features):
    print("\n" + "=" * 100)
    print("TABLE S1: COMBINED SUMMARY — Genealogical and Areal Controls")
    print("=" * 100)

    mixed_by_id = {r['feat_id']: r for r in mixed_res} if mixed_res else {}
    fam_by_id = {r['feat_id']: r for r in family_res} if family_res else {}
    perm_by_id = {r['feat_id']: r for r in perm_res} if perm_res else {}

    header = (f"  {'Feature':<40s} {'n':>5s} "
              f"{'r_raw':>7s} "
              f"{'B1b':>8s} {'B1p':>8s} "
              f"{'B2b':>8s} {'B2p':>8s} "
              f"{'B3rho':>7s} {'B3p':>8s} "
              f"{'B4p':>8s} "
              f"{'All':>4s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for feat_id, feat_name in sorted(features.items()):
        m = mixed_by_id.get(feat_id, {})
        f = fam_by_id.get(feat_id, {})
        p = perm_by_id.get(feat_id, {})

        n = m.get('n', f.get('total_langs', p.get('n', '')))
        r_raw = m.get('r_raw', f.get('r_raw', 0))

        b1c = f"{m['b1_coef']:>+7.4f}" if m.get('b1_coef') is not None else '     --'
        b1p = f"{m['b1_p']:>7.2e}" if m.get('b1_p') is not None else '      --'
        b2c = f"{m['b2_coef']:>+7.4f}" if m.get('b2_coef') is not None else '     --'
        b2p = f"{m['b2_p']:>7.2e}" if m.get('b2_p') is not None else '      --'
        b3r = f"{f['spearman_r']:>+6.4f}" if f.get('spearman_r') is not None else '    --'
        b3p = f"{f['spearman_p']:>7.2e}" if f.get('spearman_p') is not None else '      --'
        b4p = f"{p['p_perm']:>7.4f}" if p.get('p_perm') is not None else '      --'

        signs = []
        if m.get('b1_coef') is not None and r_raw != 0:
            signs.append((m['b1_coef'] > 0) == (r_raw > 0))
        if m.get('b2_coef') is not None and r_raw != 0:
            signs.append((m['b2_coef'] > 0) == (r_raw > 0))
        if f.get('sign_match') is not None:
            signs.append(f['sign_match'])

        all_ok = 'Y' if signs and all(signs) else ('N' if signs else '--')

        n_str = f"{n:>5d}" if isinstance(n, int) else f"{n:>5s}"
        print(f"  {feat_name:<40s} {n_str} "
              f"{r_raw:>+6.4f} "
              f"{b1c} {b1p} "
              f"{b2c} {b2p} "
              f"{b3r} {b3p} "
              f"{b4p} "
              f"{all_ok:>4s}")

    print()
    print("  r_raw = raw Pearson; B1b/B2b = LPM Vfinal coefficient;")
    print("  B3rho = family-mean Spearman; B4p = within-family permutation p;")
    print("  All = sign preserved across all controls")


# ================================================================
# COLLINEARITY DIAGNOSTIC
# ================================================================

def collinearity_check(data):
    x, y = [], []
    for d in data.values():
        x.append(d['vfinal_prop'])
        y.append(d['abs_lat'])
    r, p = pearson_r(x, y)
    print(f"\n  Collinearity check: Vfinal ~ abs_lat")
    print(f"    r = {r:+.4f}, p = {p:.2e}, n = {len(x)}")
    print(f"    VIF (approx) = 1/(1-r^2) = {1/(1-r**2):.2f}")
    if abs(r) > 0.5:
        print(f"    NOTE: Substantial collinearity. Vfinal coefficient in B1/B2")
        print(f"    will be attenuated. Sign preservation is the key test.")
    print()


# ================================================================
# MAIN
# ================================================================

def main():
    random.seed(42)

    print("=" * 100)
    print("09b — GENEALOGICAL & AREAL CONTROLS FOR PHONOTACTIC-MORPHOLOGICAL CASCADE")
    print("=" * 100)

    print("\n[1/5] Loading ASJP word-final V% ...")
    lang_vfinal = load_asjp_wordfinal()
    print(f"  Loaded {len(lang_vfinal)} languages with V-final data.")

    print("\n[2/5] Loading Grambank features + Macroarea ...")
    gb_data, gb_macroarea = load_grambank_with_macroarea()
    print(f"  Loaded {len(gb_data)} Grambank languages.")
    print(f"  Macroarea available for {len(gb_macroarea)} languages.")

    print("\n[3/5] Building analysis dataset (join on Glottocode) ...")
    data = build_analysis_data(lang_vfinal, gb_data, gb_macroarea)

    if not data:
        print("ERROR: No overlapping data found. Check file paths.")
        sys.exit(1)

    collinearity_check(data)

    # ==== CORE FEATURES ====
    print("\n" + "#" * 100)
    print("# CORE FEATURES: Prefix/Suffix Pairs (8 features)")
    print("#" * 100)

    print("\n[4a/5] B1/B2: Mixed-effects models (core features) ...")
    mixed_core = run_mixed_effects(data, GB_FEATURES_CORE)
    print_mixed_results(mixed_core)

    print("\n[4b/5] B3: Family-level aggregation (core features) ...")
    family_core = run_family_aggregation(data, GB_FEATURES_CORE)
    print_family_results(family_core)

    perm_core = []
    if not SKIP_PERM:
        print(f"\n[4c/5] B4: Within-family permutation ({PERM_N} iters, core) ...")
        print("  This may take a few minutes ...")
        perm_core = run_family_permutation(data, GB_FEATURES_CORE, n_perm=PERM_N)
        print_permutation_results(perm_core)
    else:
        print("\n[4c/5] B4: SKIPPED (--skip-perm flag)")

    print_combined_summary(mixed_core, family_core, perm_core, GB_FEATURES_CORE)
    sign_consistency_test(mixed_core, family_core)

    # ==== EXTENDED FEATURES ====
    print("\n\n" + "#" * 100)
    print("# EXTENDED FEATURES: Additional Grambank features (9 features)")
    print("#" * 100)

    mixed_ext = run_mixed_effects(data, GB_FEATURES_EXTENDED)
    print_mixed_results(mixed_ext)

    family_ext = run_family_aggregation(data, GB_FEATURES_EXTENDED)
    print_family_results(family_ext)

    perm_ext = []
    if not SKIP_PERM:
        perm_ext = run_family_permutation(data, GB_FEATURES_EXTENDED, n_perm=PERM_N)
        print_permutation_results(perm_ext)

    print_combined_summary(mixed_ext, family_ext, perm_ext, GB_FEATURES_EXTENDED)

    # ==== FINAL ====
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\n  B1: Mixed-effects LPM with family random intercepts — "
          f"{'DONE' if mixed_core else 'SKIPPED'}")
    print(f"  B2: + Macroarea fixed effects — "
          f"{'DONE' if mixed_core else 'SKIPPED'}")
    print(f"  B3: Family-level aggregation (Spearman) — DONE")
    print(f"  B4: Within-family permutation ({PERM_N} iters) — "
          f"{'DONE' if perm_core else 'SKIPPED'}")
    print()
    print("  Key: If all 8 core features preserve sign -> genealogical criticism refuted.")
    print(f"  Binomial probability of 8/8 by chance: 2^-8 = {2**(-8):.4f}")
    print()


if __name__ == '__main__':
    main()
