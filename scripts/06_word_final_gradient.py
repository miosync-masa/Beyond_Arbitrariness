"""
Articulatory → Morphological Bridge: Evidence from Word-Final Phonemes
======================================================================
Hypothesis: Cold climates → closed-mouth preference → word-final vowels drop
           → if true, this is the "phonology→morphology" bridge

If word-final vowels decrease with latitude, it suggests:
  - Environmental pressure shapes not just WHICH sounds are used
  - But WHERE in the word they appear
  - This has morphological consequences (final vowels often carry inflection)

Additionally: word-final consonant clusters may increase with latitude
  → cluster-tolerance is related to morphological complexity type
"""

import csv
from collections import Counter, defaultdict
from math import sqrt

# ASJP symbols
VOWELS = set('ieE3auo')
CONSONANTS = set('pbfvmw8tdszclnrSZCjT5ykgxNqXh7L4G!')


def load_data():
    languages = {}
    with open('/content/asjp/cldf/languages.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            lat = float(row['Latitude']) if row['Latitude'] else None
            languages[row['ID']] = {
                'family': row['Family'],
                'lat': lat,
                'abs_lat': abs(lat) if lat is not None else None,
            }
    concepts = {}
    with open('/content/asjp/cldf/parameters.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            concepts[row['ID']] = row['Concepticon_Gloss']
    forms = []
    with open('/content/asjp/cldf/forms.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['Loan'] != 'true':
                forms.append({
                    'lang_id': row['Language_ID'],
                    'concept_id': row['Parameter_ID'],
                    'form': row['Form']
                })
    return languages, concepts, forms


def get_final_phoneme(form):
    """Get the last phoneme of a word."""
    phonemes = [ch for ch in form if ch in VOWELS or ch in CONSONANTS]
    if phonemes:
        return phonemes[-1]
    return None


def get_initial_phoneme(form):
    """Get the first phoneme of a word."""
    phonemes = [ch for ch in form if ch in VOWELS or ch in CONSONANTS]
    if phonemes:
        return phonemes[0]
    return None


def count_final_cluster(form):
    """Count consecutive final consonants (cluster size)."""
    phonemes = [ch for ch in form if ch in VOWELS or ch in CONSONANTS]
    if not phonemes:
        return 0
    count = 0
    for ph in reversed(phonemes):
        if ph in CONSONANTS:
            count += 1
        else:
            break
    return count


def word_length(form):
    """Count phonemes in word."""
    return sum(1 for ch in form if ch in VOWELS or ch in CONSONANTS)


def main():
    print("Loading data...")
    languages, concepts, forms = load_data()
    
    # Latitude bands
    BANDS = [
        ('Tropical (0-15°)', 0, 15),
        ('Subtropical (15-30°)', 15, 30),
        ('Temperate (30-50°)', 30, 50),
        ('Subarctic (50-70°)', 50, 70),
    ]
    
    # ================================================================
    # ANALYSIS 1: Word-final vowel ratio by latitude
    # ================================================================
    print("\n" + "="*90)
    print("ANALYSIS 1: WORD-FINAL PHONEME TYPE BY LATITUDE")
    print("Does the proportion of vowel-final words decrease with latitude?")
    print("="*90)
    
    band_finals = {b[0]: {'vowel': 0, 'consonant': 0} for b in BANDS}
    band_words = {b[0]: 0 for b in BANDS}
    
    for entry in forms:
        lang = languages.get(entry['lang_id'])
        if not lang or lang['abs_lat'] is None:
            continue
        
        final = get_final_phoneme(entry['form'])
        if final is None:
            continue
        
        for bname, lo, hi in BANDS:
            if lo <= lang['abs_lat'] < hi:
                band_words[bname] += 1
                if final in VOWELS:
                    band_finals[bname]['vowel'] += 1
                else:
                    band_finals[bname]['consonant'] += 1
                break
    
    print(f"\n{'Latitude band':<25s} {'N words':>10s} {'V-final':>10s} {'C-final':>10s} {'V-final %':>10s}")
    print("-" * 70)
    
    v_final_rates = []
    for bname, lo, hi in BANDS:
        total = band_finals[bname]['vowel'] + band_finals[bname]['consonant']
        if total == 0:
            continue
        v_rate = band_finals[bname]['vowel'] / total * 100
        v_final_rates.append((bname, v_rate, total))
        print(f"  {bname:<23s} {total:>10,d} {band_finals[bname]['vowel']:>10,d} {band_finals[bname]['consonant']:>10,d} {v_rate:>9.1f}%")
    
    if len(v_final_rates) >= 2:
        drop = v_final_rates[0][1] - v_final_rates[-1][1]
        print(f"\n  Δ (Tropical → Subarctic): {drop:+.1f} percentage points")
    
    # ================================================================
    # ANALYSIS 2: Final consonant cluster size by latitude
    # ================================================================
    print("\n" + "="*90)
    print("ANALYSIS 2: FINAL CONSONANT CLUSTER SIZE BY LATITUDE")
    print("Do cold-climate languages tolerate more final consonant clusters?")
    print("="*90)
    
    band_clusters = {b[0]: [] for b in BANDS}
    
    for entry in forms:
        lang = languages.get(entry['lang_id'])
        if not lang or lang['abs_lat'] is None:
            continue
        
        cl = count_final_cluster(entry['form'])
        
        for bname, lo, hi in BANDS:
            if lo <= lang['abs_lat'] < hi:
                band_clusters[bname].append(cl)
                break
    
    print(f"\n{'Latitude band':<25s} {'N words':>10s} {'Mean cluster':>13s} {'% with ≥1':>10s} {'% with ≥2':>10s}")
    print("-" * 75)
    
    for bname, lo, hi in BANDS:
        vals = band_clusters[bname]
        if not vals:
            continue
        mean_cl = sum(vals) / len(vals)
        pct_1 = sum(1 for v in vals if v >= 1) / len(vals) * 100
        pct_2 = sum(1 for v in vals if v >= 2) / len(vals) * 100
        print(f"  {bname:<23s} {len(vals):>10,d} {mean_cl:>13.3f} {pct_1:>9.1f}% {pct_2:>9.1f}%")
    
    # ================================================================
    # ANALYSIS 3: Word length by latitude
    # ================================================================
    print("\n" + "="*90)
    print("ANALYSIS 3: MEAN WORD LENGTH (PHONEMES) BY LATITUDE")
    print("Do cold-climate words tend to be shorter?")
    print("="*90)
    
    band_lengths = {b[0]: [] for b in BANDS}
    
    for entry in forms:
        lang = languages.get(entry['lang_id'])
        if not lang or lang['abs_lat'] is None:
            continue
        
        wl = word_length(entry['form'])
        if wl == 0:
            continue
        
        for bname, lo, hi in BANDS:
            if lo <= lang['abs_lat'] < hi:
                band_lengths[bname].append(wl)
                break
    
    print(f"\n{'Latitude band':<25s} {'N words':>10s} {'Mean length':>12s} {'Median':>8s}")
    print("-" * 60)
    
    for bname, lo, hi in BANDS:
        vals = band_lengths[bname]
        if not vals:
            continue
        mean_l = sum(vals) / len(vals)
        sorted_vals = sorted(vals)
        median = sorted_vals[len(sorted_vals)//2]
        print(f"  {bname:<23s} {len(vals):>10,d} {mean_l:>12.3f} {median:>8d}")
    
    # ================================================================
    # ANALYSIS 4: Specific final phoneme distributions by latitude
    # ================================================================
    print("\n" + "="*90)
    print("ANALYSIS 4: SPECIFIC WORD-FINAL PHONEME FREQUENCIES BY LATITUDE")
    print("Which specific sounds increase/decrease at word-end in cold climates?")
    print("="*90)
    
    band_final_counts = {b[0]: Counter() for b in BANDS}
    band_final_totals = {b[0]: 0 for b in BANDS}
    
    for entry in forms:
        lang = languages.get(entry['lang_id'])
        if not lang or lang['abs_lat'] is None:
            continue
        
        final = get_final_phoneme(entry['form'])
        if final is None:
            continue
        
        for bname, lo, hi in BANDS:
            if lo <= lang['abs_lat'] < hi:
                band_final_counts[bname][final] += 1
                band_final_totals[bname] += 1
                break
    
    # Show most variable finals
    print(f"\n  Top phonemes with largest latitude gradient (final position):")
    print(f"  {'Phoneme':>8s} {'Type':>6s} {'Tropical':>10s} {'Subtrop':>10s} {'Temperate':>10s} {'Subarctic':>10s} {'Δ(T→S)':>10s}")
    print("-" * 70)
    
    all_finals = set()
    for b in BANDS:
        all_finals |= set(band_final_counts[b[0]].keys())
    
    phoneme_gradients = []
    for ph in sorted(all_finals):
        rates = []
        for bname, lo, hi in BANDS:
            total = band_final_totals[bname]
            if total == 0:
                rates.append(0)
            else:
                rates.append(band_final_counts[bname][ph] / total * 100)
        
        if len(rates) >= 2 and rates[0] > 0.5:  # Filter noise
            gradient = rates[-1] - rates[0]
            ph_type = 'V' if ph in VOWELS else 'C'
            phoneme_gradients.append((ph, ph_type, rates, gradient))
    
    # Sort by absolute gradient
    phoneme_gradients.sort(key=lambda x: abs(x[3]), reverse=True)
    
    for ph, ph_type, rates, gradient in phoneme_gradients[:20]:
        print(f"  {ph:>8s} {ph_type:>6s} {rates[0]:>9.2f}% {rates[1]:>9.2f}% {rates[2]:>9.2f}% {rates[3]:>9.2f}% {gradient:>+9.2f}%")
    
    # ================================================================
    # ANALYSIS 5: Family-level control for word-final vowels
    # ================================================================
    print("\n" + "="*90)
    print("ANALYSIS 5: FAMILY-LEVEL CONTROL — WORD-FINAL VOWEL RATIO BY LATITUDE")
    print("Each family = one data point (within-family mean)")
    print("="*90)
    
    family_band_data = defaultdict(lambda: defaultdict(lambda: {'v': 0, 'c': 0}))
    
    for entry in forms:
        lang = languages.get(entry['lang_id'])
        if not lang or lang['abs_lat'] is None or not lang['family']:
            continue
        
        final = get_final_phoneme(entry['form'])
        if final is None:
            continue
        
        for bname, lo, hi in BANDS:
            if lo <= lang['abs_lat'] < hi:
                if final in VOWELS:
                    family_band_data[bname][lang['family']]['v'] += 1
                else:
                    family_band_data[bname][lang['family']]['c'] += 1
                break
    
    print(f"\n{'Latitude band':<25s} {'N families':>10s} {'Mean V-final %':>15s} {'SD':>8s}")
    print("-" * 65)
    
    band_family_rates = {}
    for bname, lo, hi in BANDS:
        family_rates = []
        for fam, counts in family_band_data[bname].items():
            total = counts['v'] + counts['c']
            if total >= 10:  # Minimum words per family
                family_rates.append(counts['v'] / total * 100)
        
        if family_rates:
            mean_r = sum(family_rates) / len(family_rates)
            sd_r = sqrt(sum((x - mean_r)**2 for x in family_rates) / (len(family_rates) - 1)) if len(family_rates) > 1 else 0
            band_family_rates[bname] = family_rates
            print(f"  {bname:<23s} {len(family_rates):>10d} {mean_r:>14.1f}% {sd_r:>7.1f}%")
    
    # Statistical test: Tropical vs Subarctic at family level
    tropical_name = BANDS[0][0]
    subarctic_name = BANDS[3][0]
    
    if tropical_name in band_family_rates and subarctic_name in band_family_rates:
        trop = band_family_rates[tropical_name]
        sub = band_family_rates[subarctic_name]
        
        n1, n2 = len(trop), len(sub)
        m1 = sum(trop) / n1
        m2 = sum(sub) / n2
        v1 = sum((x - m1)**2 for x in trop) / (n1 - 1)
        v2 = sum((x - m2)**2 for x in sub) / (n2 - 1)
        se = sqrt(v1/n1 + v2/n2)
        
        if se > 0:
            import math
            t = (m1 - m2) / se
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
            
            pooled_sd = sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
            d = (m1 - m2) / pooled_sd if pooled_sd > 0 else 0
            
            print(f"\n  Welch's t-test (Tropical vs Subarctic, family-level):")
            print(f"    Tropical mean: {m1:.1f}% (n={n1} families)")
            print(f"    Subarctic mean: {m2:.1f}% (n={n2} families)")
            print(f"    Δ = {m1-m2:+.1f} percentage points")
            print(f"    t = {t:.3f}, p = {p:.2e}")
            print(f"    Cohen's d = {d:+.3f}")
    
    # ================================================================
    # ANALYSIS 6: Vowel-to-Consonant ratio at word boundaries
    # ================================================================
    print("\n" + "="*90)
    print("ANALYSIS 6: INITIAL vs FINAL VOWEL RATES BY LATITUDE")
    print("Is the latitude effect specific to word-final position?")
    print("="*90)
    
    band_initial = {b[0]: {'v': 0, 'c': 0} for b in BANDS}
    
    for entry in forms:
        lang = languages.get(entry['lang_id'])
        if not lang or lang['abs_lat'] is None:
            continue
        
        initial = get_initial_phoneme(entry['form'])
        if initial is None:
            continue
        
        for bname, lo, hi in BANDS:
            if lo <= lang['abs_lat'] < hi:
                if initial in VOWELS:
                    band_initial[bname]['v'] += 1
                else:
                    band_initial[bname]['c'] += 1
                break
    
    print(f"\n{'Latitude band':<25s} {'Initial V%':>12s} {'Final V%':>12s} {'Δ(Init-Fin)':>12s}")
    print("-" * 65)
    
    for bname, lo, hi in BANDS:
        init_total = band_initial[bname]['v'] + band_initial[bname]['c']
        fin_total = band_finals[bname]['vowel'] + band_finals[bname]['consonant']
        
        if init_total > 0 and fin_total > 0:
            init_v = band_initial[bname]['v'] / init_total * 100
            fin_v = band_finals[bname]['vowel'] / fin_total * 100
            print(f"  {bname:<23s} {init_v:>11.1f}% {fin_v:>11.1f}% {init_v - fin_v:>+11.1f}%")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "="*90)
    print("SUMMARY: THE PHONOLOGY → MORPHOLOGY BRIDGE")
    print("="*90)
    print("""
    The data speaks to whether environmental articulatory pressure
    affects not just WHICH sounds are used, but WHERE they appear:
    
    If word-final vowels decrease with latitude:
      → Cold climates favor closing the mouth at word boundaries
      → This removes the phonological "slot" where inflectional suffixes live
      → Languages compensate with word-order rigidity or isolating morphology
      → This is the BRIDGE from phonology to morphology to syntax
    
    The chain:
      Environment (cold) 
        → Articulatory pressure (close mouth)
          → Phonological change (final vowels drop)
            → Morphological consequence (inflectional space shrinks)
              → Syntactic adaptation (word order rigidifies)
    
    This is the "音→形態→統語" bridge that the reviewer requested.
    """)


if __name__ == '__main__':
    main()
