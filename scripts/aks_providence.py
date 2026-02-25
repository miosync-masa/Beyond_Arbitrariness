#!/usr/bin/env python3
"""
AKS 4D Validation: Providence Corpus Direct Analysis
=====================================================
Direct CHAT file parsing → %xpho (actual) vs %xmod (target) → 4D analysis
"""

import os
import re
import glob
import numpy as np
from collections import defaultdict

# ============================================================
# IPA FEATURE MAPS
# ============================================================
IPA_VOICING = {}
for c in ['p','t','k','q','ʔ','c','f','θ','s','ʃ','ʂ','ç','x','χ','ħ','h','ɬ']:
    IPA_VOICING[c] = 0
for c in ['b','d','ɡ','g','ɢ','ɟ','v','ð','z','ʒ','ʐ','ʝ','ɣ','ʁ','ʕ','ɦ','ɮ']:
    IPA_VOICING[c] = 1
for c in ['m','n','ɲ','ŋ','ɳ','ɴ','r','ɾ','ɽ','l','ɭ','ʎ','ɻ','w','j','ʋ','ɹ','ɰ']:
    IPA_VOICING[c] = 1

IPA_SONORITY = {}
for c in ['p','t','k','q','ʔ','c']: IPA_SONORITY[c] = 1        # voiceless stops
for c in ['b','d','ɡ','g','ɢ','ɟ']: IPA_SONORITY[c] = 2        # voiced stops
for c in ['f','θ','s','ʃ','ʂ','ç','x','χ','ħ','h','ɬ']: IPA_SONORITY[c] = 3  # vl fricatives
for c in ['v','ð','z','ʒ','ʐ','ʝ','ɣ','ʁ','ʕ','ɦ','ɮ']: IPA_SONORITY[c] = 4  # vd fricatives
for c in ['m','n','ɲ','ŋ','ɳ','ɴ']: IPA_SONORITY[c] = 6        # nasals
for c in ['r','ɾ','ɽ','l','ɭ','ʎ','ɻ','w','j','ʋ','ɹ','ɰ']: IPA_SONORITY[c] = 7  # approximants

IPA_PLACE = {}
for c in ['p','b','m','f','v','w','ʋ']: IPA_PLACE[c] = 1           # labial
for c in ['θ','ð','t','d','n','s','z','l','ɹ','r','ɾ','ɽ','ɬ','ɮ','ɭ','ɳ']: IPA_PLACE[c] = 2  # alveolar
for c in ['ʃ','ʒ','ʂ','ʐ','ɲ','ʎ','j','ɻ','c','ɟ']: IPA_PLACE[c] = 3  # palatal
for c in ['k','ɡ','g','ŋ','x','ɣ','ɰ']: IPA_PLACE[c] = 4          # velar
for c in ['q','ɢ','χ','ʁ','ɴ','ħ','ʕ','h','ɦ','ʔ']: IPA_PLACE[c] = 5  # uvular+

IPA_OPENNESS = {}
for v in ['i','ɪ','y','ʏ','ɨ','ʉ','ɯ','u','ʊ']: IPA_OPENNESS[v] = 1   # close
for v in ['e','ø','ɘ','ɵ','ɤ','o']: IPA_OPENNESS[v] = 2                  # mid
for v in ['ə']: IPA_OPENNESS[v] = 2.5                                      # schwa
for v in ['ɛ','œ','ɜ','ɞ','ʌ','ɔ']: IPA_OPENNESS[v] = 3                  # open-mid
for v in ['æ','ɐ','a','ɶ','ɑ','ɒ']: IPA_OPENNESS[v] = 4                  # open

DIACRITICS = set('ːˈˌʰʷʲˀˑʼ̥̩̯̠̪̤̰̃̚˞̝̞̘̙')

def extract_features(ipa_str):
    """Extract mean 4D features from IPA string."""
    voicing, sonority, place, openness = [], [], [], []
    for ch in ipa_str:
        if ch in DIACRITICS or ch in ' \t':
            continue
        if ch in IPA_VOICING: voicing.append(IPA_VOICING[ch])
        if ch in IPA_SONORITY: sonority.append(IPA_SONORITY[ch])
        if ch in IPA_PLACE: place.append(IPA_PLACE[ch])
        if ch in IPA_OPENNESS: openness.append(IPA_OPENNESS[ch])
    return {
        'voicing': np.mean(voicing) if voicing else None,
        'sonority': np.mean(sonority) if sonority else None,
        'place': np.mean(place) if place else None,
        'openness': np.mean(openness) if openness else None,
        'n_consonants': len(voicing),
        'n_vowels': len(openness),
    }

def extract_consonant_types(ipa_str):
    """Extract detailed consonant type counts."""
    stops = 0; fricatives = 0; nasals = 0; approx = 0
    labial = 0; alveolar = 0; velar = 0; palatal = 0
    voiced_c = 0; voiceless_c = 0

    for ch in ipa_str:
        if ch in DIACRITICS or ch in ' \t':
            continue
        if ch in IPA_SONORITY:
            s = IPA_SONORITY[ch]
            if s <= 2: stops += 1
            elif s <= 4: fricatives += 1
            elif s == 6: nasals += 1
            elif s == 7: approx += 1
        if ch in IPA_VOICING:
            if IPA_VOICING[ch] == 0: voiceless_c += 1
            else: voiced_c += 1
        if ch in IPA_PLACE:
            p = IPA_PLACE[ch]
            if p == 1: labial += 1
            elif p == 2: alveolar += 1
            elif p == 3: palatal += 1
            elif p == 4: velar += 1

    total_c = stops + fricatives + nasals + approx
    return {
        'stop_pct': stops / total_c if total_c > 0 else None,
        'fric_pct': fricatives / total_c if total_c > 0 else None,
        'nasal_pct': nasals / total_c if total_c > 0 else None,
        'approx_pct': approx / total_c if total_c > 0 else None,
        'labial_pct': labial / total_c if total_c > 0 else None,
        'alveolar_pct': alveolar / total_c if total_c > 0 else None,
        'velar_pct': velar / total_c if total_c > 0 else None,
        'voiced_pct': voiced_c / (voiced_c + voiceless_c) if (voiced_c + voiceless_c) > 0 else None,
        'total_c': total_c,
    }


# ============================================================
# CHAT FILE PARSER
# ============================================================
def parse_chat_file(filepath):
    """Parse a single CHAT file, extract CHI utterances with %xpho/%xmod and age."""
    child_age_months = None
    child_name = None
    utterances = []

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract child age from @ID line
    for line in lines:
        if line.startswith('@ID:') and 'Target_Child' in line:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                age_str = parts[3].strip()  # e.g., "1;04.27"
                match = re.match(r'(\d+);(\d+)(?:\.(\d+))?', age_str)
                if match:
                    years = int(match.group(1))
                    months = int(match.group(2))
                    days = int(match.group(3)) if match.group(3) else 0
                    child_age_months = years * 12 + months + days / 30.0

    # Extract from folder name
    parts = filepath.replace('\\', '/').split('/')
    for p in parts:
        if p in ['Alex', 'Ethan', 'Lily', 'Naima', 'Violet', 'William']:
            child_name = p
            break

    # Parse utterances
    current_speaker = None
    current_text = None
    current_xmod = None
    current_xpho = None

    for line in lines:
        line = line.rstrip('\n')

        if line.startswith('*'):
            # Save previous utterance
            if current_speaker == 'CHI' and current_xpho is not None:
                utterances.append({
                    'child': child_name,
                    'age_months': child_age_months,
                    'text': current_text,
                    'xmod': current_xmod,
                    'xpho': current_xpho,
                })
            # Start new utterance
            parts = line.split(':\t', 1)
            current_speaker = parts[0].replace('*', '')
            current_text = parts[1] if len(parts) > 1 else ''
            current_xmod = None
            current_xpho = None

        elif line.startswith('%xmod:'):
            current_xmod = line[6:].strip()
        elif line.startswith('%xpho:'):
            current_xpho = line[6:].strip()

    # Don't forget last utterance
    if current_speaker == 'CHI' and current_xpho is not None:
        utterances.append({
            'child': child_name,
            'age_months': child_age_months,
            'text': current_text,
            'xmod': current_xmod,
            'xpho': current_xpho,
        })

    return utterances


# ============================================================
# MAIN ANALYSIS
# ============================================================
def main():
    base_dir = '/home/claude/Providence'
    cha_files = sorted(glob.glob(os.path.join(base_dir, '*', '*.cha')))
    print(f"Found {len(cha_files)} .cha files")

    # Parse all files
    all_utts = []
    for f in cha_files:
        utts = parse_chat_file(f)
        all_utts.extend(utts)

    print(f"Total CHI utterances with %xpho: {len(all_utts)}")

    # Split babbling vs real words
    babbling = [u for u in all_utts if u['xmod'] is None or '*' in u['xmod']]
    real_words = [u for u in all_utts if u['xmod'] is not None and '*' not in u['xmod']]
    print(f"  Babbling: {len(babbling)}")
    print(f"  Real words: {len(real_words)}")

    # Per-child counts
    child_counts = defaultdict(int)
    for u in all_utts:
        child_counts[u['child']] += 1
    print(f"\nPer child: {dict(child_counts)}")

    # ================================================================
    # 4D ANALYSIS BY AGE BIN
    # ================================================================
    age_bins = [
        ('11-15', 11, 15),
        ('15-18', 15, 18),
        ('18-21', 18, 21),
        ('21-24', 21, 24),
        ('24-30', 24, 30),
        ('30-36', 30, 36),
        ('36-42', 36, 42),
        ('42-48', 42, 48),
    ]

    dims = ['voicing', 'sonority', 'place', 'openness']

    print(f"\n{'='*90}")
    print("AKS 4D VALIDATION: Child Actual (%xpho) vs Adult Target (%xmod)")
    print(f"{'='*90}")

    # Collect data
    bin_data = defaultdict(lambda: {
        'pho': {d: [] for d in dims},
        'mod': {d: [] for d in dims},
        'pho_detail': defaultdict(list),
        'mod_detail': defaultdict(list),
        'n': 0,
    })

    for u in real_words:
        if u['age_months'] is None:
            continue
        age = u['age_months']
        for label, lo, hi in age_bins:
            if lo <= age < hi:
                feat_pho = extract_features(u['xpho'])
                feat_mod = extract_features(u['xmod'])
                detail_pho = extract_consonant_types(u['xpho'])
                detail_mod = extract_consonant_types(u['xmod'])
                for d in dims:
                    if feat_pho[d] is not None:
                        bin_data[label]['pho'][d].append(feat_pho[d])
                    if feat_mod[d] is not None:
                        bin_data[label]['mod'][d].append(feat_mod[d])
                for k, v in detail_pho.items():
                    if v is not None:
                        bin_data[label]['pho_detail'][k].append(v)
                for k, v in detail_mod.items():
                    if v is not None:
                        bin_data[label]['mod_detail'][k].append(v)
                bin_data[label]['n'] += 1
                break

    # Print 4D results
    print(f"\n{'Age':>8} {'N':>7} | {'Dimension':>10} {'Target':>8} {'Actual':>8} {'Δ':>8} {'Dir':>4} | AKS Prediction")
    print('-' * 90)

    aks_predictions = {
        'voicing': ('↑', 'child MORE voiced'),
        'sonority': ('↓', 'child LOWER sonority (more stops)'),
        'place': ('↓', 'child MORE front (labial/alveolar)'),
        'openness': ('↑', 'child MORE open vowels'),
    }

    for label, lo, hi in age_bins:
        r = bin_data[label]
        if r['n'] == 0:
            continue
        first = True
        for d in dims:
            mod_vals = r['mod'][d]
            pho_vals = r['pho'][d]
            if not mod_vals or not pho_vals:
                continue
            mod_m = np.mean(mod_vals)
            pho_m = np.mean(pho_vals)
            delta = pho_m - mod_m
            sign = '↑' if delta > 0.005 else ('↓' if delta < -0.005 else '≈')
            pred_dir, pred_desc = aks_predictions[d]
            match = '✓' if sign == pred_dir else ('≈' if sign == '≈' else '✗')

            if first:
                print(f"\n  {label:>6}mo (N={r['n']:>6,})")
                first = False
            print(f"    {d:>10}: target={mod_m:.3f}  actual={pho_m:.3f}  Δ={delta:+.4f} {sign}  {match} {pred_desc}")

    # ================================================================
    # DETAILED CONSONANT ANALYSIS
    # ================================================================
    print(f"\n\n{'='*90}")
    print("DETAILED CONSONANT ANALYSIS: Actual vs Target")
    print(f"{'='*90}")

    detail_keys = ['stop_pct', 'fric_pct', 'nasal_pct', 'voiced_pct', 'labial_pct', 'alveolar_pct', 'velar_pct']
    detail_labels = {
        'stop_pct': 'Stop %',
        'fric_pct': 'Fricative %',
        'nasal_pct': 'Nasal %',
        'voiced_pct': 'Voiced %',
        'labial_pct': 'Labial %',
        'alveolar_pct': 'Alveolar %',
        'velar_pct': 'Velar %',
    }

    print(f"\n{'Age':>8} | ", end='')
    for k in detail_keys:
        print(f"  {detail_labels[k]:>12}", end='')
    print()
    print('-' * 110)

    for label, lo, hi in age_bins:
        r = bin_data[label]
        if r['n'] == 0:
            continue
        print(f"\n  {label:>6}  Target |", end='')
        for k in detail_keys:
            vals = r['mod_detail'][k]
            print(f"  {np.mean(vals)*100:>10.1f}%", end='') if vals else print(f"  {'N/A':>11}", end='')
        print()
        print(f"  {'':>6}  Actual |", end='')
        for k in detail_keys:
            vals = r['pho_detail'][k]
            print(f"  {np.mean(vals)*100:>10.1f}%", end='') if vals else print(f"  {'N/A':>11}", end='')
        print()
        print(f"  {'':>6}  Delta  |", end='')
        for k in detail_keys:
            mod_vals = r['mod_detail'][k]
            pho_vals = r['pho_detail'][k]
            if mod_vals and pho_vals:
                d = (np.mean(pho_vals) - np.mean(mod_vals)) * 100
                sign = '+' if d > 0 else ''
                print(f"  {sign}{d:>9.1f}%", end='')
            else:
                print(f"  {'N/A':>11}", end='')
        print()

    # ================================================================
    # BABBLING ANALYSIS BY AGE
    # ================================================================
    print(f"\n\n{'='*90}")
    print("BABBLING ANALYSIS (yyy/*): Articulatory Features by Age")
    print(f"{'='*90}")

    for label, lo, hi in age_bins:
        bab_in_range = [u for u in babbling if u['age_months'] is not None and lo <= u['age_months'] < hi]
        if not bab_in_range:
            continue

        v_list, s_list, p_list, o_list = [], [], [], []
        for u in bab_in_range:
            f = extract_features(u['xpho'])
            if f['voicing'] is not None: v_list.append(f['voicing'])
            if f['sonority'] is not None: s_list.append(f['sonority'])
            if f['place'] is not None: p_list.append(f['place'])
            if f['openness'] is not None: o_list.append(f['openness'])

        print(f"\n  {label}mo (N={len(bab_in_range):,} babbling utterances)")
        if v_list: print(f"    Voicing:  {np.mean(v_list):.3f}  (1.0 = all voiced)")
        if s_list: print(f"    Sonority: {np.mean(s_list):.3f}  (1=stops, 7=approx)")
        if p_list: print(f"    Place:    {np.mean(p_list):.3f}  (1=labial, 4=velar)")
        if o_list: print(f"    Openness: {np.mean(o_list):.3f}  (1=close, 4=open)")

    # ================================================================
    # SAMPLE PAIRS: Most interesting deviations
    # ================================================================
    print(f"\n\n{'='*90}")
    print("SAMPLE: Largest articulatory deviations (target vs actual)")
    print(f"{'='*90}")

    deviations = []
    for u in real_words[:10000]:
        if u['xmod'] and u['xpho'] and u['age_months']:
            f_mod = extract_features(u['xmod'])
            f_pho = extract_features(u['xpho'])
            total_dev = 0
            n_dims = 0
            for d in dims:
                if f_mod[d] is not None and f_pho[d] is not None:
                    total_dev += abs(f_pho[d] - f_mod[d])
                    n_dims += 1
            if n_dims > 0:
                deviations.append((total_dev / n_dims, u))

    deviations.sort(key=lambda x: -x[0])
    print(f"\nTop 15 deviations:")
    for dev, u in deviations[:15]:
        print(f"  [{u['child']} {u['age_months']:.0f}mo] \"{u['text'][:40]}\"")
        print(f"    Target: {u['xmod']}")
        print(f"    Actual: {u['xpho']}")
        print(f"    Deviation: {dev:.3f}")
        print()


if __name__ == '__main__':
    main()
