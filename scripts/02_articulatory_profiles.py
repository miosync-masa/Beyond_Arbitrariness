"""
Deep Articulatory Pattern Analysis
====================================
Hypothesis: The shape of the mouth during articulation is systematically
linked to meaning — not arbitrarily, but through body→sound mapping.

We analyze:
1. Vowel height (mouth openness) × meaning (big/small, etc.)
2. Vowel frontness (tongue position) × meaning categories
3. Consonant manner (stop/fricative/nasal/liquid) × meaning categories
4. Voicing (voiced/voiceless) × meaning categories
5. Full articulatory profile per concept
"""

import csv
from collections import Counter, defaultdict
from math import log2, sqrt
import json

# ===== ARTICULATORY FEATURE SYSTEM =====

# Vowel features (ASJPcode)
VOWEL_FEATURES = {
    'i': {'height': 'close', 'front': 'front', 'round': False, 'openness': 1},
    'e': {'height': 'close-mid', 'front': 'front', 'round': False, 'openness': 2},
    'E': {'height': 'open-mid', 'front': 'front', 'round': False, 'openness': 3},
    '3': {'height': 'mid', 'front': 'central', 'round': False, 'openness': 2.5},
    'a': {'height': 'open', 'front': 'central', 'round': False, 'openness': 4},
    'u': {'height': 'close', 'front': 'back', 'round': True, 'openness': 1},
    'o': {'height': 'close-mid', 'front': 'back', 'round': True, 'openness': 2},
}

# Consonant features (ASJPcode)
CONSONANT_FEATURES = {
    'p': {'place': 'labial', 'manner': 'stop', 'voiced': False, 'sonority': 1},
    'b': {'place': 'labial', 'manner': 'stop', 'voiced': True, 'sonority': 2},
    'f': {'place': 'labial', 'manner': 'fricative', 'voiced': False, 'sonority': 3},
    'v': {'place': 'labial', 'manner': 'fricative', 'voiced': True, 'sonority': 4},
    'm': {'place': 'labial', 'manner': 'nasal', 'voiced': True, 'sonority': 6},
    'w': {'place': 'labial', 'manner': 'approximant', 'voiced': True, 'sonority': 7},
    '8': {'place': 'dental', 'manner': 'fricative', 'voiced': False, 'sonority': 3},
    't': {'place': 'alveolar', 'manner': 'stop', 'voiced': False, 'sonority': 1},
    'd': {'place': 'alveolar', 'manner': 'stop', 'voiced': True, 'sonority': 2},
    's': {'place': 'alveolar', 'manner': 'fricative', 'voiced': False, 'sonority': 3},
    'z': {'place': 'alveolar', 'manner': 'fricative', 'voiced': True, 'sonority': 4},
    'c': {'place': 'alveolar', 'manner': 'affricate', 'voiced': False, 'sonority': 2},
    'n': {'place': 'alveolar', 'manner': 'nasal', 'voiced': True, 'sonority': 6},
    'r': {'place': 'alveolar', 'manner': 'trill', 'voiced': True, 'sonority': 7},
    'l': {'place': 'alveolar', 'manner': 'lateral', 'voiced': True, 'sonority': 7},
    'S': {'place': 'postalveolar', 'manner': 'fricative', 'voiced': False, 'sonority': 3},
    'Z': {'place': 'postalveolar', 'manner': 'fricative', 'voiced': True, 'sonority': 4},
    'C': {'place': 'postalveolar', 'manner': 'affricate', 'voiced': False, 'sonority': 2},
    'j': {'place': 'postalveolar', 'manner': 'affricate', 'voiced': True, 'sonority': 3},
    'T': {'place': 'retroflex', 'manner': 'stop', 'voiced': False, 'sonority': 1},
    '5': {'place': 'palatal', 'manner': 'nasal', 'voiced': True, 'sonority': 6},
    'y': {'place': 'palatal', 'manner': 'approximant', 'voiced': True, 'sonority': 7},
    'k': {'place': 'velar', 'manner': 'stop', 'voiced': False, 'sonority': 1},
    'g': {'place': 'velar', 'manner': 'stop', 'voiced': True, 'sonority': 2},
    'x': {'place': 'velar', 'manner': 'fricative', 'voiced': False, 'sonority': 3},
    'N': {'place': 'velar', 'manner': 'nasal', 'voiced': True, 'sonority': 6},
    'q': {'place': 'uvular', 'manner': 'stop', 'voiced': False, 'sonority': 1},
    'X': {'place': 'uvular', 'manner': 'fricative', 'voiced': False, 'sonority': 3},
    'h': {'place': 'glottal', 'manner': 'fricative', 'voiced': False, 'sonority': 3},
    '7': {'place': 'glottal', 'manner': 'stop', 'voiced': False, 'sonority': 1},
    'L': {'place': 'alveolar', 'manner': 'lateral_fricative', 'voiced': False, 'sonority': 4},
    '4': {'place': 'alveolar', 'manner': 'flap', 'voiced': True, 'sonority': 6},
    'G': {'place': 'velar', 'manner': 'fricative', 'voiced': True, 'sonority': 4},
    '!': {'place': 'various', 'manner': 'click', 'voiced': False, 'sonority': 1},
}

ASJP_VOWELS = set(VOWEL_FEATURES.keys())
ASJP_CONSONANTS = set(CONSONANT_FEATURES.keys())

# Semantic categories for concepts
SEMANTIC_CATEGORIES = {
    # Size/dimension
    'SIZE_BIG': ['13'],       # BIG
    'SIZE_SMALL': ['15'],     # SMALL
    'SIZE_LONG': ['14'],      # LONG
    
    # Body - oral
    'BODY_ORAL': ['42', '44', '43'],  # MOUTH, TONGUE, TOOTH
    'BODY_NOSE': ['41'],              # NOSE
    'BODY_EAR': ['39'],               # EAR
    'BODY_EYE': ['40'],               # EYE
    'BODY_HEAD': ['38'],              # HEAD
    'BODY_LIMB': ['48', '46', '47'],  # HAND, FOOT, KNEE
    'BODY_TORSO': ['49', '50', '51', '52', '53'],  # BELLY, NECK, BREAST, HEART, LIVER
    'BODY_SURFACE': ['28', '37'],     # SKIN, HAIR
    'BODY_INTERNAL': ['30', '31', '29', '32'],  # BLOOD, BONE, FLESH, FAT
    
    # Life/Death
    'LIFE_DIE': ['61'],       # DIE
    'LIFE_KILL': ['62'],      # KILL
    
    # Movement
    'MOVE_ACTIVE': ['63', '64', '65', '66'],   # SWIM, FLY, WALK, COME
    'MOVE_STATIC': ['67', '68', '69'],         # LIE, SIT, STAND
    
    # Perception
    'SENSE': ['57', '58', '59'],  # SEE, HEAR, KNOW
    
    # Ingestion
    'INGEST': ['54', '55', '56'],  # DRINK, EAT, BITE
    
    # Nature - elements
    'NATURE_WATER': ['75', '76'],     # WATER, RAIN
    'NATURE_FIRE': ['82', '84'],      # FIRE, BURN
    'NATURE_EARTH': ['77', '78', '79'],  # STONE, SAND, EARTH
    'NATURE_SKY': ['72', '73', '74', '80'],  # SUN, MOON, STAR, CLOUD
    'NATURE_OTHER': ['81', '83'],     # SMOKE, ASH
    
    # Colors
    'COLOR_WARM': ['87'],         # RED
    'COLOR_COOL': ['88'],         # GREEN
    'COLOR_LIGHT': ['89', '90'],  # YELLOW, WHITE
    'COLOR_DARK': ['91'],         # BLACK
    
    # Temperature
    'TEMP_HOT': ['93'],      # HOT
    'TEMP_COLD': ['94'],     # COLD
    
    # Pronouns
    'PRONOUN_1': ['1'],      # I
    'PRONOUN_2': ['2'],      # THOU
    'PRONOUN_WE': ['3'],     # WE
    
    # Quantity
    'QUANTITY': ['9', '10', '11', '12'],  # ALL, MANY, ONE, TWO
    
    # Living things
    'LIVING': ['16', '17', '18', '19', '20', '21', '22'],  # WOMAN, MAN, PERSON, FISH, BIRD, DOG, LOUSE
    
    # Nature features
    'NATURE_FEAT': ['23', '24', '25', '26', '27'],  # TREE, SEED, LEAF, ROOT, BARK
}


def load_data():
    """Load ASJP data."""
    languages = {}
    with open('/content/asjp/cldf/languages.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            languages[row['ID']] = {'family': row['Family'], 'lat': float(row['Latitude']) if row['Latitude'] else None}
    
    concepts = {}
    with open('/content/asjp/cldf/parameters.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            concepts[row['ID']] = row['Concepticon_Gloss']
    
    forms = []
    with open('/content/asjp/cldf/forms.csv', 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['Loan'] != 'true':
                forms.append({'lang_id': row['Language_ID'], 'concept_id': row['Parameter_ID'], 'form': row['Form']})
    
    return languages, concepts, forms


def extract_phonemes(form):
    """Extract phonemes from ASJPcode."""
    return [ch for ch in form if ch in ASJP_VOWELS or ch in ASJP_CONSONANTS]


def compute_features_per_concept(forms, languages):
    """Compute articulatory feature distributions for each concept."""
    concept_data = defaultdict(lambda: {
        'vowel_height': [], 'vowel_front': [], 'vowel_round': [],
        'cons_place': [], 'cons_manner': [], 'cons_voiced': [],
        'mean_openness': [], 'mean_sonority': [],
        'initial_phonemes': [], 'all_phonemes': [],
        'n_words': 0, 'families': set(),
    })
    
    for entry in forms:
        lang = languages.get(entry['lang_id'])
        if not lang:
            continue
        
        phonemes = extract_phonemes(entry['form'])
        if not phonemes:
            continue
        
        cd = concept_data[entry['concept_id']]
        cd['n_words'] += 1
        if lang['family']:
            cd['families'].add(lang['family'])
        cd['initial_phonemes'].append(phonemes[0])
        cd['all_phonemes'].extend(phonemes)
        
        openness_vals = []
        sonority_vals = []
        
        for ph in phonemes:
            if ph in VOWEL_FEATURES:
                vf = VOWEL_FEATURES[ph]
                cd['vowel_height'].append(vf['height'])
                cd['vowel_front'].append(vf['front'])
                cd['vowel_round'].append(vf['round'])
                openness_vals.append(vf['openness'])
            elif ph in CONSONANT_FEATURES:
                cf = CONSONANT_FEATURES[ph]
                cd['cons_place'].append(cf['place'])
                cd['cons_manner'].append(cf['manner'])
                cd['cons_voiced'].append(cf['voiced'])
                sonority_vals.append(cf['sonority'])
        
        if openness_vals:
            cd['mean_openness'].append(sum(openness_vals) / len(openness_vals))
        if sonority_vals:
            cd['mean_sonority'].append(sum(sonority_vals) / len(sonority_vals))
    
    return concept_data


def compute_global_baselines(concept_data):
    """Compute global baseline distributions."""
    all_vowel_height = []
    all_cons_place = []
    all_cons_manner = []
    all_cons_voiced = []
    all_openness = []
    all_sonority = []
    
    for cd in concept_data.values():
        all_vowel_height.extend(cd['vowel_height'])
        all_cons_place.extend(cd['cons_place'])
        all_cons_manner.extend(cd['cons_manner'])
        all_cons_voiced.extend(cd['cons_voiced'])
        all_openness.extend(cd['mean_openness'])
        all_sonority.extend(cd['mean_sonority'])
    
    def to_dist(lst):
        c = Counter(lst)
        total = sum(c.values())
        return {k: v/total for k, v in c.items()}
    
    return {
        'vowel_height': to_dist(all_vowel_height),
        'cons_place': to_dist(all_cons_place),
        'cons_manner': to_dist(all_cons_manner),
        'voiced_ratio': sum(1 for v in all_cons_voiced if v) / len(all_cons_voiced) if all_cons_voiced else 0,
        'mean_openness': sum(all_openness) / len(all_openness) if all_openness else 0,
        'mean_sonority': sum(all_sonority) / len(all_sonority) if all_sonority else 0,
    }


def analyze_mouth_shape(concept_data, concepts, baselines):
    """Analyze mouth shape patterns for all concepts."""
    print("\n" + "="*80)
    print("ANALYSIS 1: MOUTH OPENNESS (vowel height) × Concept")
    print("="*80)
    print("Hypothesis: 'big' concepts → open mouth (a, o), 'small' concepts → closed mouth (i, u)")
    print(f"Global mean openness: {baselines['mean_openness']:.3f} (1=close, 4=open)\n")
    
    results = []
    for cid, cd in concept_data.items():
        if cd['mean_openness'] and len(cd['mean_openness']) > 100:
            mean_op = sum(cd['mean_openness']) / len(cd['mean_openness'])
            delta = mean_op - baselines['mean_openness']
            name = concepts.get(cid, cid)
            results.append((name, cid, mean_op, delta, len(cd['mean_openness'])))
    
    results.sort(key=lambda x: -x[2])
    
    print(f"{'Concept':<25s} {'Mean Openness':>14s} {'Δ from baseline':>16s} {'n':>8s}  Interpretation")
    print("-" * 95)
    
    for name, cid, mean_op, delta, n in results[:15]:
        interp = ""
        if delta > 0.05: interp = "← OPEN MOUTH 👄"
        elif delta < -0.05: interp = "← CLOSED MOUTH 🤐"
        print(f"  {name:<23s} {mean_op:>14.3f} {delta:>+16.3f} {n:>8d}  {interp}")
    
    print("  ...")
    for name, cid, mean_op, delta, n in results[-15:]:
        interp = ""
        if delta > 0.05: interp = "← OPEN MOUTH 👄"
        elif delta < -0.05: interp = "← CLOSED MOUTH 🤐"
        print(f"  {name:<23s} {mean_op:>14.3f} {delta:>+16.3f} {n:>8d}  {interp}")
    
    return results


def analyze_consonant_manner(concept_data, concepts, baselines):
    """Analyze consonant manner of articulation patterns."""
    print("\n" + "="*80)
    print("ANALYSIS 2: CONSONANT MANNER × Concept")
    print("="*80)
    print("stop=破裂音(p,t,k), fricative=摩擦音(s,f,h), nasal=鼻音(m,n,N), liquid=流音(l,r)")
    print(f"Global manner dist: {dict(sorted(baselines['cons_manner'].items(), key=lambda x: -x[1]))}\n")
    
    results = []
    for cid, cd in concept_data.items():
        if len(cd['cons_manner']) < 200:
            continue
        name = concepts.get(cid, cid)
        manner_dist = Counter(cd['cons_manner'])
        total = sum(manner_dist.values())
        
        concept_profile = {}
        for manner in ['stop', 'fricative', 'nasal', 'lateral', 'trill', 'approximant', 'affricate']:
            obs = manner_dist.get(manner, 0) / total
            exp = baselines['cons_manner'].get(manner, 0.001)
            if exp > 0 and obs > 0:
                lr = log2(obs / exp)
            else:
                lr = 0
            concept_profile[manner] = {'obs': obs, 'exp': exp, 'lr': lr}
        
        # Find strongest deviation
        strongest = max(concept_profile.items(), key=lambda x: abs(x[1]['lr']))
        results.append((name, cid, concept_profile, strongest, total))
    
    results.sort(key=lambda x: -abs(x[3][1]['lr']))
    
    print(f"{'Concept':<20s} {'Strongest':>12s} {'Obs%':>6s} {'Exp%':>6s} {'Log₂':>7s}  {'nasal%':>7s} {'stop%':>7s} {'fric%':>7s} {'liquid%':>8s}")
    print("-" * 100)
    
    for name, cid, profile, (strongest_manner, strongest_data), n in results[:30]:
        nasal_pct = (profile.get('nasal', {}).get('obs', 0)) * 100
        stop_pct = (profile.get('stop', {}).get('obs', 0)) * 100
        fric_pct = (profile.get('fricative', {}).get('obs', 0)) * 100
        liquid_pct = ((profile.get('lateral', {}).get('obs', 0)) + (profile.get('trill', {}).get('obs', 0))) * 100
        
        print(f"  {name:<18s} {strongest_manner:>12s} {strongest_data['obs']*100:>5.1f}% {strongest_data['exp']*100:>5.1f}% {strongest_data['lr']:>+6.2f}  {nasal_pct:>6.1f}% {stop_pct:>6.1f}% {fric_pct:>6.1f}% {liquid_pct:>7.1f}%")


def analyze_voicing(concept_data, concepts, baselines):
    """Analyze voicing patterns."""
    print("\n" + "="*80)
    print("ANALYSIS 3: VOICING × Concept")
    print("="*80)
    print("voiced=声帯振動あり(b,d,g,m,n,l,r) vs voiceless=なし(p,t,k,s,f,h)")
    print(f"Global voiced ratio: {baselines['voiced_ratio']:.3f}\n")
    
    results = []
    for cid, cd in concept_data.items():
        if len(cd['cons_voiced']) < 200:
            continue
        name = concepts.get(cid, cid)
        voiced_count = sum(1 for v in cd['cons_voiced'] if v)
        total = len(cd['cons_voiced'])
        ratio = voiced_count / total
        delta = ratio - baselines['voiced_ratio']
        results.append((name, ratio, delta, total))
    
    results.sort(key=lambda x: -x[1])
    
    print(f"{'Concept':<25s} {'Voiced%':>8s} {'Δ':>8s} {'n':>6s}  Interpretation")
    print("-" * 70)
    
    for name, ratio, delta, n in results[:12]:
        interp = ""
        if delta > 0.02: interp = "← MORE VOICED (soft/resonant)"
        elif delta < -0.02: interp = "← MORE VOICELESS (sharp/hard)"
        print(f"  {name:<23s} {ratio*100:>7.1f}% {delta*100:>+7.1f}% {n:>6d}  {interp}")
    print("  --- baseline ---")
    for name, ratio, delta, n in results[-12:]:
        interp = ""
        if delta > 0.02: interp = "← MORE VOICED (soft/resonant)"
        elif delta < -0.02: interp = "← MORE VOICELESS (sharp/hard)"
        print(f"  {name:<23s} {ratio*100:>7.1f}% {delta*100:>+7.1f}% {n:>6d}  {interp}")


def analyze_sonority(concept_data, concepts, baselines):
    """Analyze sonority patterns."""
    print("\n" + "="*80)
    print("ANALYSIS 4: SONORITY (consonant resonance) × Concept")
    print("="*80)
    print("Low sonority: stops (p,t,k) = hard/abrupt")
    print("High sonority: nasals (m,n), liquids (l,r) = soft/flowing")
    print(f"Global mean sonority: {baselines['mean_sonority']:.3f}\n")
    
    results = []
    for cid, cd in concept_data.items():
        if len(cd['mean_sonority']) < 100:
            continue
        name = concepts.get(cid, cid)
        mean_son = sum(cd['mean_sonority']) / len(cd['mean_sonority'])
        delta = mean_son - baselines['mean_sonority']
        results.append((name, mean_son, delta, len(cd['mean_sonority'])))
    
    results.sort(key=lambda x: -x[1])
    
    print(f"{'Concept':<25s} {'Mean Sonority':>14s} {'Δ':>8s} {'n':>6s}  Interpretation")
    print("-" * 75)
    
    for name, son, delta, n in results[:12]:
        interp = ""
        if delta > 0.15: interp = "← SOFT/FLOWING 〜"
        elif delta < -0.15: interp = "← HARD/ABRUPT !"
        print(f"  {name:<23s} {son:>14.3f} {delta:>+8.3f} {n:>6d}  {interp}")
    print("  --- baseline ---")
    for name, son, delta, n in results[-12:]:
        interp = ""
        if delta > 0.15: interp = "← SOFT/FLOWING 〜"
        elif delta < -0.15: interp = "← HARD/ABRUPT !"
        print(f"  {name:<23s} {son:>14.3f} {delta:>+8.3f} {n:>6d}  {interp}")


def analyze_place_gradient(concept_data, concepts, baselines):
    """Analyze front-to-back articulation gradient."""
    print("\n" + "="*80)
    print("ANALYSIS 5: ARTICULATION PLACE GRADIENT × Concept")
    print("="*80)
    print("Front (lips → tongue tip → palate → throat): labial → alveolar → velar → glottal")
    print("Does 'place in mouth' correlate with meaning categories?\n")
    
    # Assign numeric values to places (front=1, back=5)
    place_values = {
        'labial': 1.0, 'dental': 2.0, 'alveolar': 2.5, 'postalveolar': 3.0,
        'retroflex': 3.2, 'palatal': 3.5, 'velar': 4.0, 'uvular': 4.5, 'glottal': 5.0,
    }
    
    results = []
    for cid, cd in concept_data.items():
        if len(cd['cons_place']) < 200:
            continue
        name = concepts.get(cid, cid)
        
        vals = [place_values.get(p, 3.0) for p in cd['cons_place']]
        mean_place = sum(vals) / len(vals)
        
        # Also compute labial ratio specifically
        place_counter = Counter(cd['cons_place'])
        total = sum(place_counter.values())
        labial_ratio = place_counter.get('labial', 0) / total
        alveolar_ratio = place_counter.get('alveolar', 0) / total
        velar_ratio = place_counter.get('velar', 0) / total
        glottal_ratio = place_counter.get('glottal', 0) / total
        
        results.append((name, mean_place, labial_ratio, alveolar_ratio, velar_ratio, glottal_ratio, total))
    
    # Compute global mean place
    all_places = []
    for cd in concept_data.values():
        all_places.extend([place_values.get(p, 3.0) for p in cd['cons_place']])
    global_mean_place = sum(all_places) / len(all_places)
    
    print(f"Global mean place value: {global_mean_place:.3f} (1=front/labial, 5=back/glottal)\n")
    
    # Sort by mean place (frontmost first)
    results.sort(key=lambda x: x[1])
    
    print(f"{'Concept':<22s} {'MeanPlace':>10s} {'Labial%':>8s} {'Alveol%':>8s} {'Velar%':>8s} {'Glottal%':>9s}  Location")
    print("-" * 90)
    
    for name, mp, lab, alv, vel, glt, n in results[:15]:
        loc = "← FRONT (lips)" if mp < global_mean_place - 0.05 else ""
        print(f"  {name:<20s} {mp:>10.3f} {lab*100:>7.1f}% {alv*100:>7.1f}% {vel*100:>7.1f}% {glt*100:>8.1f}%  {loc}")
    print("  --- baseline ---")
    for name, mp, lab, alv, vel, glt, n in results[-15:]:
        loc = "← BACK (throat)" if mp > global_mean_place + 0.05 else ""
        print(f"  {name:<20s} {mp:>10.3f} {lab*100:>7.1f}% {alv*100:>7.1f}% {vel*100:>7.1f}% {glt*100:>8.1f}%  {loc}")


def analyze_initial_sound_patterns(concept_data, concepts):
    """Analyze word-initial sound patterns (strongest position for sound symbolism)."""
    print("\n" + "="*80)
    print("ANALYSIS 6: WORD-INITIAL SOUND × Concept")
    print("="*80)
    print("The first sound of a word carries the strongest sound-symbolic weight.\n")
    
    # Compute global initial distribution
    all_initials = []
    for cd in concept_data.values():
        all_initials.extend(cd['initial_phonemes'])
    global_init = Counter(all_initials)
    global_total = sum(global_init.values())
    global_dist = {k: v/global_total for k, v in global_init.items()}
    
    # For each concept, find most overrepresented initial
    results = []
    for cid, cd in concept_data.items():
        if cd['n_words'] < 200:
            continue
        name = concepts.get(cid, cid)
        init_counter = Counter(cd['initial_phonemes'])
        init_total = sum(init_counter.values())
        
        best_lr = 0
        best_ph = ''
        top_3 = []
        
        for ph, count in init_counter.most_common():
            obs = count / init_total
            exp = global_dist.get(ph, 0.001)
            if exp > 0 and count > 20:
                lr = log2(obs / exp)
                if abs(lr) > abs(best_lr):
                    best_lr = lr
                    best_ph = ph
                if lr > 0.3:
                    top_3.append((ph, obs, exp, lr, count))
        
        top_3.sort(key=lambda x: -x[3])
        results.append((name, best_ph, best_lr, top_3[:3], init_total))
    
    results.sort(key=lambda x: -abs(x[2]))
    
    print(f"{'Concept':<22s} {'#1 Initial':>11s} {'Log₂':>7s}  Other overrepresented initials")
    print("-" * 85)
    
    for name, best_ph, best_lr, top_3, n in results[:35]:
        ph_type = ""
        if best_ph in VOWEL_FEATURES:
            vf = VOWEL_FEATURES[best_ph]
            ph_type = f"({vf['height']} {vf['front']} vowel)"
        elif best_ph in CONSONANT_FEATURES:
            cf = CONSONANT_FEATURES[best_ph]
            ph_type = f"({cf['place']} {cf['manner']})"
        
        others = ", ".join(f"{ph}({lr:+.2f})" for ph, obs, exp, lr, cnt in top_3[1:])
        print(f"  {name:<20s} /{best_ph}/ {best_lr:>+6.2f}  {ph_type:30s}  {others}")


def semantic_category_profiles(concept_data, concepts, baselines):
    """Compute articulatory profiles for semantic categories."""
    print("\n" + "="*80)
    print("ANALYSIS 7: SEMANTIC CATEGORY ARTICULATORY PROFILES")
    print("="*80)
    print("Aggregating concepts into semantic groups to find category-level patterns.\n")
    
    for cat_name, cid_list in sorted(SEMANTIC_CATEGORIES.items()):
        # Aggregate
        agg_manner = Counter()
        agg_place = Counter()
        agg_openness = []
        agg_sonority = []
        agg_voiced = []
        
        concept_names = []
        for cid in cid_list:
            cd = concept_data.get(cid)
            if not cd:
                continue
            concept_names.append(concepts.get(cid, cid))
            agg_manner.update(cd['cons_manner'])
            agg_place.update(cd['cons_place'])
            agg_openness.extend(cd['mean_openness'])
            agg_sonority.extend(cd['mean_sonority'])
            agg_voiced.extend(cd['cons_voiced'])
        
        if not agg_manner:
            continue
        
        total_manner = sum(agg_manner.values())
        total_place = sum(agg_place.values())
        
        mean_open = sum(agg_openness) / len(agg_openness) if agg_openness else 0
        mean_son = sum(agg_sonority) / len(agg_sonority) if agg_sonority else 0
        voiced_r = sum(1 for v in agg_voiced if v) / len(agg_voiced) if agg_voiced else 0
        
        open_delta = mean_open - baselines['mean_openness']
        son_delta = mean_son - baselines['mean_sonority']
        voiced_delta = voiced_r - baselines['voiced_ratio']
        
        # Top 3 manner
        top_manner = [(m, c/total_manner) for m, c in agg_manner.most_common(3)]
        # Top 3 place  
        top_place = [(p, c/total_place) for p, c in agg_place.most_common(3)]
        
        concepts_str = "+".join(concept_names)
        
        indicators = []
        if open_delta > 0.03: indicators.append("OPEN-MOUTH")
        elif open_delta < -0.03: indicators.append("CLOSED-MOUTH")
        if son_delta > 0.1: indicators.append("SOFT")
        elif son_delta < -0.1: indicators.append("HARD")
        if voiced_delta > 0.015: indicators.append("VOICED")
        elif voiced_delta < -0.015: indicators.append("VOICELESS")
        
        print(f"  [{cat_name}] ({concepts_str})")
        print(f"    Openness: {mean_open:.3f} (Δ{open_delta:+.3f})  Sonority: {mean_son:.3f} (Δ{son_delta:+.3f})  Voiced: {voiced_r:.3f} (Δ{voiced_delta:+.3f})")
        print(f"    Top manner: {', '.join(f'{m}={r:.1%}' for m,r in top_manner)}")
        print(f"    Top place:  {', '.join(f'{p}={r:.1%}' for p,r in top_place)}")
        if indicators:
            print(f"    → {'  '.join(indicators)}")
        print()


def main():
    print("Loading data...")
    languages, concepts, forms = load_data()
    print(f"Loaded: {len(languages)} languages, {len(forms)} forms")
    
    print("\nComputing articulatory features per concept...")
    concept_data = compute_features_per_concept(forms, languages)
    
    print("Computing global baselines...")
    baselines = compute_global_baselines(concept_data)
    
    # Run analyses
    analyze_mouth_shape(concept_data, concepts, baselines)
    analyze_consonant_manner(concept_data, concepts, baselines)
    analyze_voicing(concept_data, concepts, baselines)
    analyze_sonority(concept_data, concepts, baselines)
    analyze_place_gradient(concept_data, concepts, baselines)
    analyze_initial_sound_patterns(concept_data, concepts)
    semantic_category_profiles(concept_data, concepts, baselines)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
