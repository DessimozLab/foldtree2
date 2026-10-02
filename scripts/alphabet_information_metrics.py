"""Numerical methods extracted from alphabet_Information_content_benchmark.ipynb.

Generated once for script use; importing this module never executes a notebook.
"""
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm

def jaccard_similarity(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def weighted_jaccard_similarity(a_counts, b_counts):
    all_keys = set(a_counts) | set(b_counts)
    if not all_keys:
        return 0.0
    num = sum(min(a_counts.get(k, 0), b_counts.get(k, 0)) for k in all_keys)
    den = sum(max(a_counts.get(k, 0), b_counts.get(k, 0)) for k in all_keys)
    return num / den if den > 0 else 0.0

def build_kmers(seq, k):
    if len(seq) < k:
        return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def make_stratified_folds(indices_by_family, n_folds, rng):
    """
    Create stratified folds over indices, preserving family proportions as much as possible.
    Returns: list of fold index-lists (each fold is a list of global indices).
    """
    # shuffle indices within each family
    fam_to_indices = {fam: idxs.copy() for fam, idxs in indices_by_family.items()}
    for fam in fam_to_indices:
        rng.shuffle(fam_to_indices[fam])

    folds = [[] for _ in range(n_folds)]
    # round-robin assign examples from each family into folds
    for fam, idxs in fam_to_indices.items():
        for i, idx in enumerate(idxs):
            folds[i % n_folds].append(idx)

    # shuffle within folds (optional)
    for f in folds:
        rng.shuffle(f)

    return folds

def run_kmer_fold_discrimination(all_results_df, n_families=100, n_folds=5, seed=42, use_weighted=False, supports=None):
    family_results = {'k_mer_discrimination': {}}
    # Sample families
    unique_families = np.array(sorted(all_results_df['family'].unique()))
    n_families = min(n_families, len(unique_families))
    rng = np.random.default_rng(seed)
    sampled_families = rng.choice(unique_families, size=n_families, replace=False)

    family_results['sampled_families'] = list(sampled_families)

    # Filter all_results_df for sampled families
    family_df = all_results_df[all_results_df['family'].isin(sampled_families)].copy()

    print(f"   Sampled {len(sampled_families)} families.")

    # Group by model
    for model_name in family_df['models'].unique():
        model_df = family_df[family_df['models'] == model_name].reset_index(drop=True)
        if 'id' in model_df:
            model_df = model_df.sort_values(['family', 'id']).reset_index(drop=True)
        if len(model_df) == 0:
            continue

        sequences = model_df['seq'].tolist()
        families = model_df['family'].tolist()

        print(f"   {model_name}:")

        # Build stratified folds over sequences (by family)
        indices_by_family = defaultdict(list)
        for idx, fam in enumerate(families):
            indices_by_family[fam].append(idx)

        # If some families have 1 sequence, stratified K-fold still works (some folds won't contain them)
        cur_n_folds = min(n_folds, len(model_df))
        if cur_n_folds < 2:
            print("      (Skipping: not enough sequences for CV)")
            continue

        folds = make_stratified_folds(indices_by_family, n_folds=cur_n_folds, rng=np.random.default_rng(seed))

        k_mer_results = {}
        for k in [1, 2, 3, 4]:
            fold_accuracies = []

            # Precompute per-sequence k-mer sets/counts once (speed)
            seq_kmer_sets = []
            seq_kmer_counts = []
            for seq in sequences:
                km = build_kmers(seq, k)
                if supports is not None:
                    km = [word for word in km if all(c in supports[model_name] for c in word)]
                seq_kmer_sets.append(set(km))
                seq_kmer_counts.append(Counter(km))

            for fold_idx in range(cur_n_folds):
                test_indices = set(folds[fold_idx])
                train_indices = [i for i in range(len(model_df)) if i not in test_indices]

                # Build family prototypes from TRAIN ONLY
                family_kmer_sets = {fam: set() for fam in sampled_families}
                family_kmer_counts = {fam: Counter() for fam in sampled_families}

                for i in train_indices:
                    fam = families[i]
                    # only build prototypes for sampled families
                    if fam not in family_kmer_sets:
                        continue
                    family_kmer_sets[fam].update(seq_kmer_sets[i])
                    family_kmer_counts[fam].update(seq_kmer_counts[i])

                # Classify TEST sequences
                correct = 0
                total = 0
                for i in test_indices:
                    true_fam = families[i]
                    # only evaluate sampled families
                    if true_fam not in family_kmer_sets:
                        continue

                    best_fam = None
                    best_score = -1.0

                    for fam in sampled_families:
                        if use_weighted:
                            score = weighted_jaccard_similarity(seq_kmer_counts[i], family_kmer_counts[fam])
                        else:
                            score = jaccard_similarity(seq_kmer_sets[i], family_kmer_sets[fam])

                        if score > best_score:
                            best_score = score
                            best_fam = fam

                    if best_fam == true_fam:
                        correct += 1
                    total += 1

                if total > 0:
                    fold_accuracies.append(correct / total)

            # Summarize across folds
            if fold_accuracies:
                mean_acc = float(np.mean(fold_accuracies))
                std_acc = float(np.std(fold_accuracies, ddof=1)) if len(fold_accuracies) > 1 else 0.0
                sem_acc = float(std_acc / np.sqrt(len(fold_accuracies))) if len(fold_accuracies) > 1 else 0.0
            else:
                mean_acc, sem_acc = 0.0, 0.0

            k_mer_results[f'k={k}'] = {
                'accuracy_mean': mean_acc,
                'accuracy_sem': sem_acc,
                'fold_accuracies': fold_accuracies,
                'n_sequences': len(sequences),
                'n_families': len(sampled_families),
                'n_folds': cur_n_folds
            }
            print(f"      k={k}: accuracy={mean_acc:.3f} ± {sem_acc:.3f} (SEM over {cur_n_folds} folds)")

        family_results['k_mer_discrimination'][model_name] = k_mer_results

    return family_results

def train_markov_counts(train_seqs, alphabet, char_position_map, max_order):
    counts0 = Counter()
    context_counts = [None] + [Counter() for _ in range(max_order)]  # 1..max_order
    trans_counts   = [None] + [Counter() for _ in range(max_order)]  # 1..max_order

    for seq in train_seqs:
        if not seq:
            continue

        counts0.update(c for c in seq if c in char_position_map)

        for k in range(1, max_order + 1):
            if len(seq) <= k:
                continue
            for i in range(len(seq) - k):
                ctx = tuple(seq[i:i+k])
                nxt = seq[i+k]
                if nxt not in char_position_map or any(c not in char_position_map for c in ctx):
                    continue
                context_counts[k][ctx] += 1
                trans_counts[k][(ctx, nxt)] += 1

    return counts0, context_counts, trans_counts

def make_prob_fn(counts0, context_counts, trans_counts, alphabet, alpha0=0.5, alpha_backoff=5.0):
    A = len(alphabet)
    total0 = sum(counts0.values())
    denom0 = total0 + alpha0 * A
    p0 = {ch: (counts0.get(ch, 0) + alpha0) / denom0 for ch in alphabet}

    def p_sym(ctx_tuple, ch):
        k = len(ctx_tuple)
        if k == 0:
            return p0[ch]
        backoff_ctx = ctx_tuple[1:]
        p_back = p_sym(backoff_ctx, ch)
        ctx_total = context_counts[k].get(ctx_tuple, 0)
        trans = trans_counts[k].get((ctx_tuple, ch), 0)
        alpha = alpha_backoff
        return (trans + alpha * p_back) / (ctx_total + alpha)

    return p_sym

def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        means[b] = np.mean(rng.choice(values, size=n, replace=True))
    alpha = (1.0 - ci) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))

def mean_sem(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, 0)
    m = float(np.mean(values))
    if n == 1:
        return (m, 0.0, 1)
    s = float(np.std(values, ddof=1))
    return (m, s / np.sqrt(n), n)

def compute_henikoff_weights(msa: np.ndarray, valid_symbols: set, gap_char: str = '-') -> np.ndarray:
    """
    Additive Henikoff weights on an MSA (usually AA MSA). Only counts valid_symbols (excluding gap).
    """
    n_seqs, L = msa.shape
    w = np.zeros(n_seqs, dtype=float)

    sym_list = list(valid_symbols)
    for j in range(L):
        col = msa[:, j]
        mask_valid = (col != gap_char) & np.isin(col, sym_list)
        if not np.any(mask_valid):
            continue
        col_valid = col[mask_valid]
        syms, counts = np.unique(col_valid, return_counts=True)
        r_j = len(syms)
        if r_j == 0:
            continue
        count_map = dict(zip(syms, counts))
        for sym in syms:
            n_a = count_map[sym]
            if n_a > 0:
                w[col == sym] += 1.0 / (r_j * n_a)

    if w.sum() > 0:
        w /= w.sum()
    else:
        w[:] = 1.0 / n_seqs
    return w

def weighted_column_entropy(col: np.ndarray, weights: np.ndarray, alphabet: list, gap_char: str = '-') -> float:
    """
    Weighted Shannon entropy over the provided alphabet only (gap excluded).
    Assumes weights sum to 1.
    """
    H = 0.0
    for sym in alphabet:
        p = weights[col == sym].sum()
        if p > 0:
            H -= p * np.log2(p)
    return float(H)

def shannon_entropy_from_counts_fixed_support(counts, states, alpha=1e-3):
    if not states:
        return 0.0
    N = sum(counts.values())
    n = len(states)
    den = N + alpha * n
    H = 0.0
    for x in states:
        p = (counts.get(x, 0) + alpha) / den
        H -= p * safe_log2(p)
    return float(H)

def mi_from_joint_fixed_support(joint_counts, X_states, Y_states, alpha=1e-3):
    if not X_states or not Y_states:
        return 0.0
    N = sum(joint_counts.values())
    nx = len(X_states)
    ny = len(Y_states)
    den = N + alpha * nx * ny

    p_x = {x: 0.0 for x in X_states}
    p_y = {y: 0.0 for y in Y_states}

    for x in X_states:
        for y in Y_states:
            p = (joint_counts.get((x, y), 0) + alpha) / den
            p_x[x] += p
            p_y[y] += p

    mi = 0.0
    for x in X_states:
        for y in Y_states:
            p_xy = (joint_counts.get((x, y), 0) + alpha) / den
            mi += p_xy * (safe_log2(p_xy) - safe_log2(p_x[x]) - safe_log2(p_y[y]))
    return float(mi)

def entropy_from_weighted_freqs(freqs: dict):
    total = sum(freqs.values())
    if total <= 0:
        return 0.0
    H = 0.0
    for p in freqs.values():
        if p > 0:
            pn = p / total
            H -= pn * np.log2(pn)
    return float(H)

def project_ft2_onto_aa_gaps(aa_aligned: str, ft2_ungapped: str, gap_char: str = '-') -> str:
    out = []
    k = 0
    for ch in aa_aligned:
        if ch == gap_char:
            out.append(gap_char)
        else:
            if k >= len(ft2_ungapped):
                out.append(gap_char)
            else:
                out.append(ft2_ungapped[k])
                k += 1
    return ''.join(out)

def bootstrap_mean_ci(values, n_boot=10000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return (np.nan, np.nan, np.nan)

    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idxs = rng.integers(0, n, size=n)
        boot_means[b] = float(np.mean(values[idxs]))

    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1 - alpha))
    return float(np.mean(values)), lo, hi

def train_ngram_model(seqs, alphabet, order, alpha=1e-3):
    """
    Returns dict of probabilities for:
      order=0: P(x)
      order>0: P(x | context)
    Also returns some counts useful for optional model-cost proxy.
    """
    alphabet = list(alphabet)

    if order == 0:
        counts = Counter()
        for s in seqs:
            for ch in s:
                if ch in alphabet:
                    counts[ch] += 1
        total = sum(counts.values())
        den = total + alpha * len(alphabet)
        probs = {ch: (counts.get(ch, 0) + alpha) / den for ch in alphabet}
        return probs, {"unigram_counts": counts, "total": total}

    # order > 0
    context_counts = Counter()
    trans_counts = Counter()

    for s in seqs:
        if len(s) <= order:
            continue
        for i in range(order, len(s)):
            ctx = s[i-order:i]
            ch = s[i]
            if all(c in alphabet for c in ctx) and ch in alphabet:
                context_counts[ctx] += 1
                trans_counts[(ctx, ch)] += 1

    # Conditional probs for contexts seen in training; for unseen contexts we backoff to uniform
    probs = {}
    for ctx, ctx_total in context_counts.items():
        den = ctx_total + alpha * len(alphabet)
        for ch in alphabet:
            probs[(ctx, ch)] = (trans_counts.get((ctx, ch), 0) + alpha) / den

    return probs, {"context_counts": context_counts, "trans_counts": trans_counts}

def encode_bits(seqs, alphabet, order, probs, alpha=1e-3):
    """
    Compute total codelength (bits) on seqs under the provided model probs.
    Uses:
      order=0: probs[ch]
      order>0: probs[(ctx,ch)] if ctx seen, otherwise uniform 1/|alphabet|
    Returns: (total_bits, n_tokens_encoded, bits_per_token)
    """
    alphabet = list(alphabet)
    A = len(alphabet)
    uniform_p = 1.0 / A if A > 0 else 1.0

    total_bits = 0.0
    n = 0

    for s in seqs:
        if len(s) == 0:
            continue
        start = 0 if order == 0 else order
        for i in range(start, len(s)):
            ch = s[i]
            if ch not in alphabet:
                continue

            if order == 0:
                p = probs.get(ch, uniform_p)
            else:
                ctx = s[i-order:i]
                if any(c not in alphabet for c in ctx):
                    continue
                p = probs.get((ctx, ch), None)
                if p is None:
                    # unseen context: use uniform over alphabet
                    p = uniform_p

            # numerical safety
            p = max(p, 1e-12)
            total_bits += -np.log2(p)
            n += 1

    bpt = total_bits / n if n > 0 else np.nan
    return float(total_bits), int(n), float(bpt)

def model_cost_proxy(alphabet, order, train_stats, n_train_tokens):
    """
    Crude MDL-ish header cost:
      ~ (#free parameters) * log2(n_train_tokens)
    where #free parameters ~ (#contexts)*(A-1) for order>0, and (A-1) for order=0.
    This is not rigorous MDL, but gives a consistent penalty as order grows.
    """
    A = len(alphabet)
    if A <= 1 or n_train_tokens <= 1:
        return 0.0

    if order == 0:
        n_params = max(A - 1, 0)
    else:
        n_contexts = len(train_stats.get("context_counts", {}))
        n_params = n_contexts * max(A - 1, 0)

    return float(n_params * np.log2(n_train_tokens))

def safe_log2(x):
    return np.log2(x)

def compute_positionwise_stats_fixed_support(ft2_seqs, aa_seqs, FT2_STATES, AA_STATES, FT2_BIGRAM_STATES, alpha=1e-3):
    aa_counts = Counter()
    ft2_counts = Counter()
    joint_1 = Counter()
    N1 = 0

    ft2_bigram_counts = Counter()
    joint_2 = Counter()
    N2 = 0

    for s_ft2, s_aa in zip(ft2_seqs, aa_seqs):
        L = min(len(s_ft2), len(s_aa))
        if L < 2:
            continue

        for t in range(L):
            x = s_ft2[t]
            y = s_aa[t]
            if x in FT2_STATES and y in AA_STATES:
                ft2_counts[x] += 1
                aa_counts[y] += 1
                joint_1[(x, y)] += 1
                N1 += 1

        for t in range(1, L):
            x_prev = s_ft2[t-1]
            x_cur = s_ft2[t]
            y = s_aa[t]
            x2 = (x_prev, x_cur)
            if (x_prev in FT2_STATES) and (x_cur in FT2_STATES) and (y in AA_STATES):
                ft2_bigram_counts[x2] += 1
                joint_2[(x2, y)] += 1
                N2 += 1

    H_aa = shannon_entropy_from_counts_fixed_support(aa_counts, AA_STATES, alpha=alpha)
    H_ft2 = shannon_entropy_from_counts_fixed_support(ft2_counts, FT2_STATES, alpha=alpha)
    H_ft2_bigram = shannon_entropy_from_counts_fixed_support(ft2_bigram_counts, FT2_BIGRAM_STATES, alpha=alpha)

    MI1 = mi_from_joint_fixed_support(joint_1, FT2_STATES, AA_STATES, alpha=alpha)
    MI2 = mi_from_joint_fixed_support(joint_2, FT2_BIGRAM_STATES, AA_STATES, alpha=alpha)
    CMI = MI2 - MI1

    return {
        'H_aa_bits': H_aa,
        'H_ft2_bits': H_ft2,
        'H_ft2_bigram_bits': H_ft2_bigram,
        'mi1_bits': MI1,
        'mi2_bits': MI2,
        'cmi_bits': float(CMI),
        'n_samples_t': int(N1),
        'n_samples_tminus1': int(N2),
        'n_ft2_states': int(len(FT2_STATES)),
        'n_ft2_bigram_states': int(len(FT2_BIGRAM_STATES)),
        'n_aa_states': int(len(AA_STATES)),
    }

def compute_pairwise_stats_fixed_support(ft2_seqs, aa_seqs, FT2_STATES, AA_STATES, FT2_BIGRAM_STATES, alpha=1e-3):
    """
    Per-pair: H(AA), MI1, MI2, CMI + normalized variants:
      nmi1 = MI1/H(AA)
      nmi2 = MI2/H(AA)
      ncmi = CMI/H(AA)
    """
    Haa_list, mi1_list, mi2_list, cmi_list, L_list = [], [], [], [], []
    nmi1_list, nmi2_list, ncmi_list = [], [], []

    for s_ft2, s_aa in zip(ft2_seqs, aa_seqs):
        L = min(len(s_ft2), len(s_aa))
        if L < 2:
            continue

        aa_counts = Counter()
        joint_1 = Counter()
        joint_2 = Counter()

        for t in range(L):
            x = s_ft2[t]
            y = s_aa[t]
            if x in FT2_STATES and y in AA_STATES:
                aa_counts[y] += 1
                joint_1[(x, y)] += 1

        for t in range(1, L):
            x_prev = s_ft2[t-1]
            x_cur = s_ft2[t]
            y = s_aa[t]
            x2 = (x_prev, x_cur)
            if (x_prev in FT2_STATES) and (x_cur in FT2_STATES) and (y in AA_STATES):
                joint_2[(x2, y)] += 1

        Haa = shannon_entropy_from_counts_fixed_support(aa_counts, AA_STATES, alpha=alpha)
        mi1 = mi_from_joint_fixed_support(joint_1, FT2_STATES, AA_STATES, alpha=alpha)
        mi2 = mi_from_joint_fixed_support(joint_2, FT2_BIGRAM_STATES, AA_STATES, alpha=alpha)
        cmi = mi2 - mi1

        Haa_list.append(Haa)
        mi1_list.append(mi1)
        mi2_list.append(mi2)
        cmi_list.append(cmi)
        L_list.append(L)

        den = Haa if Haa > 0 else np.nan
        nmi1_list.append(mi1 / den)
        nmi2_list.append(mi2 / den)
        ncmi_list.append(cmi / den)

    return (
        np.array(Haa_list),
        np.array(mi1_list),
        np.array(mi2_list),
        np.array(cmi_list),
        np.array(nmi1_list),
        np.array(nmi2_list),
        np.array(ncmi_list),
        np.array(L_list),
    )
