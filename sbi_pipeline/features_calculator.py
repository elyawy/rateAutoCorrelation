"""
Shannon Entropy and Parsimony calculation for Multiple Sequence Alignments.

Provides functions to calculate entropy and parsimony statistics across MSA columns.
"""
import warnings
import numpy as np
from collections import Counter
import msastats

AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

warnings.filterwarnings('ignore', category=RuntimeWarning)


def calculate_column_entropy(column):
    valid_chars = [char for char in column if char in AMINO_ACIDS]
    if len(valid_chars) == 0:
        return 0.0
    char_counts = Counter(valid_chars)
    total = len(valid_chars)
    entropy = 0.0
    for count in char_counts.values():
        if count > 0:
            p_i = count / total
            entropy -= p_i * np.log2(p_i)
    return entropy


def calculate_lag1_autocorr(values):
    if len(values) < 2:
        return 0.0
    original = values[:-1]
    lagged = values[1:]
    if np.std(original) == 0 or np.std(lagged) == 0:
        return 0.0
    correlation = np.corrcoef(original, lagged)[0, 1]
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def fit_gamma_to_values(values):
    from scipy import stats
    valid_values = values[values > 0]
    if len(valid_values) < 10:
        return 0.0
    try:
        shape, loc, scale = stats.gamma.fit(valid_values, floc=0)
        return float(shape)
    except Exception:
        return 0.0


def calculate_bimodality_coefficient(values):
    from scipy import stats
    if len(values) < 10:
        return 0.0
    try:
        skewness = stats.skew(values)
        kurtosis = stats.kurtosis(values)
        bc = (skewness**2 + 1) / (kurtosis + 3)
        return float(bc)
    except Exception:
        return 0.0

def calculate_entropy_histogram(entropies, n_bins=10):
    valid_entropies = entropies[entropies > 0]
    if len(valid_entropies) < 10:
        return {f'entropy_bin_{i}': 0.0 for i in range(n_bins - 1)}
    counts, _ = np.histogram(valid_entropies, bins=n_bins)
    frequencies = counts / np.sum(counts)
    # Drop last bin (linearly redundant since bins sum to 1, and outlier-prone)
    return {f'entropy_bin_{i}': float(frequencies[i]) for i in range(n_bins - 1)}


def calculate_autocorrelation_lags(values, max_lag=10):
    if len(values) < max_lag + 5:
        return {f'lag{i}_autocorr': 0.0 for i in range(1, max_lag + 1)}
    result = {}
    for lag in range(1, max_lag + 1):
        original = values[:-lag]
        lagged = values[lag:]
        if np.std(original) == 0 or np.std(lagged) == 0:
            correlation = 0.0
        else:
            correlation = np.corrcoef(original, lagged)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        result[f'lag{lag}_autocorr'] = float(correlation)
    return result


def calculate_run_length_features(entropies):
    """
    Run-length statistics normalized by alignment length.
    mean and max are divided by L (fraction of alignment).
    var is log1p-transformed to reduce outlier impact.
    count is already normalised by L.
    """
    if len(entropies) < 10:
        return {
            'high_run_mean': 0.0, 'high_run_var': 0.0,
            'high_run_max': 0.0,  'high_run_count': 0.0,
            'low_run_mean': 0.0,  'low_run_var': 0.0,
            'low_run_max': 0.0,   'low_run_count': 0.0,
        }

    threshold = float(np.median(entropies))
    # Degenerate case: all entropies equal (e.g. fully conserved) → runs meaningless
    if np.all(entropies == threshold):
        return {
            'high_run_mean': 0.0, 'high_run_var': 0.0,
            'high_run_max': 0.0,  'high_run_count': 0.0,
            'low_run_mean': 0.0,  'low_run_var': 0.0,
            'low_run_max': 0.0,   'low_run_count': 0.0,
        }
    is_high = entropies >= threshold

    high_runs = []
    low_runs = []
    current_val = is_high[0]
    current_len = 1

    for val in is_high[1:]:
        if val == current_val:
            current_len += 1
        else:
            (high_runs if current_val else low_runs).append(current_len)
            current_val = val
            current_len = 1
    (high_runs if current_val else low_runs).append(current_len)

    L = len(entropies)

    def _summarise(runs, L):
        if not runs:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.array(runs, dtype=float)
        return (
            float(np.log1p(np.mean(arr))),     # log1p of raw mean run length
            float(np.log1p(np.var(arr))),      # log1p of variance
            float(np.log1p(np.max(arr))),      # log1p of raw max run length
            len(runs) / L,                     # normalised count
        )

    hm, hv, hx, hc = _summarise(high_runs, L)
    lm, lv, lx, lc = _summarise(low_runs, L)

    return {
        'high_run_mean':  hm,
        'high_run_var':   hv,
        'high_run_max':   hx,
        'high_run_count': hc,
        'low_run_mean':   lm,
        'low_run_var':    lv,
        'low_run_max':    lx,
        'low_run_count':  lc,
    }


def calculate_msa_entropy_stats(alignment: np.ndarray):
    alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
    counts = np.array([(alignment == aa).sum(axis=0) for aa in alphabet])
    total = counts.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(counts > 0, counts / total, 0)
        log_p = np.where(p > 0, np.log2(p), 0)
    entropies = -(p * log_p).sum(axis=0)

    entropy_skewness, entropy_kurtosis = calculate_distribution_shape_features(entropies)
    bimodality_coef = (entropy_skewness**2 + 1) / (entropy_kurtosis + 3)
    histogram_features = calculate_entropy_histogram(entropies, n_bins=10)
    autocorr_features = calculate_autocorrelation_lags(entropies, max_lag=10)
    gamma_shape = fit_gamma_to_values(entropies)
    run_length_features = calculate_run_length_features(entropies)

    stats = {
        'avg_entropy': float(np.mean(entropies)),
        'entropy_variance': float(np.var(entropies, ddof=1)),
        'max_entropy': float(np.max(entropies)),
        'lag1_autocorr': calculate_lag1_autocorr(entropies),
        'entropy_skewness': float(np.clip(entropy_skewness, -10, 10)),
        'entropy_kurtosis': float(np.clip(entropy_kurtosis, -10, 10)),
        'bimodality_coefficient': bimodality_coef,
        'gamma_shape_entropy': float(np.log1p(gamma_shape)),
        **histogram_features,
        **autocorr_features,
        **run_length_features,
    }
    return stats




def calculate_alignment_features(sequences):
    n_sequences = len(sequences)
    seq_length = len(sequences[0]) if sequences else 0
    n_variable = 0
    for col_idx in range(seq_length):
        column = [seq[col_idx] for seq in sequences]
        if calculate_column_entropy(column) > 0:
            n_variable += 1
    return {
        'n_sequences': n_sequences,
        'fraction_variable_sites': n_variable / seq_length if seq_length > 0 else 0.0
    }


def calculate_gamma_shape_features(sequences):
    n_columns = len(sequences[0])
    entropies = []
    for col_idx in range(n_columns):
        column = [seq[col_idx] for seq in sequences]
        entropy = calculate_column_entropy(column)
        if entropy == 0:
            entropy = 1e-6
        entropies.append(entropy)
    entropies = np.array(entropies)
    if len(entropies) > 0:
        entropies /= np.mean(entropies)
    gamma_shape_entropy = fit_gamma_to_values(entropies)
    return {'gamma_shape_entropy': gamma_shape_entropy}


def calculate_distribution_shape_features(values):
    from scipy import stats
    if len(values) < 10:
        return 0.0, 0.0
    skew = float(stats.skew(values))
    kurt = float(stats.kurtosis(values))
    if not np.isfinite(skew): skew = 0.0
    if not np.isfinite(kurt): kurt = 0.0
    return skew, kurt


def read_phylip_sequences(phylip_file):
    from Bio import SeqIO
    sequences = []
    for record in SeqIO.parse(phylip_file, "phylip-sequential"):
        sequences.append(str(record.seq))
    return sequences

def read_fasta_sequences(fasta_file):
    from Bio import SeqIO
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    return sequences


def calculate_indel_features(sequences):
    """
    Indel features with transformations to reduce outlier impact:
    - avg_gap_size: log1p-transformed
    - msa_len: log-transformed
    - msa_max_len: divided by msa_len (fraction of alignment covered by longest seq)
    - msa_min_len: divided by msa_len (fraction of alignment covered by shortest seq)
    - tot_num_gaps: divided by (n_sequences * msa_len) → gap density
    """
    raw = msastats.calculate_msa_stats(sequences)[:9]
    avg_gap_size  = raw[0]
    msa_len       = raw[1]
    msa_max_len   = raw[2]
    msa_min_len   = raw[3]
    tot_num_gaps  = raw[4]
    n_seq = len(sequences)

    return {
        'avg_gap_size':            float(np.log1p(avg_gap_size)),
        'msa_len':                 float(np.log(msa_len)) if msa_len > 0 else 0.0,
        'msa_max_len':             float(msa_max_len / msa_len) if msa_len > 0 else 0.0,
        'msa_min_len':             float(msa_min_len / msa_len) if msa_len > 0 else 0.0,
        'tot_num_gaps':            float(tot_num_gaps / (n_seq * msa_len)) if (n_seq * msa_len) > 0 else 0.0,
        'num_gaps_len_one':        float(raw[5] / (n_seq * msa_len)) if (n_seq * msa_len) > 0 else 0.0,
        'num_gaps_len_two':        float(raw[6] / (n_seq * msa_len)) if (n_seq * msa_len) > 0 else 0.0,
        'num_gaps_len_three':      float(raw[7] / (n_seq * msa_len)) if (n_seq * msa_len) > 0 else 0.0,
        'num_gaps_len_at_least_four': float(raw[8] / (n_seq * msa_len)) if (n_seq * msa_len) > 0 else 0.0,
    }


def calculate_all_features(sequences) -> dict:
    """
    Single entry point for feature extraction used by both training and evaluation.
    Add or remove feature groups here — callers never need to change.
    """
    alignment = np.array([list(seq) for seq in sequences])  # built once
    #number of taxa:
    n_taxa = len(sequences)
    features = calculate_msa_entropy_stats(alignment)
    features.update(calculate_indel_features(sequences))
    # add number of taxa, log-transformed to reduce outlier impact (e.g. very large trees)
    features.update({'n_taxa': np.log(n_taxa)})
    return features