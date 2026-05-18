"""
Shannon Entropy and Parsimony calculation for Multiple Sequence Alignments.

Provides functions to calculate entropy and parsimony statistics across MSA columns.
"""
import warnings
import numpy as np
from collections import Counter
from Bio import Phylo
from io import StringIO
import msastats

import config

warnings.filterwarnings('ignore', category=RuntimeWarning)


def calculate_column_entropy(column):
    """
    Calculate Shannon entropy for a single MSA column.
    
    Args:
        column: List or array of characters representing one column
        
    Returns:
        float: Entropy in bits (base-2), or 0.0 if column is all gaps
        
    Process:
        1. Filter out gap characters ('-')
        2. Count character frequencies
        3. Calculate proportions
        4. Apply Shannon formula: H = -sum(p_i * log2(p_i))
    """
    # Filter out gaps
    valid_chars = [char for char in column if char in config.AMINO_ACIDS]
    
    # Handle empty columns (all gaps)
    if len(valid_chars) == 0:
        return 0.0
    
    # Count frequencies
    char_counts = Counter(valid_chars)
    total = len(valid_chars)
    
    # Calculate entropy
    entropy = 0.0
    for count in char_counts.values():
        if count > 0:  # Skip zero counts (though Counter shouldn't have them)
            p_i = count / total
            # Handle p_i * log2(p_i) where we treat 0 * log(0) as 0
            entropy -= p_i * np.log2(p_i)
    
    return entropy


def calculate_lag1_autocorr(values):
    """
    Calculate lag-1 autocorrelation of values.
    
    This captures spatial correlation between neighboring sites.
    
    Args:
        values: numpy array of values (e.g., entropy or parsimony scores)
        
    Returns:
        float: Lag-1 autocorrelation coefficient, or 0.0 if calculation fails
    """
    if len(values) < 2:
        return 0.0
    
    # Original sequence (exclude last value)
    original = values[:-1]
    
    # Lagged sequence (exclude first value)
    lagged = values[1:]
    
    # Calculate Pearson correlation
    # Handle edge case where variance is zero
    if np.std(original) == 0 or np.std(lagged) == 0:
        return 0.0
    
    correlation = np.corrcoef(original, lagged)[0, 1]
    
    # Handle NaN (shouldn't happen, but just in case)
    if np.isnan(correlation):
        return 0.0
    
    return float(correlation)


def fit_gamma_to_values(values):
    """
    Fit a continuous gamma distribution to values and return shape parameter.
    
    This approximates the discrete gamma used in simulations.
    
    Args:
        values: numpy array of positive values (e.g., entropy or parsimony scores)
        
    Returns:
        float: Gamma shape parameter (alpha), or 0.0 if fitting fails
    """
    from scipy import stats
    
    # Filter out zeros and negatives (shouldn't happen, but be safe)
    valid_values = values[values > 0]
    
    if len(valid_values) < 10:  # Need enough data points
        return 0.0
    
    try:
        # Fit gamma distribution using MLE
        # Returns: (shape, loc, scale)
        shape, loc, scale = stats.gamma.fit(valid_values, floc=0)  # Fix location at 0
        
        # Return shape parameter (this is the alpha we want)
        return float(shape)
    
    except Exception:
        # Fitting failed (numerical issues, etc.)
        return 0.0



def calculate_bimodality_coefficient(values):
    """
    Calculate bimodality coefficient for a distribution.
    
    BC = (skewness² + 1) / (kurtosis + 3)
    
    BC > 0.555 suggests bimodality
    BC < 0.555 suggests unimodality
    
    Args:
        values: numpy array of values
        
    Returns:
        float: Bimodality coefficient, or 0.0 if calculation fails
    """
    from scipy import stats
    
    if len(values) < 10:
        return 0.0
    
    try:
        skewness = stats.skew(values)
        kurtosis = stats.kurtosis(values)  # Excess kurtosis (already -3)
        
        # Formula uses Pearson's kurtosis (excess + 3)
        bc = (skewness**2 + 1) / (kurtosis + 3)
        
        return float(bc)
    
    except Exception:
        return 0.0
    
def calculate_entropy_histogram(entropies, n_bins=10):
    """
    Calculate normalized histogram of entropy values.
    
    Args:
        entropies: numpy array of entropy values
        n_bins: number of bins for histogram
        
    Returns:
        dict: Contains 'entropy_bin_0' through 'entropy_bin_{n_bins-1}'
              representing normalized frequencies in each bin
    """
    # Filter positive entropies
    valid_entropies = entropies[entropies > 0]
    
    if len(valid_entropies) < 10:
        # Return zeros if insufficient data
        return {f'entropy_bin_{i}': 0.0 for i in range(n_bins)}
    
    # Create histogram with bins from min to max entropy
    counts, _ = np.histogram(valid_entropies, bins=n_bins)
    
    # Normalize to get frequencies (sum to 1)
    frequencies = counts / np.sum(counts)
    
    return {f'entropy_bin_{i}': float(frequencies[i]) for i in range(n_bins)}


def calculate_autocorrelation_lags(values, max_lag=10):
    """
    Calculate autocorrelation for multiple lags.
    
    This captures how correlation decays with distance along the sequence.
    
    Args:
        values: numpy array of values (e.g., entropy or parsimony scores)
        max_lag: maximum lag to calculate (default 10)
        
    Returns:
        dict: Contains 'lag1_autocorr' through 'lag{max_lag}_autocorr'
    """
    if len(values) < max_lag + 5:  # Need enough data points
        return {f'lag{i}_autocorr': 0.0 for i in range(1, max_lag + 1)}
    
    result = {}
    
    for lag in range(1, max_lag + 1):
        # Original sequence (exclude last 'lag' values)
        original = values[:-lag]
        
        # Lagged sequence (exclude first 'lag' values)
        lagged = values[lag:]
        
        # Calculate Pearson correlation
        if np.std(original) == 0 or np.std(lagged) == 0:
            correlation = 0.0
        else:
            correlation = np.corrcoef(original, lagged)[0, 1]
            
            # Handle NaN
            if np.isnan(correlation):
                correlation = 0.0
        
        result[f'lag{lag}_autocorr'] = float(correlation)
    
    return result


def calculate_run_length_features(entropies):
    """
    Calculate run-length statistics for high- and low-entropy sites separately.

    Sites are binarized at the median entropy. Consecutive sites of the same
    type form a "run". Summary statistics of those runs are informative about
    rho (clustering) and potentially alpha (asymmetry between high/low runs).

    Args:
        entropies: numpy array of per-site entropy values

    Returns:
        dict with keys:
            high_run_mean    - mean run length of high-entropy sites
            high_run_var     - variance of run lengths of high-entropy sites
            high_run_max     - longest run of high-entropy sites
            high_run_count   - number of high-entropy runs, normalised by L
            low_run_mean     - mean run length of low-entropy sites
            low_run_var      - variance of run lengths of low-entropy sites
            low_run_max      - longest run of low-entropy sites
            low_run_count    - number of low-entropy runs, normalised by L
    """
    if len(entropies) < 10:
        return {
            'high_run_mean': 0.0, 'high_run_var': 0.0,
            'high_run_max': 0.0,  'high_run_count': 0.0,
            'low_run_mean': 0.0,  'low_run_var': 0.0,
            'low_run_max': 0.0,   'low_run_count': 0.0,
        }

    threshold = float(np.median(entropies))
    is_high = entropies >= threshold  # boolean array

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
    # flush last run
    (high_runs if current_val else low_runs).append(current_len)

    L = len(entropies)

    def _summarise(runs, L):
        if not runs:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.array(runs, dtype=float)
        return (
            float(np.mean(arr)),
            float(np.var(arr)),
            float(np.max(arr)),
            len(runs) / L,          # normalised count
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


def calculate_msa_entropy_stats(sequences):
    """
    Calculate entropy statistics for an entire MSA.
    
    Args:
        sequences: List of sequence strings (aligned, same length)
        
    Returns:
        dict: Contains entropy statistics, histogram bins, and multi-lag autocorrelations
        
    Process:
        1. Iterate through each column
        2. Calculate entropy for each column
        3. Compute statistics across all column entropies
        4. Calculate lag-1 autocorrelation
        5. Calculate histogram bins
        6. Calculate multi-lag autocorrelations
        7. Calculate run-length features
    """
    alignment = np.array([list(seq) for seq in sequences])  # (N, L)
    alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
    
    # count each amino acid per site: (20, L)
    counts = np.array([(alignment == aa).sum(axis=0) for aa in alphabet])
    
    total = counts.sum(axis=0)  # (L,) non-gap counts
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(counts > 0, counts / total, 0)  # (20, L)
        log_p = np.where(p > 0, np.log2(p), 0)
    
    entropies = -(p * log_p).sum(axis=0)  # (L,)
    
    entropy_skewness, entropy_kurtosis = calculate_distribution_shape_features(entropies)
    # Calculate bimodality coefficient
    bimodality_coef = (entropy_skewness**2 + 1) / (entropy_kurtosis + 3)
    
    # Calculate histogram bins
    histogram_features = calculate_entropy_histogram(entropies, n_bins=10)
    
    # Calculate multi-lag autocorrelations
    autocorr_features = calculate_autocorrelation_lags(entropies, max_lag=10)

    # Calculate Gamma shape feature
    gamma_shape = fit_gamma_to_values(entropies)

    # Calculate run-length features
    run_length_features = calculate_run_length_features(entropies)
    
    # Calculate statistics
    stats = {
        'avg_entropy': float(np.mean(entropies)),
        'entropy_variance': float(np.var(entropies, ddof=1)),  # Sample variance
        'max_entropy': float(np.max(entropies)),
        'lag1_autocorr': calculate_lag1_autocorr(entropies),  # Keep original for compatibility
        'entropy_skewness': entropy_skewness,
        'entropy_kurtosis': entropy_kurtosis,
        'bimodality_coefficient': bimodality_coef,
        'gamma_shape_entropy': gamma_shape,
        **histogram_features,
        **autocorr_features,
        **run_length_features,
    }
    
    return stats

def calc_boolean_consensus_msa(sequences):
    # convert the list of sequences into a 2D numpy array of shape (n_taxa, n_sites)
    alignment = np.array([list(seq) for seq in sequences])
    gap = '-'
    non_gap = alignment != gap

    alphabet = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
    counts = np.array([(alignment == aa) for aa in alphabet])  # (20, N, L)
    counts *= non_gap
    consensus_idx = counts.sum(axis=1).argmax(axis=0)  # (L,)
    consensus = alphabet[consensus_idx]  # (L,)

    r = np.where(non_gap, alignment != consensus, np.nan).astype(float)
    return r


def calculate_msa_conservation_stats(sequences):
    from scipy.optimize import curve_fit
    lags = [1, 2, 5, 10, 20]
    r = calc_boolean_consensus_msa(sequences)  # (N, L)
    n_taxa, n_sites = r.shape
    features = {}
    phi_values = []

    for lag in lags:
        X = r[:, :n_sites - lag]   # (N, L-lag)
        Y = r[:, lag:]             # (N, L-lag)

        valid = ~np.isnan(X) & ~np.isnan(Y)  # (N, L-lag)
        n_valid = valid.sum(axis=0)           # (L-lag,)

        X = np.where(valid, X, np.nan)
        Y = np.where(valid, Y, np.nan)

        mean_x = np.nanmean(X, axis=0)
        mean_y = np.nanmean(Y, axis=0)
        std_x  = np.nanstd(X, axis=0)
        std_y  = np.nanstd(Y, axis=0)

        cov = np.nanmean((X - mean_x) * (Y - mean_y), axis=0)
        good = (std_x > 1e-10) & (std_y > 1e-10) & (n_valid >= 10)
        phi = np.where(good, cov / (std_x * std_y), np.nan)

        phi_mean = float(np.nanmean(phi)) if good.any() else 0.0
        features[f'cons_lag{lag}_phi'] = phi_mean
        phi_values.append(phi_mean)



    # Fit exponential decay: phi(lag) = A * decay^lag
    # decay constant is a direct estimate of rho
    def exp_decay(lag, A, decay):
        return A * np.power(decay, lag)
 
    try:
        lags_arr = np.array(lags, dtype=float)
        phi_arr = np.array(phi_values)
        if np.any(phi_arr > 1e-6):
            popt, _ = curve_fit(
                exp_decay, lags_arr, phi_arr,
                p0=[phi_arr[0], 0.5],
                bounds=([0, 0], [1, 1]),
                maxfev=1000
            )
            features['cons_decay_A'] = float(popt[0])
            features['cons_decay_rho'] = float(popt[1])
        else:
            features['cons_decay_A'] = 0.0
            features['cons_decay_rho'] = 0.0
    except Exception:
        features['cons_decay_A'] = 0.0
        features['cons_decay_rho'] = 0.0
 
    return features


def calculate_alignment_features(sequences):
    """
    Calculate basic alignment properties.
    
    Returns:
        dict with n_sequences, fraction_variable_sites
    """
    n_sequences = len(sequences)
    seq_length = len(sequences[0]) if sequences else 0
    
    # Count variable sites (entropy > 0)
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
    """
    Fit gamma distributions to entropy and parsimony values.
    
    Returns shape parameters which should correlate with the true alpha parameter.
    
    Args:
        sequences: List of sequence strings (aligned, same length)
        tree_file: Path to Newick tree file
        
    Returns:
        dict: Contains 'gamma_shape_entropy' and 'gamma_shape_parsimony'
    """
    # Get entropy values for all columns
    n_columns = len(sequences[0])
    entropies = []
    for col_idx in range(n_columns):
        column = [seq[col_idx] for seq in sequences]
        entropy = calculate_column_entropy(column)
        if entropy == 0:
            entropy = 1e-6
        entropies.append(entropy)
    
    entropies = np.array(entropies)
    # normalize so that the average is 1.0
    if len(entropies) > 0:
        entropies /= np.mean(entropies)
    
    
    # Fit gamma distributions
    gamma_shape_entropy = fit_gamma_to_values(entropies)
    
    return {
        'gamma_shape_entropy': gamma_shape_entropy,
    }


def calculate_distribution_shape_features(values):
    """
    Calculate shape statistics that relate to gamma shape parameter.
    
    Low alpha → high skewness (long right tail)
    High alpha → low skewness (more uniform)
    """
    from scipy import stats
    
    if len(values) < 10:
        return {'skewness': 0.0, 'kurtosis': 0.0}
    
    return float(stats.skew(values)), float(stats.kurtosis(values))
    

def read_phylip_sequences(phylip_file):
    """
    Read sequences from a PHYLIP format file.
    
    Args:
        phylip_file: Path to PHYLIP format alignment file
        
    Returns:
        list: List of sequence strings (without headers)
    """
    from Bio import SeqIO
    
    sequences = []
    for record in SeqIO.parse(phylip_file, "phylip-sequential"):
        sequences.append(str(record.seq))
    
    return sequences

def read_fasta_sequences(fasta_file):
    """
    Read sequences from a FASTA format file.
    
    Args:
        fasta_file: Path to FASTA format alignment file
        
    Returns:
        list: List of sequence strings (without headers)
    """
    from Bio import SeqIO
    
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    
    return sequences

# calculate indel related features:
# return them with the following lower case names:
# 'avg_gap_size', 'msa_len', 'msa_max_len', 'msa_min_len', 'tot_num_gaps', 'num_gaps_len_one', 'num_gaps_len_two', 'num_gaps_len_three', 'num_gaps_len_at_least_four'
def calculate_indel_features(sequences):
    stats = msastats.calculate_msa_stats(sequences)[:9]
    stats = {
        'avg_gap_size': stats[0],
        'msa_len': stats[1],
        'msa_max_len': stats[2],
        'msa_min_len': stats[3],
        'tot_num_gaps': stats[4],
        'num_gaps_len_one': stats[5],
        'num_gaps_len_two': stats[6],
        'num_gaps_len_three': stats[7],
        'num_gaps_len_at_least_four': stats[8],
    }
    return stats


def calculate_all_features(sequences) -> dict:
    """
    Single entry point for feature extraction used by both training and evaluation.
    Add or remove feature groups here — callers never need to change.
    """
    features = calculate_msa_entropy_stats(sequences)
    features.update(calculate_msa_conservation_stats(sequences))
    features.update(calculate_indel_features(sequences))
    return features


    # if __name__ == "__main__":
    #     # Example usage
    #     example_sequences = [
    #         "ACDEFGHIKLMNPQRSTVWY",
    #         "ACDEFGHIKLPNPQRSTVW-",
    #         "ACDEFGHIKLMNMQRSTV--",
    #         "ACDEFGHIKLMNPQRSTVWY",
    #         "ACDEFGHIKLLNMQRSTVWY",
    #         "ACDEFGHIKLMNPQRSTVWY",
    #         "ACDEFGHIKLMNPQRSTVWY",
    #         "ACDEFGHIKLMNPQRSTVWY",
    #         "ACDEFGHIKLPNMQRSTVWY",
    #         "ACDEFGHIKLMNPQRSTVWY",
    #         "ACDEFGHIKLMNPQRSTVWY",
    #     ]
        # stats = calculate_msa_entropy_stats(example_sequences)
    # print("Entropy Statistics:")
    # print(stats)
    # stats = calculate_gamma_shape_features(example_sequences)
    # print("Gamma Shape Features:")
    # print(stats)
    # stats = calculate_alignment_features(example_sequences)
    # print("Alignment Features:")
    # print(stats)
    # indel_stats = calculate_indel_features(example_sequences)
    # print("Indel Features:")
    # print(indel_stats)
    # stats = calculate_msa_conservation_stats(example_sequences)
    # print("Conservation Statistics:")
    # print(stats)
