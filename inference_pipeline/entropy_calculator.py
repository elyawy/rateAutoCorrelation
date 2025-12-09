"""
Shannon Entropy calculation for Multiple Sequence Alignments.

Provides functions to calculate entropy statistics across MSA columns.
"""
import numpy as np
from collections import Counter
import config


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


def calculate_lag1_autocorr(entropies):
    """
    Calculate lag-1 autocorrelation of entropy values.
    
    This captures spatial correlation between neighboring sites,
    which is theoretically linked to the rho parameter (Markov process).
    
    Args:
        entropies: numpy array of per-site entropy values
        
    Returns:
        float: Lag-1 autocorrelation coefficient, or 0.0 if calculation fails
    """
    if len(entropies) < 2:
        return 0.0
    
    # Original sequence (exclude last value)
    original = entropies[:-1]
    
    # Lagged sequence (exclude first value)
    lagged = entropies[1:]
    
    # Calculate Pearson correlation
    # Handle edge case where variance is zero
    if np.std(original) == 0 or np.std(lagged) == 0:
        return 0.0
    
    correlation = np.corrcoef(original, lagged)[0, 1]
    
    # Handle NaN (shouldn't happen, but just in case)
    if np.isnan(correlation):
        return 0.0
    
    return float(correlation)


def calculate_msa_entropy_stats(sequences):
    """
    Calculate entropy statistics for an entire MSA.
    
    Args:
        sequences: List of sequence strings (aligned, same length)
        
    Returns:
        dict: Contains 'avg_entropy', 'entropy_variance', 'max_entropy', 'lag1_autocorr'
        
    Process:
        1. Iterate through each column
        2. Calculate entropy for each column
        3. Compute statistics across all column entropies
        4. Calculate lag-1 autocorrelation
    """
    if not sequences or len(sequences) == 0:
        return {
            'avg_entropy': 0.0,
            'entropy_variance': 0.0,
            'max_entropy': 0.0,
            'lag1_autocorr': 0.0
        }
    
    # Get alignment length
    n_columns = len(sequences[0])
    
    # Verify all sequences have the same length
    if not all(len(seq) == n_columns for seq in sequences):
        raise ValueError("All sequences must have the same length (alignment required)")
    
    # Calculate entropy for each column
    entropies = []
    for col_idx in range(n_columns):
        column = [seq[col_idx] for seq in sequences]
        entropy = calculate_column_entropy(column)
        entropies.append(entropy)
    
    # Convert to numpy array for easy statistics
    entropies = np.array(entropies)
    
    # Calculate statistics
    stats = {
        'avg_entropy': float(np.mean(entropies)),
        'entropy_variance': float(np.var(entropies, ddof=1)),  # Sample variance
        'max_entropy': float(np.max(entropies)),
        'lag1_autocorr': calculate_lag1_autocorr(entropies)
    }
    
    return stats


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

if __name__ == "__main__":
    # Example usage
    example_sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "ACDEFGHIKLPNPQRSTVW-",
        "ACDEFGHIKLMNPQRSTV--",
        "ACDEFGHIKLMNPQRSTVWY"
    ]
    
    stats = calculate_msa_entropy_stats(example_sequences)
    print("Entropy Statistics:")
    print(stats)