"""
Shannon Entropy and Parsimony calculation for Multiple Sequence Alignments.

Provides functions to calculate entropy and parsimony statistics across MSA columns.
"""
import numpy as np
from collections import Counter
from Bio import Phylo
from io import StringIO
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


def fitch_parsimony_column(column, tree):
    """
    Calculate Fitch parsimony score for a single alignment column.
    
    Uses the Fitch algorithm (1971) for unrooted trees.
    Gaps are excluded - only columns with valid amino acids are scored.
    
    Args:
        column: List of characters (one per taxon in tree order)
        tree: Bio.Phylo tree object
        
    Returns:
        int: Parsimony score (minimum number of substitutions), or None if all gaps
    """
    # Get terminal nodes (leaves) in consistent order
    terminals = list(tree.get_terminals())
    
    # Create a mapping from leaf names to states
    leaf_states = {}
    for i, terminal in enumerate(terminals):
        char = column[i]
        if char != '-' and char in config.AMINO_ACIDS:
            leaf_states[terminal] = {char}
        else:
            # Gap - will be skipped
            return None
    
    # If all gaps, skip this column
    if not leaf_states:
        return None
    
    # Check if we have states for all leaves
    if len(leaf_states) != len(terminals):
        return None
    
    # Fitch algorithm: bottom-up pass
    node_states = {}
    
    def postorder_fitch(clade):
        if clade.is_terminal():
            node_states[clade] = leaf_states[clade]
            return leaf_states[clade]
        
        # Get child states
        child_clades = clade.clades
        child_state_sets = [postorder_fitch(child) for child in child_clades]
        
        # Intersection of child states
        intersection = set.intersection(*child_state_sets)
        
        if intersection:
            # If intersection is non-empty, use it (no substitution needed)
            node_states[clade] = intersection
            return intersection
        else:
            # If intersection is empty, union all states (substitution occurred)
            union = set.union(*child_state_sets)
            node_states[clade] = union
            return union
    
    # Run postorder traversal
    postorder_fitch(tree.root)
    
    # Count substitutions: top-down pass
    score = 0
    
    def preorder_count(clade, parent_state=None):
        nonlocal score
        
        if parent_state is not None:
            # Check if current node's states intersect with parent
            if not (node_states[clade] & parent_state):
                score += 1
                # Choose arbitrary state from current node for children
                current_state = node_states[clade]
            else:
                # Choose from intersection
                current_state = node_states[clade] & parent_state
        else:
            # Root - choose arbitrary state
            current_state = node_states[clade]
        
        # Recurse to children
        if not clade.is_terminal():
            for child in clade.clades:
                preorder_count(child, current_state)
    
    preorder_count(tree.root)
    
    return score


def calculate_msa_parsimony_scores(sequences, tree_file):
    """
    Calculate parsimony scores for all columns in an MSA.
    
    Args:
        sequences: List of sequence strings (aligned, same length)
        tree_file: Path to Newick tree file
        
    Returns:
        numpy array: Parsimony scores for each column (excludes all-gap columns)
    """
    # Load tree
    tree = Phylo.read(tree_file, "newick")
    
    # Get alignment length
    n_columns = len(sequences[0])
    
    # Calculate parsimony score for each column
    scores = []
    for col_idx in range(n_columns):
        column = [seq[col_idx] for seq in sequences]
        score = fitch_parsimony_column(column, tree)
        if score is not None:  # Skip gap-only columns
            scores.append(score)
    
    return np.array(scores)


def calculate_entropy_quantile_features(entropies):
    """
    Calculate quantile-based features from entropy distribution.
    
    Captures the shape of the entropy distribution using percentiles,
    which should relate to the underlying gamma shape parameter (alpha).
    
    Args:
        entropies: numpy array of entropy values
        
    Returns:
        dict: Contains quantile features and coefficient of variation
    """
    # Filter positive entropies
    valid_entropies = entropies[entropies > 0]
    
    if len(valid_entropies) < 10:
        return {
            'entropy_q25': 0.0,
            'entropy_q50': 0.0,
            'entropy_q75': 0.0,
            'entropy_iqr': 0.0,
            'entropy_cv': 0.0
        }
    
    # Calculate quantiles
    q25 = float(np.percentile(valid_entropies, 25))
    q50 = float(np.percentile(valid_entropies, 50))  # median
    q75 = float(np.percentile(valid_entropies, 75))
    
    # Interquartile range (measure of spread)
    iqr = q75 - q25
    
    # Coefficient of variation (normalized variability)
    # CV = std / mean - directly measures relative dispersion
    mean_val = np.mean(valid_entropies)
    std_val = np.std(valid_entropies)
    cv = float(std_val / mean_val) if mean_val > 0 else 0.0
    
    return {
        'entropy_q25': q25,
        'entropy_q50': q50,
        'entropy_q75': q75,
        'entropy_iqr': iqr,
        'entropy_cv': cv
    }

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
    
    entropy_skewness, entropy_kurtosis = calculate_distribution_shape_features(entropies)
    # Calculate bimodality coefficient
    bimodality_coef = (entropy_skewness**2 + 1) / (entropy_kurtosis + 3)
    # Calculate statistics
    stats = {
        'avg_entropy': float(np.mean(entropies)),
        'entropy_variance': float(np.var(entropies, ddof=1)),  # Sample variance
        'max_entropy': float(np.max(entropies)),
        'lag1_autocorr': calculate_lag1_autocorr(entropies),
        'entropy_skewness': entropy_skewness,
        'entropy_kurtosis': entropy_kurtosis,
        'bimodality_coefficient': bimodality_coef
        
    }
    
    return stats

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

def calculate_msa_parsimony_stats(sequences, tree_file):
    """
    Calculate parsimony statistics for an entire MSA.
    
    Args:
        sequences: List of sequence strings (aligned, same length)
        tree_file: Path to Newick tree file
        
    Returns:
        dict: Contains parsimony-based features
    """
    # Calculate parsimony scores for all columns
    parsimony_scores = calculate_msa_parsimony_scores(sequences, tree_file)
    
    if len(parsimony_scores) == 0:
        return {
            'avg_parsimony_score': 0.0,
            'var_parsimony_score': 0.0,
            'lag1_parsimony_autocorr': 0.0
        }
    
    # Calculate statistics
    stats = {
        'avg_parsimony_score': float(np.mean(parsimony_scores)),
        'var_parsimony_score': float(np.var(parsimony_scores, ddof=1)),  # Sample variance
        'lag1_parsimony_autocorr': calculate_lag1_autocorr(parsimony_scores)
    }
    
    return stats


def calculate_parsimony_entropy_correlation(sequences, tree_file):
    """
    Calculate Pearson correlation between parsimony scores and entropy values.
    
    Both calculated on the same set of non-gap columns.
    
    Args:
        sequences: List of sequence strings (aligned, same length)
        tree_file: Path to Newick tree file
        
    Returns:
        float: Pearson correlation coefficient, or 0.0 if calculation fails
    """
    # Load tree
    tree = Phylo.read(tree_file, "newick")
    
    # Get alignment length
    n_columns = len(sequences[0])
    
    # Calculate both entropy and parsimony for each column (excluding gaps)
    entropies = []
    parsimony_scores = []
    
    for col_idx in range(n_columns):
        column = [seq[col_idx] for seq in sequences]
        
        # Calculate parsimony
        pars_score = fitch_parsimony_column(column, tree)
        
        # Only include if column has valid parsimony score (no gaps)
        if pars_score is not None:
            entropy = calculate_column_entropy(column)
            entropies.append(entropy)
            parsimony_scores.append(pars_score)
    
    # Convert to arrays
    entropies = np.array(entropies)
    parsimony_scores = np.array(parsimony_scores)
    
    # Calculate correlation
    if len(entropies) < 2:
        return 0.0
    
    if np.std(entropies) == 0 or np.std(parsimony_scores) == 0:
        return 0.0
    
    correlation = np.corrcoef(entropies, parsimony_scores)[0, 1]
    
    if np.isnan(correlation):
        return 0.0
    
    return float(correlation)


def calculate_gamma_shape_features(sequences, tree_file):
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
        if entropy > 0:  # Only include non-zero entropies
            entropies.append(entropy)
    
    entropies = np.array(entropies)
    
    # Get parsimony scores
    # parsimony_scores = calculate_msa_parsimony_scores(sequences, tree_file)
    
    # Fit gamma distributions
    gamma_shape_entropy = fit_gamma_to_values(entropies)
    # gamma_shape_parsimony = fit_gamma_to_values(parsimony_scores)
    
    return {
        'gamma_shape_entropy': gamma_shape_entropy,
        # 'gamma_shape_parsimony': gamma_shape_parsimony
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