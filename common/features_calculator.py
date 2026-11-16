"""
MSA feature extraction for the rho regressor.

calculate_cnn_input     -> (2, n_sites) tensor: per-site entropy and per-site gap fraction
calculate_scalar_features -> (2,) tensor: log mean pairwise divergence, log number of sequences
"""
import numpy as np
import torch


def encode_msa_to_integers(string_msa_array, alphabet_size=None):
    """
    Optimized single-pass string-to-integer encoder using a lookup array.
    Maps valid characters to 0->(N-1) and all gaps/unrecognized characters to -1.
    """
    # Create an ASCII lookup array (256 slots) initialized to our gap value (-1)
    lookup = np.full(256, fill_value=-1, dtype=np.int32)

    # Fill in only the ASCII positions of our valid characters
    if alphabet_size == 20:
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
    elif alphabet_size == 4:
        alphabet = "ACGT"
    else:
        raise ValueError("Unsupported alphabet size. Please specify either 4 (DNA) or 20 (Protein).")

    for idx, char in enumerate(alphabet):
        lookup[ord(char)] = idx
        # Map lowercase variants to the same index to stay case-insensitive
        lookup[ord(char.lower())] = idx

    # Convert string matrix to a raw 8-bit unsigned integer view (ASCII values)
    # This avoids Python string loops and lets NumPy do a direct index lookup
    ascii_view = string_msa_array.view('uint8')

    return lookup[ascii_view]


def calculate_cnn_input(sequences, alphabet_size=None) -> torch.Tensor:
    alignment = np.array([list(seq) for seq in sequences], dtype='S1')

    encoded_msa = encode_msa_to_integers(alignment, alphabet_size=alphabet_size)

    # per-site Shannon entropy over non-gap residues (all-gap columns get 0)
    counts = np.array([np.sum(encoded_msa == i, axis=0) for i in range(alphabet_size)])  # (alphabet, sites)
    valid = counts.sum(axis=0)
    probs = counts / np.where(valid > 0, valid, 1)
    entropies = -np.sum(probs * np.log2(np.where(probs > 0, probs, 1.0)), axis=0)

    # fraction of sequences with a gap at each site
    gap_fraction = (encoded_msa == -1).mean(axis=0)

    return torch.tensor(
        np.stack([entropies, gap_fraction], axis=0),  # (2, n_sites)
        dtype=torch.float32
    )


def calculate_scalar_features(sequences, alphabet_size=None) -> torch.Tensor:
    """Alignment-level scalars: log mean pairwise divergence (gap-aware), log number of sequences."""
    alignment = np.array([list(seq) for seq in sequences], dtype='S1')
    encoded_msa = encode_msa_to_integers(alignment, alphabet_size=alphabet_size)

    counts = np.array([np.sum(encoded_msa == i, axis=0) for i in range(alphabet_size)])  # (alphabet, sites)
    valid = counts.sum(axis=0)  # sequences with a residue in each column
    total_pairs = (valid * (valid - 1) / 2).sum()
    agreeing_pairs = (counts * (counts - 1) / 2).sum()
    divergence = (total_pairs - agreeing_pairs) / total_pairs if total_pairs > 0 else 0.0

    return torch.tensor(
        [np.log(max(divergence, 1e-4)), np.log(len(sequences))],
        dtype=torch.float32,
    )