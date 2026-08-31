"""
Self-contained SBI predictor for alpha and rho inference from protein MSAs.

Usage:
    predictor = SBIPredictor("sbi_models/posterior.pt")
    alpha, rho, high_disagreement = predictor.predict(sequences)

Where `sequences` is a list of strings (one per taxon), all the same length
(i.e. columns of an MSA, gaps included).
"""

import numpy as np
import torch
import torch.nn as nn


class CNN1dEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x).mean(dim=-1)  # global average pool → (batch, 128)

WINDOW_SIZE = 500
ALPHABET_SIZE = 20
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
N_POSTERIOR_SAMPLES = 1000
DISAGREEMENT_THRESHOLD = 0.2  # std across chunk medians that triggers the flag


# ---------------------------------------------------------------------------
# Feature extraction (inlined from features_calculator.py)
# ---------------------------------------------------------------------------

def _encode_msa_to_integers(string_msa_array):
    """Map valid AA characters to 0-19, gaps/unknowns to -1."""
    lookup = np.full(256, fill_value=-1, dtype=np.int32)
    for idx, char in enumerate(ALPHABET):
        lookup[ord(char)] = idx
        lookup[ord(char.lower())] = idx
    ascii_view = string_msa_array.view("uint8")
    return lookup[ascii_view]


def _calculate_entropy_mi(msa_array, gap_value=-1):
    """
    Per-site Shannon entropy and adjacent-site MI, ignoring gaps.

    Returns
    -------
    entropies   : np.ndarray, shape (num_sites,)
    mi_adjacent : np.ndarray, shape (num_sites - 1,)
    """
    num_seqs, num_sites = msa_array.shape

    # --- entropy ---
    valid_mask = msa_array != gap_value
    valid_counts = np.sum(valid_mask, axis=0)
    site_counts = np.array([np.sum(msa_array == i, axis=0) for i in range(ALPHABET_SIZE)])
    safe_denom = np.where(valid_counts > 0, valid_counts, 1)
    site_probs = site_counts / safe_denom
    entropies = -np.sum(
        site_probs * np.log2(np.where(site_probs > 0, site_probs, 1.0)), axis=0
    )
    entropies = np.where(valid_counts > 0, entropies, 0.0)

    # --- MI between adjacent sites ---
    left = msa_array[:, :-1]
    right = msa_array[:, 1:]
    valid_pairs_mask = (left != gap_value) & (right != gap_value)
    valid_pair_counts = np.sum(valid_pairs_mask, axis=0)

    pair_codes = left * ALPHABET_SIZE + right
    pair_codes_masked = np.where(valid_pairs_mask, pair_codes, 0)

    num_pairs = ALPHABET_SIZE ** 2
    joint_counts = np.array(
        [np.sum((pair_codes_masked == i) & valid_pairs_mask, axis=0) for i in range(num_pairs)]
    )
    safe_pair_denom = np.where(valid_pair_counts > 0, valid_pair_counts, 1)
    joint_probs = joint_counts / safe_pair_denom
    joint_entropies = -np.sum(
        joint_probs * np.log2(np.where(joint_probs > 0, joint_probs, 1.0)), axis=0
    )
    joint_entropies = np.where(valid_pair_counts > 0, joint_entropies, 0.0)

    h_x = entropies[:-1]
    h_y = entropies[1:]
    mi_adjacent = np.clip(h_x + h_y - joint_entropies, 0.0, None)
    mi_adjacent = np.where(valid_pair_counts > 0, mi_adjacent, 0.0)

    return entropies, mi_adjacent


def _calculate_cnn_input(sequences):
    """
    Convert a list of aligned sequences (each exactly WINDOW_SIZE columns) into
    the (2, 500) tensor expected by the SBI CNN posterior.
    """
    alignment = np.array([list(seq) for seq in sequences], dtype="S1")
    encoded = _encode_msa_to_integers(alignment)
    entropies, mi_adjacent = _calculate_entropy_mi(encoded)
    mi_padded = np.append(mi_adjacent, 0.0)          # (499,) → (500,)
    return torch.tensor(
        np.stack([entropies, mi_padded], axis=0),     # (2, 500)
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _make_windows(sequences):
    """
    Split sequences into the minimal number of 500-column windows that cover
    the full alignment length, with the last window back-shifted to avoid
    a short tail.

    Examples
    --------
    length 500  → [(0, 500)]
    length 700  → [(0, 500), (200, 700)]
    length 1000 → [(0, 500), (500, 1000)]
    length 1100 → [(0, 500), (300, 800), (600, 1100)]
    """
    L = len(sequences[0])
    if L <= WINDOW_SIZE:
        return [(0, L)]

    # Number of windows needed so they cover L without a short tail
    n_windows = int(np.ceil(L / WINDOW_SIZE))
    if n_windows == 1:
        return [(0, L)]

    # Distribute starts evenly so the last window ends exactly at L
    step = (L - WINDOW_SIZE) / (n_windows - 1)
    windows = []
    for i in range(n_windows):
        start = round(i * step)
        end = start + WINDOW_SIZE
        windows.append((start, end))
    return windows


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class SBIPredictor:
    """
    Load once, call predict() many times.

    Parameters
    ----------
    posterior_path : str
        Path to the .pt file produced by torch.save(posterior, ...).
    n_samples : int
        Number of posterior samples drawn per chunk (default 1000).
    disagreement_threshold : float
        Std of per-chunk medians above which high_disagreement is True (default 0.2).
    """

    def __init__(
        self,
        posterior_path,
        n_samples=N_POSTERIOR_SAMPLES,
        disagreement_threshold=DISAGREEMENT_THRESHOLD,
    ):
        self.posterior = torch.load(posterior_path, map_location="cpu", weights_only=False)
        self.n_samples = n_samples
        self.disagreement_threshold = disagreement_threshold

    def _infer_chunk(self, sequences):
        """Run one 500-column sub-alignment through the posterior. Returns (alpha, rho)."""
        features = _calculate_cnn_input(sequences)           # (2, 500)
        samples = self.posterior.sample((self.n_samples,), x=features, show_progress_bars=False)
        median = samples.median(dim=0).values                # (3,): alpha, rho, tree_scale
        return float(median[0]), float(median[1])

    def predict(self, sequences):
        """
        Infer alpha and rho for a protein MSA.

        Parameters
        ----------
        sequences : list[str]
            Aligned sequences (gaps included), all the same length.

        Returns
        -------
        alpha             : float
        rho               : float
        high_disagreement : bool
            True when the std of per-chunk alpha or rho medians exceeds
            self.disagreement_threshold.
        """
        windows = _make_windows(sequences)

        alphas, rhos = [], []
        for start, end in windows:
            chunk = [seq[start:end] for seq in sequences]
            a, r = self._infer_chunk(chunk)
            alphas.append(a)
            rhos.append(r)

        alpha = float(np.mean(alphas))
        rho = float(np.mean(rhos))

        if len(windows) == 1:
            high_disagreement = False
        else:
            high_disagreement = (
                float(np.std(alphas)) > self.disagreement_threshold
                or float(np.std(rhos)) > self.disagreement_threshold
            )

        return alpha, rho, high_disagreement