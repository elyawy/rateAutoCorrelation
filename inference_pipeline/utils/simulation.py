"""
Shared simulation utilities for generating random trees and MSAs.
"""
from dataclasses import dataclass
import random
import numpy as np
from ete3 import Tree

try:
    from msasim import protocol, simulator as sim
    from msasim.msa import Msa
    from msasim.constants import SITE_RATE_MODELS
    from msasim.constants import MODEL_CODES
except ImportError:
    print("Error: 'msasim' library not found.")

import config


@dataclass
class SimulationParams:
    """Parameters controlling the shape of simulated MSAs."""
    min_taxa: int = 5
    max_taxa: int = 200
    min_seq_length: int = 50
    max_seq_length: int = 5000


def generate_random_tree(n_taxa: int, seed: int) -> Tree:
    """Generate a random ultrametric tree with n_taxa leaves."""
    np.random.seed(seed)
    tree = Tree()
    tree.populate(n_taxa, random_branches=True)
    for node in tree.traverse():
        if node.dist == 0:
            continue
        node.dist = np.random.exponential(scale=0.1)
    return tree


def setup_sim(tree: Tree, sim_seed: int):
    """
    Setup the simulator for a given tree and seed.

    Args:
        tree: ete3.Tree object
        sim_seed: Seed for this simulation

    Returns:
        msasim.Simulator object
    """
    newick_string = tree.write(format=1)
    simulation_protocol = protocol.SimProtocol(newick_string)
    simulation_protocol.set_insertion_rates(0.0)
    simulation_protocol.set_deletion_rates(0.0)
    simulation_protocol.set_site_rate_model(SITE_RATE_MODELS.SIMPLE)
    simulation_protocol.set_seed(sim_seed)
    simulator = sim.Simulator(simulation_protocol, simulation_type=sim.SIMULATION_TYPE.PROTEIN)
    return simulator


def simulate_msa(simulator, params: SimulationParams):
    """
    Simulate a single MSA from the given simulator.

    Args:
        simulator: msasim.Simulator object
        params: SimulationParams controlling seq length range

    Returns:
        tuple: (sequences, true_alpha, true_rho)
            sequences: list of sequence strings
            true_alpha: float
            true_rho: float
    """
    true_alpha = round(random.uniform(*config.ALPHA_RANGE), 3)
    true_rho = round(random.uniform(*config.RHO_RANGE), 3)
    seq_length = random.randint(params.min_seq_length, params.max_seq_length)

    simulator.protocol.set_sequence_size(seq_length)
    simulator.set_replacement_model(
        model=MODEL_CODES.WAG,
        gamma_parameters_alpha=true_alpha,
        gamma_parameters_categories=8,
        site_rate_correlation=true_rho
    )

    msa: Msa = simulator()
    sequences = [msa.get_msa_row(i).split("\n")[1] for i in range(msa.get_num_sequences())]

    return sequences, true_alpha, true_rho


def sequences_to_fasta(sequences: list) -> str:
    """Convert a list of sequence strings to a FASTA-formatted string."""
    lines = []
    for i, seq in enumerate(sequences):
        lines.append(f">seq_{i}")
        lines.append(seq)
    return "\n".join(lines)
