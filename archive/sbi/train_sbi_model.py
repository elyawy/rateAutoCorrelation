"""
SBI based posterior inference for alpha and rho parameters of the gamma distribution of rate variation across sites.
This script uses existing training data (features + true parameters) to train a neural network model that can predict alpha and rho from features.
1. Trains a neural network using sbi (NPE) to learn the posterior distribution of alpha and rho given the features.
2. present the results with the script visualize_posteriors.py, which will show how well the model can recover the true parameters on a test set and visualize the learned posterior distributions.
"""
from copyreg import pickle
from dataclasses import dataclass
import math
import random
import pickle
import pathlib
import torch
from sbi.utils import BoxUniform
from sbi.inference import NPE
from sbi.inference import simulate_for_sbi

from ete3 import Tree
import numpy as np

from msasim import protocol, simulator as sim
from msasim.msa import Msa
from msasim.distributions import ZipfDistribution
from msasim.constants import SITE_RATE_MODELS
from msasim.constants import MODEL_CODES

import training.prior as prior
from common.features_calculator import calculate_all_features

@dataclass
class SimulationParams:
    """Parameters controlling the shape of simulated MSAs."""
    min_taxa: int = 20
    max_taxa: int = 200
    min_seq_length: int = 100
    max_seq_length: int = 500


def generate_random_tree(n_taxa: int, scale: float) -> Tree:
    """Generate a random ultrametric tree with n_taxa leaves."""

    tree = Tree()
    tree.populate(n_taxa, random_branches=True)
    for node in tree.traverse():
        if node.dist == 0:
            continue
        node.dist = np.random.exponential(scale=scale)
    return tree


def setup_sim(tree: Tree) -> sim.Simulator:
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
    # indel model parameters based on mammalian rates from https://doi.org/10.1093/bioinformatics/btaf686
    simulation_protocol.set_insertion_rates(0.007)
    simulation_protocol.set_deletion_rates(0.035)
    simulation_protocol.set_insertion_length_distributions(ZipfDistribution(p=1.53, truncation=50))
    simulation_protocol.set_deletion_length_distributions(ZipfDistribution(p=1.11, truncation=50))
    simulation_protocol.set_site_rate_model(SITE_RATE_MODELS.INDEL_AWARE)
    simulator = sim.Simulator(simulation_protocol, simulation_type=sim.SIMULATION_TYPE.PROTEIN)
    return simulator


prior_dist = BoxUniform(
    low=torch.tensor([prior.ALPHA_RANGE[0], prior.RHO_RANGE[0], math.log10(prior.TREE_SCALE_RANGE[0])]),
    high=torch.tensor([prior.ALPHA_RANGE[1], prior.RHO_RANGE[1], math.log10(prior.TREE_SCALE_RANGE[1])])
)    


def simulate(theta: torch.Tensor):
    """

    Simulate a single an MSA based on a random tree.

    Args:
        theta: torch.Tensor of shape (3,) containing the parameters (alpha, rho, tree_scale) for this simulation
    Returns:
        tuple: (sequences, true_alpha, true_rho, true_tree_scale)
            sequences: list of sequence strings
            true_alpha: float
            true_rho: float
            true_tree_scale: float
    """

    params = SimulationParams()  # Use default simulation parameters for taxa and sequence length ranges
    # go through all params in tensor and simulate for each then return all features as tensor
    # loop through each row of theta and simulate an MSA for each set of parameters, then save features for each MSA and return as tensor
    features_list = []
    for i in range(theta.shape[0]):
        true_alpha, true_rho, true_tree_scale = theta[i].tolist()
        # convert log10 tree_scale back to linear scale
        true_tree_scale = 10 ** true_tree_scale


        n_taxa = random.randint(params.min_taxa, params.max_taxa)
        simulator = setup_sim(generate_random_tree(n_taxa=n_taxa, scale=true_tree_scale))  # Example simulator for testing

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

        stats = calculate_all_features(sequences)
        features_list.append(torch.tensor(list(stats.values()), dtype=torch.float32))

    # return tensor of features for this simulation
    return torch.stack(features_list)

#%%
def main():
    # Example of how to use the prior and simulator together


    theta_0 = prior_dist.sample((1,))
    feature_0 = simulate(theta_0)

    # print theta and features for this example simulation
    print("Sampled parameters (alpha, rho, tree_scale):", theta_0)
    # print("Calculated features:", feature_0)
    num_simulations = 10000
    thetas, features = simulate_for_sbi(simulate, prior_dist, 
                                        num_simulations=num_simulations,
                                        num_workers=7)  # Simulate MSAs and calculate features for each

    inference = NPE(prior=prior_dist)


    inference.append_simulations(thetas, features).train()
    
    posterior = inference.build_posterior()
    print(posterior)

    x_obs = feature_0
    samples = posterior.sample((1000,), x=x_obs)

    # mean and std of samples
    print("Posterior samples mean:", samples.mean(dim=0))
    print("Posterior samples std:", samples.std(dim=0))
#%%

    with open("sbi_models/my_posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

    with open("sbi_models/my_inference.pkl", "wb") as handle:
        pickle.dump(inference, handle)


if __name__ == "__main__":
    main()
