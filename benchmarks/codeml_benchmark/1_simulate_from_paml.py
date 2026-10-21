"""
Parametric bootstrap: simulate one MSA per gene on the gene's real tree, using the
codeml-estimated rho, the fixed alpha, and the gene's inferred indel parameters.

Input (base dir):  trees/{gene}.nwk, results_real.csv (gene, codeml_rho, alpha),
                   orthomam_indel_params.csv
Output:            sim_trees/{gene}.nwk, sim_msas/{gene}.fasta, sim_truth.csv
Genes without a codeml rho in results_real.csv are skipped.
"""

import csv
import pathlib

import pandas as pd
from msasim import (
    ALPHABET_CODES,
    MODEL_CODES,
    ReplacementModelSpec,
    SimProtocol,
    Simulator,
    SiteRateModelSpec,
)
from msasim.constants import SITE_RATE_MODELS
from msasim.distributions import ZipfDistribution

RESULTS_CSV = pathlib.Path("results_real.csv")
INDEL_CSV = pathlib.Path("orthomam_indel_params.csv")
TREE_DIR = pathlib.Path("trees")
SIM_TREE_DIR = pathlib.Path("sim_trees")
SIM_MSA_DIR = pathlib.Path("sim_msas")
TRUTH_CSV = pathlib.Path("sim_truth.csv")
GAMMA_CATEGORIES = 8
MAX_INDEL_LENGTH = 150  # from the indel-model training prior
SEED = 0


def simulate(newick, alpha, rho, indel, seed):
    site_rate_model = SiteRateModelSpec(
        gamma_alpha=alpha,
        gamma_categories=GAMMA_CATEGORIES,
        site_rate_correlation=rho,
        indel_awareness=SITE_RATE_MODELS.INDEL_AWARE,
    )
    replacement_model = ReplacementModelSpec(
        model=MODEL_CODES.LG,
        alphabet=ALPHABET_CODES.PROTEIN,
        site_rate_model=site_rate_model,
    )
    protocol = SimProtocol(
        tree=newick,
        root_seq_size=int(round(indel["root"])),
        insertion_rate=indel["ins_rate"],
        deletion_rate=indel["del_rate"],
        seed=seed,
    )
    protocol.set_insertion_length_distributions(ZipfDistribution(p=indel["len_ins"], truncation=MAX_INDEL_LENGTH))
    protocol.set_deletion_length_distributions(ZipfDistribution(p=indel["len_del"], truncation=MAX_INDEL_LENGTH))
    msa = Simulator(protocol, replacement_model=replacement_model)()

    rows = []
    for i in range(msa.get_num_sequences()):
        header, seq = msa.get_msa_row(i).split("\n")[:2]
        rows.append((header.lstrip(">").strip(), seq))
    return rows


def main():
    results = pd.read_csv(RESULTS_CSV, dtype={"gene": str}).dropna(subset=["codeml_rho"])
    indel = pd.read_csv(INDEL_CSV)
    indel["gene"] = indel["name"].str.replace(".fasta", "", regex=False)
    indel = indel.set_index("gene")

    SIM_TREE_DIR.mkdir(exist_ok=True)
    SIM_MSA_DIR.mkdir(exist_ok=True)
    print(f"{len(results)} genes to simulate")

    with open(TRUTH_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gene", "true_rho", "alpha", "n_columns"])
        for i, row in enumerate(results.itertuples()):
            newick = (TREE_DIR / f"{row.gene}.nwk").read_text().strip()
            rows = simulate(newick, row.alpha, row.codeml_rho, indel.loc[row.gene], seed=SEED + i + 1)
            (SIM_TREE_DIR / f"{row.gene}.nwk").write_text(newick + "\n")
            (SIM_MSA_DIR / f"{row.gene}.fasta").write_text("".join(f">{n}\n{s}\n" for n, s in rows))
            writer.writerow([row.gene, row.codeml_rho, row.alpha, len(rows[0][1])])
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(results)}")


if __name__ == "__main__":
    main()
