# plotting the histogram of rho values for both the entire orthomam database (minus several genes)
# predicted values from the neural network and PAML on the empirical unfiltered MSAs.
# codeml should be in orange and nn in blue, both slightly transparent, with a legend.
# a color-blind friendly way to distinguish the two distributions should be used, such as different hatch patterns or line styles.
# style should be clean with right and top spines removed, and the x and y axis labels should be larger than the default size.
# seaborn style should be used, and the figure should be saved as a high-resolution PNG file (300dpi).
# rho should use the LaTeX symbol for rho.

import pandas as pd
from matplotlib import pyplot as plt

RESULTS_FILE_NN = "empirical_analysis/results/results_real.csv"
RESULTS_FILE_PAML = "empirical_analysis/results/codeml_rho_raw.csv"
# gene,nn_rho,codeml_rho,alpha <- we only plot rho values from NN, also alpha for completeness.


df_cnn = pd.read_csv(RESULTS_FILE_NN)
df_paml = pd.read_csv(RESULTS_FILE_PAML)
df_paml = df_paml.rename(columns={"rho": "codeml_rho"})
# clip values to be between 0 and 1, those above 1 become 1, those below 0 become 0. This is because the neural network can predict values outside of the range [0, 1], but we want to plot the distribution of rho values within this range.
df_cnn["nn_rho"] = df_cnn["nn_rho"].clip(0, 1)
df_paml["codeml_rho"] = df_paml["codeml_rho"].clip(0, 1)

plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(10, 6))
plt.hist(df_cnn["nn_rho"], bins=30, alpha=0.6, label="Neural Network", color="blue", edgecolor="black")
plt.hist(df_paml["codeml_rho"], bins=30, alpha=0.6, label="PAML", color="orange", edgecolor="black", hatch="//")

plt.xlabel(r"$\rho$", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Distribution of $\\rho$ Values", fontsize=16)
plt.legend()
plt.savefig("empirical_analysis/results/plots/rho_distribution_histogram.png", dpi=300)