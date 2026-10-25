# plotting the histogram of rho values for both the entire orthomam database (minus several genes)
# predicted values from the neural network on the empirical unfiltered MSAs.

import pandas as pd
from matplotlib import pyplot as plt

RESULTS_FILE = "data/results_real.csv"
# gene,nn_rho,codeml_rho,alpha <- we only plot rho values from NN, also alpha for completeness.


df = pd.read_csv(RESULTS_FILE)
# should also show median, mean, and std of the distribution of rho values for the neural network predictions.
print(f"Neural Network Rho - Mean: {df['nn_rho'].mean():.3f}, Median: {df['nn_rho'].median():.3f}, Std: {df['nn_rho'].std():.3f}")
plt.hist(df["nn_rho"], bins=30, alpha=0.7, label="Neural Network")
plt.axvline(df['nn_rho'].mean(), color='red', linestyle='--', label=f"Mean: {df['nn_rho'].mean():.3f}")
plt.axvline(df['nn_rho'].median(), color='blue', linestyle='--', label=f"Median: {df['nn_rho'].median():.3f}")

plt.xlabel("Rho")
plt.ylabel("Frequency")
plt.title("Distribution of Rho Values")
plt.legend()
plt.show()