# Alignment effect

Question: how does alignment quality alter the recovered rho under MAFFT alignment?

Inputs: simulated alignments and downstream MAFFT / SP comparisons.

Output: `results/alignment_effect/` metrics.

The existing MAFFT workflow is retained as a reproducibility folder. Its
historical simulations use per-gene codeml-estimated rho and therefore mix rho
with gene effects; a clean fixed-gene rho grid is not implemented.
