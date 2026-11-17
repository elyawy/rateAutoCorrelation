# Stripped-core experiment

Question: does trimming gap columns before running codeml improve or degrade the recovered rho?

Inputs: gapped and stripped alignments and matched codeml runs.

Commands: `python experiments/stripped_core/2_prepare_data.py`,
`python pipeline/3_codeml/run_codeml.py --mode core`, then
`python pipeline/4_inference/run_cnn_stripped.py`.

Output: `results/stripped_core/` metrics.

Expected headline: 192 usable genes; refit codeml versus stripped CNN
correlation 0.894, mean difference -0.076, and MAE 0.105.

Paper link: stripped-core benchmark.
