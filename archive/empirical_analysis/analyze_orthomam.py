import os
import zipfile
import csv
import statistics
import matplotlib.pyplot as plt

INPUT_DIR = "/home/pupkolab/temp/orthomam_AA"
OUTPUT_CSV = "orthomam_stats.csv"
OUTPUT_PLOT = "orthomam_scatter.png"


def parse_fasta(text):
    """Return (num_taxa, msa_length, median_seq_length).

    msa_length = len of first sequence incl gaps (alignment width).
    median_seq_length = median, across taxa, of each sequence's length
                         with gap characters ('-') stripped out.
    """
    sequences = []
    current_seq = []
    for line in text.splitlines():
        if line.startswith(">"):
            if current_seq:
                sequences.append("".join(current_seq))
                current_seq = []
        else:
            current_seq.append(line.strip())
    if current_seq:
        sequences.append("".join(current_seq))

    num_taxa = len(sequences)
    msa_length = len(sequences[0]) if sequences else 0

    if sequences:
        ungapped_lengths = [len(seq.replace("-", "")) for seq in sequences]
        median_seq_length = statistics.median(ungapped_lengths)
    else:
        median_seq_length = 0

    return num_taxa, msa_length, median_seq_length


INPUT_DIR_EXISTS = os.path.isdir(INPUT_DIR)
rows = []

if INPUT_DIR_EXISTS:
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.endswith(".zip"):
            continue
        zip_path = os.path.join(INPUT_DIR, fname)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                # find the fasta inside omm_filtered_AA_CDS/
                fasta_name = next(
                    n for n in zf.namelist()
                    if "omm_filtered_AA_CDS/" in n and n.endswith(".fasta")
                )
                with zf.open(fasta_name) as f:
                    text = f.read().decode("utf-8")
            num_taxa, msa_length, median_seq_length = parse_fasta(text)
            rows.append({
                "filename": fname,
                "num_taxa": num_taxa,
                "msa_length": msa_length,
                "median_seq_length": median_seq_length,
            })
        except Exception as e:
            print(f"SKIP {fname}: {e}")

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "num_taxa", "msa_length", "median_seq_length"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written: {OUTPUT_CSV} ({len(rows)} genes)")

    # Scatter plot
    taxa = [r["num_taxa"] for r in rows]
    lengths = [r["msa_length"] for r in rows]

    plt.figure(figsize=(8, 6))
    plt.scatter(taxa, lengths, alpha=0.4, s=10, color="steelblue")
    plt.xlabel("Number of Taxa")
    plt.ylabel("MSA Length (columns)")
    plt.title("OrthoMaM AA — Taxa vs MSA Length")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"Plot saved: {OUTPUT_PLOT}")
else:
    print(f"INPUT_DIR not found here ({INPUT_DIR}); "
          f"see self-test below for a demonstration of parse_fasta().")


# ---------------------------------------------------------------------------
# Self-test for parse_fasta(), runs regardless of whether INPUT_DIR exists.
# Confirms median_seq_length differs from msa_length when gaps are present,
# and matches msa_length when there are no gaps.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_no_gaps = ">a\nACGT\n>b\nACGA\n>c\nACGG\n"
    n, ml, med = parse_fasta(sample_no_gaps)
    assert (n, ml, med) == (3, 4, 4), (n, ml, med)

    # Sequences with different amounts of padding via gaps:
    #   a: ACGT----  -> ungapped length 4
    #   b: ACG-----  -> ungapped length 3
    #   c: ACGTACGT  -> ungapped length 8
    # msa_length (len of first seq incl gaps) = 8
    # median of [4, 3, 8] = 4
    sample_with_gaps = ">a\nACGT----\n>b\nACG-----\n>c\nACGTACGT\n"
    n, ml, med = parse_fasta(sample_with_gaps)
    assert (n, ml, med) == (3, 8, 4), (n, ml, med)

    print("Self-test passed: parse_fasta() returns "
          "(num_taxa, msa_length, median_seq_length) correctly, "
          "and median_seq_length reflects ungapped per-taxon lengths.")
