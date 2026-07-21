import os
import zipfile
import csv
import io
import matplotlib.pyplot as plt

INPUT_DIR = "/home/pupkolab/temp/orthomam_AA"
OUTPUT_CSV = "orthomam_stats.csv"
OUTPUT_PLOT = "orthomam_scatter.png"


def parse_fasta(text):
    """Return (num_taxa, msa_length). msa_length = len of first sequence incl gaps."""
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
    return num_taxa, msa_length


rows = []

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
        num_taxa, msa_length = parse_fasta(text)
        rows.append({"filename": fname, "num_taxa": num_taxa, "msa_length": msa_length})
    except Exception as e:
        print(f"SKIP {fname}: {e}")

# Write CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "num_taxa", "msa_length"])
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
