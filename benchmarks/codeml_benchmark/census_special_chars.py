"""
Census of special characters (X, !, *) in the unfiltered OrthoMaM protein alignments.

Per gene, counts taxa containing each symbol and how many taxa the planned policy would drop:
  drop a taxon if it has X, '!', or an internal '*' (a '*' that is not the last non-gap character).
Also prints the overall character alphabet, to reveal any other unexpected symbols.
Writes special_chars_census.csv.
"""

import csv
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

AA_DIR = Path("/home/elyalab/temp/orthomam_AA_unfiltered")  # *.aln.zip or *.aln, searched recursively
OUTPUT = Path("special_chars_census.csv")
N_WORKERS = 10
FIELDS = ["gene", "n_seqs", "columns", "taxa_with_X", "taxa_with_bang", "taxa_with_internal_star",
          "taxa_with_terminal_star_only", "taxa_dropped", "taxa_remaining"]


def parse_fasta(text: str) -> list:
    """Return [(name, sequence)] in file order."""
    records, name, cur = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(cur)))
            name, cur = line[1:].strip(), []
        else:
            cur.append(line.strip())
    if name is not None:
        records.append((name, "".join(cur)))
    return records


def read_records(path: Path) -> list:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            fasta_name = next(f for f in zf.namelist() if f.endswith(".aln"))
            return parse_fasta(zf.read(fasta_name).decode("utf-8"))
    return parse_fasta(path.read_text())


def inspect(path: Path):
    records = read_records(path)
    alphabet = Counter()
    n_x = n_bang = n_internal_star = n_terminal_star = n_dropped = 0
    for _, seq in records:
        seq = seq.upper()
        alphabet.update(seq)
        ungapped = seq.replace("-", "")
        has_x = "X" in seq
        has_bang = "!" in seq
        internal_star = "*" in ungapped[:-1]
        terminal_star = ungapped.endswith("*") and not internal_star
        n_x += has_x
        n_bang += has_bang
        n_internal_star += internal_star
        n_terminal_star += terminal_star
        n_dropped += has_x or has_bang or internal_star
    gene = path.name.split("_")[0]
    row = {"gene": gene, "n_seqs": len(records), "columns": len(records[0][1]), "taxa_with_X": n_x,
           "taxa_with_bang": n_bang, "taxa_with_internal_star": n_internal_star,
           "taxa_with_terminal_star_only": n_terminal_star, "taxa_dropped": n_dropped,
           "taxa_remaining": len(records) - n_dropped}
    return row, alphabet


def main():
    paths = sorted([*AA_DIR.rglob("*.aln.zip"), *AA_DIR.rglob("*.aln")])
    print(f"{len(paths)} alignments found")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(inspect, paths, chunksize=50))

    rows = [r for r, _ in results]
    with open(OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    alphabet = Counter()
    for _, a in results:
        alphabet.update(a)
    print("\ncharacter counts over all alignments:")
    for ch, n in alphabet.most_common():
        print(f"  {ch!r}: {n}")

    n = len(rows)
    total_taxa = sum(r["n_seqs"] for r in rows)
    total_dropped = sum(r["taxa_dropped"] for r in rows)
    print(f"\ngenes with at least one taxon dropped: {sum(r['taxa_dropped'] > 0 for r in rows)}/{n}")
    print(f"taxa dropped overall: {total_dropped}/{total_taxa} ({100 * total_dropped / total_taxa:.1f}%)")
    fractions = sorted(r["taxa_dropped"] / r["n_seqs"] for r in rows)
    for q in (0.5, 0.9, 0.99):
        print(f"  {int(q * 100)}th percentile of fraction dropped per gene: {fractions[int(q * (n - 1))]:.2f}")
    print(f"details in {OUTPUT}")


if __name__ == "__main__":
    main()
