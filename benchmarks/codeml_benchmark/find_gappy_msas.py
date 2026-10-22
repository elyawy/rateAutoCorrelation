"""
Find cleaned protein MSAs (see msa_cleaning.py) with MIN_COLUMNS..MAX_COLUMNS columns and
>= MIN_UNIQUE_LENGTHS distinct ungapped sequence lengths.
Writes selected_msas.csv (gene, columns, n_seqs, n_dropped, n_unique_lengths), measured after cleaning.
"""

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from msa_cleaning import clean_alignment, find_alignments, read_records

OUTPUT = Path("selected_msas.csv")
MIN_COLUMNS = 100 # 500
MAX_COLUMNS = 50000 # 3000
MIN_UNIQUE_LENGTHS = 1 # 50
N_WORKERS = 6


def inspect(item):
    gene, path = item
    cleaned, n_dropped, reason = clean_alignment(read_records(path))
    if cleaned is None:
        return gene, None, reason
    seqs = list(cleaned.values())
    n_unique = len({len(s.replace("-", "")) for s in seqs})
    return gene, (gene, len(seqs[0]), len(seqs), n_dropped, n_unique), None


def main():
    alignments = find_alignments()
    print(f"{len(alignments)} alignments found")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        results = list(ex.map(inspect, alignments.items(), chunksize=50))

    skipped = sum(r[1] is None for r in results)
    stats = [r[1] for r in results if r[1] is not None]
    selected = [s for s in stats if MIN_COLUMNS <= s[1] <= MAX_COLUMNS and s[4] >= MIN_UNIQUE_LENGTHS]
    with open(OUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene", "columns", "n_seqs", "n_dropped", "n_unique_lengths"])
        w.writerows(selected)

    print(f"{skipped} genes skipped by cleaning (too many taxa dropped, or too few left)")
    print(f"{sum(MIN_COLUMNS <= s[1] <= MAX_COLUMNS for s in stats)} with {MIN_COLUMNS}-{MAX_COLUMNS} columns")
    print(f"{len(selected)} also with >= {MIN_UNIQUE_LENGTHS} unique lengths -> {OUTPUT}")


if __name__ == "__main__":
    main()