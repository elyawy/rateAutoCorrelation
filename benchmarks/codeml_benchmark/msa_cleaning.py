"""
Shared cleaning of the unfiltered OrthoMaM protein alignments, used by find_gappy_msas.py and
estimate_selected_trees.py so both see identical alignments.

Rules:
  1. drop taxa containing 'X', '!' or an internal '*' (a '*' that is not the last non-gap character)
  2. a terminal '*' becomes '-'
  3. optionally keep only the taxa in `allowed` (e.g. those in the tree)
  4. remove columns that are all gaps afterwards
  5. skip the gene if rule 1 dropped more than MAX_DROPPED_FRACTION of its taxa, or fewer than MIN_TAXA remain
"""

import zipfile
from pathlib import Path

import numpy as np

AA_DIR = Path("/home/elyalab/temp/orthomam_AA_unfiltered")  # *.aln.zip or *.aln, searched recursively
MAX_DROPPED_FRACTION = 0.2
MIN_TAXA = 4


def gene_id(path: Path) -> str:
    return path.name.split("_")[0]


def find_alignments() -> dict:
    """{gene: path} for every alignment under AA_DIR."""
    paths = [*AA_DIR.rglob("*.aln.zip"), *AA_DIR.rglob("*.aln")]
    return {gene_id(p): p for p in sorted(paths)}


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


def clean_alignment(records: list, allowed: set = None):
    """
    Returns ({taxon: cleaned sequence}, n_special_dropped, None) or (None, n_special_dropped, reason).
    The dropped fraction is computed over all records, whether or not they are in `allowed`.
    """
    good, n_bad = {}, 0
    for name, seq in records:
        seq = seq.upper()
        ungapped = seq.replace("-", "")
        if "X" in seq or "!" in seq or "*" in ungapped[:-1]:
            n_bad += 1
            continue
        if ungapped.endswith("*"):
            i = seq.rindex("*")
            seq = seq[:i] + "-" + seq[i + 1:]
        good[name] = seq

    if n_bad / len(records) > MAX_DROPPED_FRACTION:
        return None, n_bad, f"{n_bad}/{len(records)} taxa dropped for X, ! or internal *"
    if allowed is not None:
        good = {t: s for t, s in good.items() if t in allowed}
    if len(good) < MIN_TAXA:
        return None, n_bad, f"only {len(good)} taxa remain"

    names = list(good)
    arr = np.stack([np.frombuffer(good[t].encode(), dtype=np.uint8) for t in names])
    arr = arr[:, (arr != ord("-")).any(axis=0)]
    return {t: row.tobytes().decode() for t, row in zip(names, arr)}, n_bad, None
