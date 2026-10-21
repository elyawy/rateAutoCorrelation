"""
Realign the simulated MSAs with MAFFT (FFT-NS-1, via the itaxotools-mafftpy package) to measure
how much alignment error alone changes the rho estimates.

For each sim_msas/{gene}.fasta: drop the gaps, realign, write sim_msas_mafft/{gene}.fasta.
Trees are unchanged (sim_trees/), so the realigned MSAs can go straight into 2_run_codeml_real.py:
    python 2_run_codeml_real.py --tree-dir sim_trees --msa-dir sim_msas_mafft --runs-dir codeml_runs_sim_mafft
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor

IN_DIR = pathlib.Path("sim_msas")
OUT_DIR = pathlib.Path("sim_msas_mafft")
MAFFT_CMD = "mafftpy-fftns1"  # console script installed by: pip install itaxotools-mafftpy


def read_fasta(path):
    records, name, cur = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(cur)))
            name, cur = line[1:].strip(), []
        else:
            cur.append(line.strip())
    records.append((name, "".join(cur)))
    return records


def realign(gene, in_dir, out_dir):
    out_path = out_dir / f"{gene}.fasta"
    if out_path.exists():
        return gene
    records = read_fasta(in_dir / f"{gene}.fasta")
    with tempfile.TemporaryDirectory() as tmp:
        unaligned, aligned = pathlib.Path(tmp) / "in.fasta", pathlib.Path(tmp) / "out.fasta"
        unaligned.write_text("".join(f">{n}\n{s.replace('-', '')}\n" for n, s in records))
        subprocess.run([shutil.which(MAFFT_CMD) or MAFFT_CMD, str(unaligned), str(aligned)],
                       check=True, capture_output=True)
        new = read_fasta(aligned)

    # MAFFT keeps the sequence names; check nothing was lost, then keep the input order
    by_name = {n: s.upper() for n, s in new}
    assert set(by_name) == {n for n, _ in records}, f"{gene}: sequences changed during realignment"
    out_path.write_text("".join(f">{n}\n{by_name[n]}\n" for n, _ in records))
    return gene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=pathlib.Path, default=IN_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--cores", type=int, default=os.cpu_count())
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    genes = sorted(p.stem for p in args.in_dir.glob("*.fasta"))
    print(f"{len(genes)} genes, {args.cores} cores")
    with ProcessPoolExecutor(max_workers=args.cores) as ex:
        futures = [ex.submit(realign, g, args.in_dir, args.out_dir) for g in genes]
        for i, fut in enumerate(futures, start=1):
            fut.result()
            if i % 20 == 0 or i == len(genes):
                print(f"  {i}/{len(genes)}")


if __name__ == "__main__":
    main()
