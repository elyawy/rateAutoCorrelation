"""
Manage compressed feature archives.

Usage:
    python manage_features.py compress
    python manage_features.py decompress features_20260317_143022.parquet [--force]
    python manage_features.py combine features_20260317_143022.parquet features_20260318_090000.parquet [--force]
    python manage_features.py --info features_20260317_143022.parquet
"""

import argparse
import json
import pathlib
import sys
from datetime import datetime

import pandas as pd

FEATURES_DIR = pathlib.Path("features")
CSV_PATH = FEATURES_DIR / "features.csv"


# ==========================================
# METADATA HELPERS
# ==========================================

def build_metadata(master_seed, n_train_trees, n_sims_per_tree, alpha_range, rho_range, feature_columns):
    """Build metadata dict from explicitly provided values."""
    return {
        "created_at": datetime.now().isoformat(),
        "master_seed": str(master_seed),
        "n_train_trees": str(n_train_trees),
        "n_sims_per_tree": str(n_sims_per_tree),
        "alpha_range": json.dumps(alpha_range),
        "rho_range": json.dumps(rho_range),
        "feature_columns": json.dumps(feature_columns),
    }


def read_metadata(parquet_path):
    """Read embedded metadata from a parquet file."""
    import pyarrow.parquet as pq
    pf = pq.read_metadata(parquet_path)
    raw = pf.metadata
    # Keys/values are bytes in pyarrow
    return {k.decode(): v.decode() for k, v in raw.items() if not k.startswith(b"pandas")}


def print_metadata(parquet_path):
    """Pretty-print metadata and basic stats for a parquet file."""
    parquet_path = pathlib.Path(parquet_path)
    if not parquet_path.exists():
        print(f"Error: {parquet_path} does not exist.")
        return

    meta = read_metadata(parquet_path)
    df = pd.read_parquet(parquet_path)

    print(f"\n{'=' * 50}")
    print(f"File:         {parquet_path.name}")
    print(f"Size:         {parquet_path.stat().st_size / 1024:.1f} KB")
    print(f"Rows:         {len(df)}")
    print(f"Columns:      {len(df.columns)}")
    print(f"{'=' * 50}")
    print(f"Created:      {meta.get('created_at', 'N/A')}")
    print(f"Master seed:  {meta.get('master_seed', 'N/A')}")
    print(f"Train trees:  {meta.get('n_train_trees', 'N/A')}")
    print(f"Sims/tree:    {meta.get('n_sims_per_tree', 'N/A')}")
    print(f"Alpha range:  {meta.get('alpha_range', 'N/A')}")
    print(f"Rho range:    {meta.get('rho_range', 'N/A')}")
    feature_cols = json.loads(meta.get("feature_columns", "[]"))
    print(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")
    print(f"{'=' * 50}\n")


# ==========================================
# ID COLUMN ENCODING / DECODING
# ==========================================
# tree_000_n163      -> tree_idx (int16), n_taxa (int16)
# sim_003_a1.925_r0.784 -> sim_idx (int16), sim_alpha_str, sim_rho_str

import re

_TREE_RE = re.compile(r'^tree_(\d+)_n(\d+)$')
_SIM_RE  = re.compile(r'^sim_(\d+)_a([\d.]+)_r([\d.]+)$')


def encode_id_columns(df):
    """Replace tree/simulation string columns with compact numeric equivalents."""
    tree_parsed = df['tree'].str.extract(_TREE_RE.pattern).rename(columns={0: 'tree_idx', 1: 'n_taxa'})
    sim_parsed  = df['simulation'].str.extract(_SIM_RE.pattern).rename(columns={0: 'sim_idx', 1: 'sim_alpha_str', 2: 'sim_rho_str'})

    df = df.drop(columns=['tree', 'simulation'])
    df.insert(0, 'sim_rho_str',   sim_parsed['sim_rho_str'])
    df.insert(0, 'sim_alpha_str', sim_parsed['sim_alpha_str'])
    df.insert(0, 'sim_idx',       sim_parsed['sim_idx'].astype('int16'))
    df.insert(0, 'n_taxa',        tree_parsed['n_taxa'].astype('int16'))
    df.insert(0, 'tree_idx',      tree_parsed['tree_idx'].astype('int16'))

    return df


def decode_id_columns(df):
    """Reconstruct tree/simulation string columns from their numeric parts."""
    tree = df.apply(lambda r: f"tree_{r['tree_idx']:03d}_n{r['n_taxa']}", axis=1)
    sim  = df.apply(lambda r: f"sim_{r['sim_idx']:03d}_a{r['sim_alpha_str']}_r{r['sim_rho_str']}", axis=1)

    df = df.drop(columns=['tree_idx', 'n_taxa', 'sim_idx', 'sim_alpha_str', 'sim_rho_str'])
    df.insert(0, 'simulation', sim)
    df.insert(0, 'tree', tree)

    return df


# ==========================================
# COMMANDS
# ==========================================

def compress(master_seed, n_train_trees, n_sims_per_tree, alpha_range, rho_range):
    """Compress features.csv into a timestamped parquet archive."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} does not exist. Run 1_extract_features.py first.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = FEATURES_DIR / f"features_{timestamp}.parquet"

    if out_path.exists():
        print(f"Error: {out_path} already exists. Wait a moment and retry.")
        sys.exit(1)

    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    feature_columns = [c for c in df.columns if c not in {"tree", "simulation"}]

    print("  Encoding id columns...")
    df = encode_id_columns(df)

    meta = build_metadata(master_seed, n_train_trees, n_sims_per_tree, alpha_range, rho_range, feature_columns)
    schema = pa.Schema.from_pandas(df)
    schema = schema.with_metadata({**schema.metadata, **{k.encode(): v.encode() for k, v in meta.items()}})

    table = pa.Table.from_pandas(df, schema=schema)
    pq.write_table(table, out_path, compression="zstd")

    size_kb = out_path.stat().st_size / 1024
    print(f"Saved: {out_path.name} ({size_kb:.1f} KB)")
    print(f"Metadata embedded: seed={meta['master_seed']}, "
          f"trees={meta['n_train_trees']}, sims/tree={meta['n_sims_per_tree']}")


def decompress(filename, force=False):
    """Decompress a parquet archive back to features.csv."""
    parquet_path = FEATURES_DIR / filename

    if not parquet_path.exists():
        print(f"Error: {parquet_path} does not exist.")
        sys.exit(1)

    if CSV_PATH.exists() and not force:
        print(f"Error: {CSV_PATH} already exists. Use --force to overwrite.")
        print("       (Consider backing it up first.)")
        sys.exit(1)

    print(f"Reading {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)
    print("  Decoding id columns...")
    df = decode_id_columns(df)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved: {CSV_PATH}")


def combine(filenames, force=False):
    """Combine multiple parquet archives into features.csv."""
    if CSV_PATH.exists() and not force:
        print(f"Error: {CSV_PATH} already exists. Use --force to overwrite.")
        sys.exit(1)

    dfs = []
    for filename in filenames:
        p = FEATURES_DIR / filename
        if not p.exists():
            print(f"Error: {p} does not exist.")
            sys.exit(1)
        print(f"Reading {p.name}...")
        df = pd.read_parquet(p)
        df = decode_id_columns(df)
        print(f"  {len(df)} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    total_before = len(combined)

    combined_deduped = combined.drop_duplicates()
    n_dupes = total_before - len(combined_deduped)

    if n_dupes > 0:
        print(f"\nFound and removed {n_dupes} duplicate row(s).")
    else:
        print("\nNo duplicates found.")

    print(f"Combined total: {len(combined_deduped)} rows")
    combined_deduped.to_csv(CSV_PATH, index=False)
    print(f"Saved: {CSV_PATH}")


# ==========================================
# CLI
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Manage compressed feature archives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--info", metavar="FILENAME",
        help="Print metadata and stats for a parquet file."
    )

    subparsers = parser.add_subparsers(dest="command")

    # compress
    p_compress = subparsers.add_parser("compress", help="Compress features.csv to a timestamped parquet archive.")
    p_compress.add_argument("--master-seed", type=int, required=True, help="Master random seed used for this dataset.")
    p_compress.add_argument("--n-train-trees", type=int, required=True, help="Number of training trees.")
    p_compress.add_argument("--n-sims-per-tree", type=int, required=True, help="Number of simulations per tree.")
    p_compress.add_argument("--alpha-range", type=float, nargs=2, required=True, metavar=("MIN", "MAX"), help="Alpha range used (e.g. 0.01 5.0).")
    p_compress.add_argument("--rho-range", type=float, nargs=2, required=True, metavar=("MIN", "MAX"), help="Rho range used (e.g. 0.01 0.99).")

    # decompress
    p_decompress = subparsers.add_parser("decompress", help="Decompress a parquet archive to features.csv.")
    p_decompress.add_argument("filename", help="Parquet filename (inside features/ dir).")
    p_decompress.add_argument("--force", action="store_true", help="Overwrite existing features.csv.")

    # combine
    p_combine = subparsers.add_parser("combine", help="Combine multiple parquet archives into features.csv.")
    p_combine.add_argument("filenames", nargs="+", help="Parquet filenames (inside features/ dir).")
    p_combine.add_argument("--force", action="store_true", help="Overwrite existing features.csv.")

    args = parser.parse_args()

    if args.info:
        print_metadata(FEATURES_DIR / args.info)
        return

    if args.command == "compress":
        compress(args.master_seed, args.n_train_trees, args.n_sims_per_tree, args.alpha_range, args.rho_range)
    elif args.command == "decompress":
        decompress(args.filename, force=args.force)
    elif args.command == "combine":
        combine(args.filenames, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()