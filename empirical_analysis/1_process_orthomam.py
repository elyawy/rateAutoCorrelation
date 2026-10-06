"""
Step 1: Process OrthoMaM dataset and infer rho.

For each (nucleotide) FASTA file in the OrthoMaM archive:
  - Pick the largest window (500/250/100) that fits the alignment
  - Cut non-overlapping chunks, run the matching CNN regressor on each
  - Average the chunk predictions into one rho per gene
  - Save results to CSV with metadata
"""

import pathlib
import zipfile
import pandas as pd
import torch
import numpy as np
from io import StringIO
from Bio import SeqIO
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "training"))
sys.path.insert(0, str(REPO_DIR / "common"))

import common.config as config

# common/ goes first so its features_calculator wins over the one in this folder
from features_calculator import calculate_cnn_input, calculate_scalar_features
from train_rho_regressor import RhoRegressor

WINDOWS = [500, 250, 100]  # tried in this order, biggest that fits


def extract_fasta_from_zip(zip_path):
    """
    Extract FASTA file from a zip archive.
    
    Args:
        zip_path: Path to .fasta.zip file
        
    Returns:
        tuple: (filename, fasta_content_string) or (None, None) if error
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in the archive
            file_list = zip_ref.namelist()
            
            # Find FASTA file (should be only one)
            fasta_files = [f for f in file_list if f.endswith('.fasta')]
            
            if len(fasta_files) == 0:
                print(f"  WARNING: No FASTA file found in {zip_path.name}")
                return None, None
            
            if len(fasta_files) > 1:
                print(f"  WARNING: Multiple FASTA files in {zip_path.name}, using first")
            
            fasta_file = fasta_files[0]
            
            # Read FASTA content
            with zip_ref.open(fasta_file) as f:
                content = f.read().decode('utf-8')
            
            # Extract just the filename without path
            filename = pathlib.Path(fasta_file).name
            
            return filename, content
    
    except Exception as e:
        print(f"  ERROR extracting {zip_path.name}: {e}")
        return None, None


def parse_fasta_content(fasta_content):
    """
    Parse FASTA content string into list of sequences.
    
    Args:
        fasta_content: String containing FASTA formatted sequences
        
    Returns:
        list: List of sequence strings (without headers)
    """
    sequences = []
    handle = StringIO(fasta_content)
    
    for record in SeqIO.parse(handle, "fasta"):
        sequences.append(str(record.seq))
    
    return sequences


def calculate_alignment_metadata(sequences):
    """
    Calculate basic metadata about the alignment.
    
    Returns:
        dict: n_sequences, alignment_length, n_variable_sites
    """
    if not sequences:
        return {
            'n_sequences': 0,
            'alignment_length': 0,
            'n_variable_sites': 0
        }
    
    n_sequences = len(sequences)
    alignment_length = len(sequences[0]) if sequences else 0
    
    # Count variable sites (sites with entropy > 0)
    # We'll use a simple heuristic: column has >1 unique character
    n_variable = 0
    for col_idx in range(alignment_length):
        column = [seq[col_idx] for seq in sequences]
        # Remove gaps and count unique residues
        unique_residues = set(c for c in column if c != '-')
        if len(unique_residues) > 1:
            n_variable += 1
    
    return {
        'n_sequences': n_sequences,
        'alignment_length': alignment_length,
        'n_variable_sites': n_variable
    }


def load_models():
    """Load one trained regressor per window size from the repo root."""
    models = {}
    for window in WINDOWS:
        model = RhoRegressor()
        model.load_state_dict(torch.load(REPO_DIR / f"rho_regressor_w{window}.pt", map_location="cpu"))
        model.eval()
        models[window] = model
    return models


def is_dna(sequences, n_check=20):
    """True if the residues of the first sequences are mostly A/C/G/T."""
    chars = "".join(sequences[:n_check]).upper().replace("-", "")
    return len(chars) > 0 and sum(chars.count(b) for b in "ACGT") / len(chars) >= 0.9


def predict_rho(sequences, models):
    """Chunk the alignment into non-overlapping windows and average the predictions."""
    length = len(sequences[0])
    window = next((w for w in WINDOWS if length >= w), None)
    if window is None:
        return None

    chunks, scalars = [], []
    for start in range(0, length - window + 1, window):
        chunk = [seq[start:start + window] for seq in sequences]
        chunks.append(calculate_cnn_input(chunk, alphabet_size=4))
        scalars.append(calculate_scalar_features(chunk, alphabet_size=4))

    with torch.no_grad():
        preds = models[window](torch.stack(chunks), torch.stack(scalars)).numpy()

    return {
        'pred_rho': float(preds.mean()),
        'rho_std': float(preds.std()),
        'window_used': window,
        'n_chunks': len(chunks),
    }


def process_single_zip(zip_path, models):
    """
    Process a single zip file: extract, chunk, predict rho.
    
    Args:
        zip_path: Path to .fasta.zip file
        models: dict of trained regressors, keyed by window size
        
    Returns:
        dict: Results dictionary or None if error
    """
    try:
        # Extract FASTA from zip
        filename, fasta_content = extract_fasta_from_zip(zip_path)
        
        if filename is None:
            return None
        
        # Parse sequences
        sequences = parse_fasta_content(fasta_content)
        
        if not sequences:
            print(f"  WARNING: No sequences found in {filename}")
            return None
        
        # Calculate metadata
        metadata = calculate_alignment_metadata(sequences)
        
        if not is_dna(sequences):
            print(f"  WARNING: {filename} does not look like DNA, skipping")
            return None
        
        # Predict rho
        prediction = predict_rho(sequences, models)
        if prediction is None:
            print(f"  WARNING: {filename} shorter than {min(WINDOWS)} sites, skipping")
            return None
        
        return {'filename': filename, **metadata, **prediction}
    
    except Exception as e:
        print(f"  ERROR processing {zip_path.name}: {e}")
        return None


def main():
    """Main processing function."""
    print("=" * 60)
    print("ORTHOMAM EMPIRICAL ANALYSIS")
    print("=" * 60)
    print(f"OrthoMaM directory: {config.ORTHOMAM_DIR}")
    print()
    
    # Check paths exist
    if not config.ORTHOMAM_DIR.exists():
        print(f"ERROR: OrthoMaM directory not found: {config.ORTHOMAM_DIR}")
        return
    
    # Load trained models
    print("Loading trained models...")
    models = load_models()
    print("  Models loaded successfully")
    print()
    
    # Find all zip files
    zip_files = sorted(config.ORTHOMAM_DIR.glob("*.fasta.zip"))
    
    if not zip_files:
        print(f"ERROR: No .fasta.zip files found in {config.ORTHOMAM_DIR}")
        return
    
    print(f"Found {len(zip_files)} zip files to process")
    print()
    
    # Determine number of workers
    n_workers = config.N_WORKERS or os.cpu_count()
    print(f"Using {n_workers} parallel workers")
    print()
    
    # Process files
    print("Processing files...")
    print("-" * 60)
    
    results = []
    
    # Process sequentially (simpler, can be parallelized later if needed)
    # Note: We process sequentially because loading the model in each worker
    # would be memory-intensive. For better parallelization, we'd need to
    # use shared memory or a different approach.
    
    for idx, zip_path in enumerate(zip_files, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"Processing {idx}/{len(zip_files)}: {zip_path.name}")
        
        result = process_single_zip(zip_path, models)
        
        if result is not None:
            results.append(result)
    
    print()
    print("-" * 60)
    print(f"Successfully processed {len(results)}/{len(zip_files)} files")
    
    if not results:
        print("ERROR: No results to save")
        return
    
    # Create results directory
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(results)
    output_file = config.RESULTS_DIR / "orthomam_predictions.csv"
    df.to_csv(output_file, index=False)
    
    print()
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_file}")
    print()
    print("Summary statistics:")
    print(f"  Rho   - Mean: {df['pred_rho'].mean():.3f}, Std: {df['pred_rho'].std():.3f}")
    print(f"  Window used: {df['window_used'].value_counts().to_dict()}")
    print()
    print("Next step: Run python 2_plot_distributions.py")


if __name__ == "__main__":
    main()