"""
Step 1: Process OrthoMaM dataset and infer alpha/rho parameters.

For each FASTA file in the OrthoMaM archive:
  - Extract features using the trained feature calculator
  - Load trained neural network model
  - Predict alpha and rho parameters
  - Save results to CSV with metadata
"""

import pathlib
import zipfile
import pandas as pd
import joblib
import numpy as np
from io import StringIO
from Bio import SeqIO
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import common.config as config

# Import feature calculator from inference pipeline
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "inference_pipeline"))
from features_calculator import calculate_msa_entropy_stats, read_fasta_sequences


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


def process_single_zip(zip_path, model):
    """
    Process a single zip file: extract, calculate features, predict parameters.
    
    Args:
        zip_path: Path to .fasta.zip file
        model: Trained neural network model
        
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
        
        # Calculate features
        features = calculate_msa_entropy_stats(sequences)
        
        # Extract feature vector in correct order
        feature_vector = np.array([features[col] for col in config.FEATURE_COLUMNS])
        feature_vector = feature_vector.reshape(1, -1)  # Shape: (1, n_features)
        
        # Predict alpha and rho
        predictions = model.predict(feature_vector)
        pred_alpha = float(predictions[0, 0])
        pred_rho = float(predictions[0, 1])
        
        # Combine results
        result = {
            'filename': filename,
            **metadata,
            'pred_alpha': pred_alpha,
            'pred_rho': pred_rho
        }
        
        return result
    
    except Exception as e:
        print(f"  ERROR processing {zip_path.name}: {e}")
        return None


def main():
    """Main processing function."""
    print("=" * 60)
    print("ORTHOMAM EMPIRICAL ANALYSIS")
    print("=" * 60)
    print(f"OrthoMaM directory: {config.ORTHOMAM_DIR}")
    print(f"Model path: {config.MODEL_PATH}")
    print()
    
    # Check paths exist
    if not config.ORTHOMAM_DIR.exists():
        print(f"ERROR: OrthoMaM directory not found: {config.ORTHOMAM_DIR}")
        return
    
    if not config.MODEL_PATH.exists():
        print(f"ERROR: Model file not found: {config.MODEL_PATH}")
        return
    
    # Load trained model
    print("Loading trained model...")
    model = joblib.load(config.MODEL_PATH)
    print("  Model loaded successfully")
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
        
        result = process_single_zip(zip_path, model)
        
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
    print(f"  Alpha - Mean: {df['pred_alpha'].mean():.3f}, Std: {df['pred_alpha'].std():.3f}")
    print(f"  Rho   - Mean: {df['pred_rho'].mean():.3f}, Std: {df['pred_rho'].std():.3f}")
    print()
    print("Next step: Run python 2_plot_distributions.py")


if __name__ == "__main__":
    main()
