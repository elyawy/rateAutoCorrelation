"""
Step 3: Extract inferred alpha and rho from baseml output files.

Parse all mlb files and extract:
  - Tree name
  - Simulation ID
  - Inferred alpha
  - Inferred rho

Save to results/inferred_parameters.csv
"""

import pathlib
import re
import pandas as pd
import config

def parse_baseml_output(mlb_file):
    """
    Parse a baseml output file and extract alpha and rho.
    
    Returns:
        tuple: (alpha, rho) or (None, None) if parsing fails
    """
    try:
        with open(mlb_file, 'r') as f:
            content = f.read()
        
        # Look for alpha in the output
        # Pattern: "alpha (gamma)" or similar
        alpha_match = re.search(r'alpha\s*\(gamma\)\s*=\s*([\d.]+)', content, re.IGNORECASE)
        
        # Look for rho in the output
        # Pattern: "rho (c)" or similar
        rho_match = re.search(r'rho\s*\(c\)\s*=\s*([\d.]+)', content, re.IGNORECASE)
        
        alpha = float(alpha_match.group(1)) if alpha_match else None
        rho = float(rho_match.group(1)) if rho_match else None
        
        return alpha, rho
    
    except Exception as e:
        print(f"    Error parsing {mlb_file}: {e}")
        return None, None

def extract_all_parameters(baseml_runs_dir):
    """
    Extract parameters from all baseml runs.
    
    Returns:
        list: List of dicts with tree, simulation, alpha, rho
    """
    results = []
    
    # Iterate through all tree directories
    for tree_dir in sorted(baseml_runs_dir.iterdir()):
        if not tree_dir.is_dir():
            continue
        
        tree_name = tree_dir.name
        print(f"\nProcessing tree: {tree_name}")
        
        # Iterate through all simulation runs
        sim_dirs = sorted(tree_dir.iterdir())
        for sim_dir in sim_dirs:
            if not sim_dir.is_dir():
                continue
            
            sim_name = sim_dir.name
            mlb_file = sim_dir / 'mlb'
            
            if not mlb_file.exists():
                print(f"  WARNING: Output file not found: {mlb_file}")
                continue
            
            # Parse the output
            alpha, rho = parse_baseml_output(mlb_file)
            
            if alpha is None or rho is None:
                print(f"  WARNING: Could not extract parameters from {sim_name}")
                continue
            
            results.append({
                'tree': tree_name,
                'simulation': sim_name,
                'inferred_alpha': alpha,
                'inferred_rho': rho
            })
        
        print(f"  Extracted parameters from {len([r for r in results if r['tree'] == tree_name])} simulations")
    
    return results

def main():
    """Main function to extract all parameters."""
    baseml_runs_dir = pathlib.Path(config.BASEML_RUNS_DIR)
    results_dir = pathlib.Path(config.RESULTS_DIR)
    
    if not baseml_runs_dir.exists():
        print(f"Error: {baseml_runs_dir}/ does not exist. Run step 2 first.")
        return
    
    print("Extracting parameters from baseml outputs...")
    print("=" * 50)
    
    # Extract all parameters
    results = extract_all_parameters(baseml_runs_dir)
    
    if not results:
        print("\nError: No results extracted. Check baseml outputs.")
        return
    
    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(results)
    output_file = results_dir / 'inferred_parameters.csv'
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 50)
    print(f"Extraction complete!")
    print(f"Total parameters extracted: {len(results)}")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    main()
