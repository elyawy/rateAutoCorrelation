import pathlib
import pandas as pd

def load_ground_truth(simulated_data_path: pathlib.Path) -> pd.DataFrame:
    """
    Load all ground truth files and combine them.
    
    Returns:
        DataFrame with columns: tree, simulation, true_alpha, true_rho
    """
    all_truth = []
    
    
    for tree_dir in sorted(simulated_data_path.iterdir()):
        if not tree_dir.is_dir():
            continue
        
        tree_name = tree_dir.name
        truth_file = tree_dir / 'ground_truth.csv'
        
        if not truth_file.exists():
            continue
        
        df = pd.read_csv(truth_file)
        
        # Extract simulation name from filename (remove .phy extension)
        df['simulation'] = df['filename'].str.replace('.phy', '', regex=False)
        df['tree'] = tree_name
        
        # Select relevant columns
        df = df[['tree', 'simulation', 'true_alpha', 'true_rho']]
        
        all_truth.append(df)
    
    return pd.concat(all_truth, ignore_index=True)


# print("Data loader module loaded.")
# # Test the function (this line can be removed in production)
# data_dir_path = pathlib.Path("../simulated_data")
# df = load_ground_truth(data_dir_path)  # Test loading function
# print(df.head())