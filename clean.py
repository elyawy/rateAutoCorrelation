"""
Clean up all generated data directories.
Run this before regenerating the entire pipeline.
"""

import shutil
import pathlib
import config

def clean():
    """Remove all generated directories."""
    dirs_to_remove = [
        config.SIMULATED_DATA_DIR,
        config.CODEML_RUNS_DIR,
        config.RESULTS_DIR
    ]
    
    for dir_name in dirs_to_remove:
        dir_path = pathlib.Path(dir_name)
        if dir_path.exists():
            print(f"Removing {dir_path}/")
            shutil.rmtree(dir_path)
        else:
            print(f"Skipping {dir_path}/ (does not exist)")
    
    print("\nCleanup complete!")

if __name__ == "__main__":
    response = input("This will delete all generated data. Continue? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        clean()
    else:
        print("Cleanup cancelled.")
