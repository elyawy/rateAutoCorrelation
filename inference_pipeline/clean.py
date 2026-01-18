"""
Clean up generated data in the inference pipeline.
Run this to start fresh.
"""

import shutil
import pathlib

def clean():
    """Remove all generated directories."""
    dirs_to_remove = [
        "training_data",
        "features", 
        "results"
    ]
    
    for dir_name in dirs_to_remove:
        dir_path = pathlib.Path(dir_name)
        if dir_path.exists():
            print(f"Removing {dir_path}/")
            shutil.rmtree(dir_path)
        else:
            print(f"Skipping {dir_path}/ (does not exist)")
    
    print("\nCleanup complete!")
    print("Note: Trained models in models/ are preserved.")

if __name__ == "__main__":
    response = input("This will delete training_data, features, and results. Continue? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        clean()
    else:
        print("Cleanup cancelled.")