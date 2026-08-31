"""
Rename taxa in all tree files to standardized short IDs.

Creates:
  - trees_renamed/ directory with modified trees
  - {tree_name}_mapping.csv for each tree (original_name -> short_id)
"""

import pathlib
from ete3 import Tree
import csv

def rename_tree_taxa(tree_path, output_dir):
    """
    Rename all taxa in a tree to short IDs (s01, s02, etc.).
    
    Returns the mapping dictionary.
    """
    tree_name = tree_path.stem
    print(f"\nProcessing: {tree_name}")
    
    # Load tree
    tree = Tree(str(tree_path), format=1)
    
    # Get all leaf names
    leaf_names = [leaf.name for leaf in tree.get_leaves()]
    print(f"  Found {len(leaf_names)} taxa")
    
    # Create mapping
    mapping = {}
    for i, original_name in enumerate(sorted(leaf_names), start=1):
        short_id = f"s{i:02d}"  # s01, s02, ..., s99, s100, etc.
        mapping[original_name] = short_id
    
    # Rename leaves in tree
    for leaf in tree.get_leaves():
        leaf.name = mapping[leaf.name]
    
    # Save renamed tree
    output_tree_path = output_dir / f"{tree_name}.newick"
    tree.write(outfile=str(output_tree_path), format=1)
    print(f"  Saved renamed tree: {output_tree_path}")
    
    # # Save mapping
    # mapping_path = output_dir / f"{tree_name}_mapping.csv"
    # with open(mapping_path, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['original_name', 'short_id'])
    #     for original, short in sorted(mapping.items()):
    #         writer.writerow([original, short])
    # print(f"  Saved mapping: {mapping_path}")
    

def main():
    """Process all tree files."""
    trees_dir = pathlib.Path("trees")
    output_dir = pathlib.Path("trees_renamed")
    
    if not trees_dir.exists():
        print(f"Error: {trees_dir}/ does not exist")
        return
    
    # Find all tree files
    tree_files = sorted(trees_dir.glob("*.newick"))
    
    if not tree_files:
        print(f"Error: No .newick files found in {trees_dir}/")
        return
    
    print(f"Found {len(tree_files)} tree file(s)")
    print("=" * 50)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Process each tree
    for tree_path in tree_files:
        rename_tree_taxa(tree_path, output_dir)
    
    print("\n" + "=" * 50)
    print("All trees renamed successfully!")
    print(f"\nNext steps:")
    print(f"  1. Rename 'trees/' to 'old_trees/'")
    print(f"  2. Rename 'trees_renamed/' to 'trees/'")
    print(f"  3. Run the pipeline normally")

if __name__ == "__main__":
    main()