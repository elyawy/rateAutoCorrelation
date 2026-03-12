#!/bin/bash
# Packages all necessary files for server-side training into server_upload.zip
# Run from the project root directory.

set -e

STAGING="server_upload/inference_pipeline"

echo "Creating staging directory..."
rm -rf server_upload
mkdir -p "$STAGING/models"
mkdir -p "$STAGING/utils"
mkdir -p "$STAGING/features"

echo "Copying scripts and config..."
cp inference_pipeline/config.py                "$STAGING/"
cp inference_pipeline/1_extract_features.py    "$STAGING/"
cp inference_pipeline/2_train_models.py        "$STAGING/"
cp inference_pipeline/3_evaluate.py            "$STAGING/"
cp inference_pipeline/features_calculator.py   "$STAGING/"
cp inference_pipeline/train_gpu.sh             "$STAGING/"
cp inference_pipeline/models/random_forest.py  "$STAGING/models/"
cp inference_pipeline/models/neural_net.py     "$STAGING/models/"
cp inference_pipeline/utils/data_loader.py     "$STAGING/utils/"
cp inference_pipeline/__init__.py              "$STAGING/"

echo "Copying features..."
cp inference_pipeline/features/features.csv    "$STAGING/features/"

echo "Copying ground truth files..."
for tree_dir in inference_pipeline/training_data/*/; do
    tree_name=$(basename "$tree_dir")
    mkdir -p "$STAGING/training_data/$tree_name"
    cp "$tree_dir/ground_truth.csv" "$STAGING/training_data/$tree_name/"
done

echo "Creating zip archive..."
zip -r server_upload.zip server_upload/

echo ""
echo "Done! Archive ready: server_upload.zip"
echo "Contents summary:"
echo "  Scripts:       $(find $STAGING -maxdepth 1 -name '*.py' | wc -l) Python files"
echo "  Trees:         $(ls $STAGING/training_data/ | wc -l) ground_truth.csv files"
echo "  Features:      $(wc -l < $STAGING/features/features.csv) rows (including header)"
