#!/bin/bash

# Run the entire Rectified Flow for MRI Translation workflow
# This script executes the complete pipeline from data download to evaluation
# Optimized for CPU-only training within a 6-hour window

# Set up variables
OUTPUT_DIR="./results"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
EVAL_DIR="$OUTPUT_DIR/evaluation"
COMPARISON_DIR="$OUTPUT_DIR/comparison"

# Create directories
mkdir -p $OUTPUT_DIR $CHECKPOINT_DIR $EVAL_DIR $COMPARISON_DIR

# Step 1: Download IXI dataset
echo "Step 1: Downloading IXI dataset..."
python download_data.py

# Step 2: Process the data
python process_data.py --t1_dir ./IXI-T1 --t2_dir "./IXI-T2 2" --output_dir ./processed_dataset --no_registration
python 2d-dataset.py --t1_dir ./processed_dataset/IXI-T1 --t2_dir ./processed_dataset/IXI-T2 --visualize

# Step 3: Train the model with 
echo "Step 3: Training Rectified Flow model (CPU-optimized)..."
python train_monai.py --t1_dir ./processed_dataset/IXI-T1 --t2_dir ./processed_dataset/IXI-T2 --batch_size 2 --epochs 1 --num_steps 10 --features 32 64 --output_dir ./test_fix_visualization --num_workers 1 --device cpu --reflow_steps 0

# Step 4: Evaluate the model with smaller test set
echo "Step 4: Evaluating trained model..."
python evaluate_monai.py --t1_dir ./processed_dataset/IXI-T1 --t2_dir ./processed_dataset/IXI-T2 --model_path ./monai_results_middle30/run_20250404_145820/checkpoint_epoch_3.pt --output_dir ./test_fix_identical_metrics --test_subset 3 --device cpu
