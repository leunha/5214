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
echo "Step 2: Processing downloaded data..."
mkdir -p ./processed_dataset/IXI-T1 ./processed_dataset/IXI-T2
python process_data.py

# Step 3: Train the model with 
echo "Step 3: Training Rectified Flow model (CPU-optimized)..."
python train.py \
    --t1_dir ./processed_dataset/IXI-T1 \
    --t2_dir ./processed_dataset/IXI-T2 \
    --batch_size 4 \
    --epochs 30 \
    --learning_rate 2e-4 \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_interval 5 \
    --save_interval 5 \
    --subset_ratio 0.15 \
    --image_size 128 \
    --reduced_channels

# Step 4: Evaluate the model with smaller test set
echo "Step 4: Evaluating trained model..."
python evaluate.py \
    --model_path "$CHECKPOINT_DIR/final_model.pt" \
    --t1_dir ./processed_dataset/IXI-T1 \
    --t2_dir ./processed_dataset/IXI-T2 \
    --batch_size 4 \
    --num_samples 10 \
    --num_flow_steps 50 \
    --output_dir $EVAL_DIR

# Step 5: Compare with other methods (limited samples)
echo "Step 5: Comparing with other methods (limited samples)..."
python compare_methods.py \
    --rf_model_path "$CHECKPOINT_DIR/final_model.pt" \
    --t1_dir ./processed_dataset/IXI-T1 \
    --t2_dir ./processed_dataset/IXI-T2 \
    --num_samples 10 \
    --output_dir $COMPARISON_DIR

echo "CPU-optimized experiment completed!"
echo "Results saved to $OUTPUT_DIR" 