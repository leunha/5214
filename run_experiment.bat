@echo off
REM Run the entire Rectified Flow for MRI Translation workflow
REM This script executes the complete pipeline from data download to evaluation
REM Optimized for CPU-only training within a 6-hour window

REM Set up variables
set OUTPUT_DIR=.\results
set CHECKPOINT_DIR=%OUTPUT_DIR%\checkpoints
set EVAL_DIR=%OUTPUT_DIR%\evaluation
set COMPARISON_DIR=%OUTPUT_DIR%\comparison

REM Create directories
mkdir %OUTPUT_DIR% %CHECKPOINT_DIR% %EVAL_DIR% %COMPARISON_DIR% 2>nul

REM Step 1: Download IXI dataset
echo Step 1: Downloading IXI dataset...
python download_data.py

REM Step 2: Process the data
python process_data.py --t1_dir .\IXI-T1 --t2_dir ".\IXI-T2 2" --output_dir .\processed_dataset --no_registration
python 2d-dataset.py --t1_dir .\processed_dataset\IXI-T1 --t2_dir .\processed_dataset\IXI-T2 --visualize

REM Step 3: Train the model
echo Step 3: Training Rectified Flow model (CPU-optimized)...
python train_monai.py --t1_dir .\processed_dataset\IXI-T1-full --t2_dir .\processed_dataset\IXI-T2-full --batch_size 64 --epochs 20 --num_steps 100 --features 64 128 256 256 --output_dir .\test_fix_visualization --num_workers 1 --reflow_steps 0

REM Step 4: Evaluate the model with smaller test set
echo Step 4: Evaluating trained model...
python evaluate_monai.py --t1_dir .\processed_dataset\IXI-T1-full --t2_dir .\processed_dataset\IXI-T2-full --model_path .\monai_results_middle30\run_20250404_145820\checkpoint_epoch_3.pt --output_dir .\test_fix_identical_metrics --test_subset 3
