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
python train_monai.py --t1_dir ./processed_dataset/IXI-T1 --t2_dir ./processed_dataset/IXI-T2 --batch_size 2 --epochs 5 --num_steps 20 --features 32 64 128 --reflow_steps 3 --use_combined_loss
REM option 2: another model
python efficient_pretrained_flow.py --t1_dir ./processed_dataset/IXI-T1 --t2_dir ./processed_dataset/IXI-T2 --epochs 1 --batch_size 2 --freeze_ratio 0.95 --output_dir ./quick_pretrained_results

REM option 3: another training
python simple_pretrained_demo.py --t1_dir ./processed_dataset/IXI-T1 --t2_dir ./processed_dataset/IXI-T2 --num_steps 50 --batch_size 2 --output_dir ./demo_results



REM Step 4: Evaluate the model with smaller test set
echo Step 4: Evaluating trained model...
python evaluate_monai.py --t1_dir .\processed_dataset\IXI-T1-full --t2_dir .\processed_dataset\IXI-T2-full --model_path .\monai_results_middle30\run_20250404_145820\checkpoint_epoch_3.pt --output_dir .\test_fix_identical_metrics --test_subset 3
