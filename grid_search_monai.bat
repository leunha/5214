@echo off
setlocal enabledelayedexpansion

REM Define parameter ranges to search
set "num_steps=10 20 50"
@REM set "reflow_steps=0 1 2"
@REM set "learning_rates=0.1"
@REM set "num_steps=1"


REM Create results directory
set OUTPUT_DIR=grid_search_results
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%
set "ns_count=0" 
for %%n in (%num_steps%) do set /a "ns_count+=1"
set /a "total=ns_count"
REM Loop through all parameter combinations
set /a "count=0"
for %%n in (%num_steps%) do (
    set /a "count+=1"
    echo Running experiment !count! of %total%
    REM Create unique output directory for this run
    set "RUN_DIR=%OUTPUT_DIR%\step%%n"
    if not exist !RUN_DIR! mkdir !RUN_DIR!

    echo Running with parameters:
    echo Output directory: !RUN_DIR!
    echo.

    REM Run training with current parameter combination
    python train_monai.py ^
        --t1_dir ./processed_dataset/IXI-T1-full ^
        --t2_dir ./processed_dataset/IXI-T2-full ^
        --batch_size 64 ^
        --learning_rate 0.0001 ^
        --num_steps %%n ^
        --reflow_steps 2 ^
        --epochs 25 ^
        --output_dir !RUN_DIR! ^
        --use_combined_loss

    echo Finished run with current parameters
    echo =============================
    echo.
)


echo Grid search completed!
