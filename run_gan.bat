@echo off
REM Grid search for CycleGAN training

REM Define parameter grids
REM Path to your training script
mkdir ./cycle_gan_results

REM Loop over all combinations
echo Training with lr=0.0001, batch_size=8, epochs=100 CycleGAN
python cyclegan.py --t1_dir ./processed_dataset/IXI-T1-full --t2_dir ./processed_dataset/IXI-T2-full --batch_size 64 --epochs 10 --output_dir ./cycle_gan_results