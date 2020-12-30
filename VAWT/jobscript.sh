#!/bin/sh
#BSUB -q gpuv100
#BSUB -J vawt
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -R "rusage[mem=5GB]"
#BSUB -u <user> # dtu username e.g. sxxxxxx@student.dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err

# -- run --
module load cuda/10.2
module load numpy/1.18.2-python-3.7.7-openblas-0.3.9
python3 -m pip install pyCuda --user
nvidia-smi
python3 ./SHSLBM_vawt_gpu.py
