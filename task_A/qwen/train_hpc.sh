#!/bin/bash
#SBATCH --job-name=semeval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<ADDYOUREMAIL>
#SBATCH --partition=gpu_a40           # choose between gpu_a40, gpu_a40_ext, gpu_a100, cpu_sapphire, cpu_sapphire_ext
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:00:00             # Max time
#SBATCH --cpus-per-task=8             # CPUs per task
#SBATCH --mem=64G                     # Memory required per node

module purge
module load miniforge/24.3.0-0
module load nvhpc/25.1
module load gcc/12.4.0

eval "$(conda shell.bash hook)"
# before running this script, make sure you have created the conda environment
conda activate semeval_env

python train.py