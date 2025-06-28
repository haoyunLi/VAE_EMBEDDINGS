#!/bin/bash
#SBATCH --job-name=vae_training_direct       
#SBATCH --account=kwang222-ic         
#SBATCH --partition=bioshare
#SBATCH --time=12:00:00                    
#SBATCH --nodes=1                          
#SBATCH --ntasks-per-node=64         
#SBATCH --mem=160G
#SBATCH --gres=gpu:A40:1
#SBATCH --output=vae_training_direct_job  
     
# Navigate to project root directory using absolute path
PROJECT_ROOT="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/VAE_embeddings"
cd "${PROJECT_ROOT}"

# Verify we're in the right directory
echo "Working directory: $(pwd)"

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh

conda activate vae_env

# Run training script directly (bypassing run.py)
python -m src.training.train_vae 