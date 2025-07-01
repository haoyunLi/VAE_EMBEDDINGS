#!/bin/bash
#SBATCH --job-name=run_training_job       
#SBATCH --account=kwang222-ic         
#SBATCH --partition=bioshare
#SBATCH --time=12:00:00                    
#SBATCH --nodes=1                          
#SBATCH --ntasks-per-node=64         
#SBATCH --mem=160G
#SBATCH --gres=gpu:A40:1
#SBATCH --output=run_training_job  
     
# Navigate to project root directory using absolute path
PROJECT_ROOT="/taiga/illinois/vetmed/cb/kwang222/Haoyun_Li/VAE_embeddings"
cd "${PROJECT_ROOT}"

# Verify we're in the right directory
echo "Working directory: $(pwd)"
echo "Contents: $(ls -la run.py 2>/dev/null || echo 'run.py not found')"

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh

conda activate vae_env

# Run script using the unified entry point (training only, no preprocessing)
python run.py --mode train    