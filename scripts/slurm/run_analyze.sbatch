#!/bin/bash
#SBATCH --job-name=run_analysis_job       
#SBATCH --account=kwang222-ic         
#SBATCH --partition=bioshare
#SBATCH --time=12:00:00                    
#SBATCH --nodes=1                          
#SBATCH --ntasks-per-node=64         
#SBATCH --mem=160G
#SBATCH --gres=gpu:A40:1
#SBATCH --output=run_analysis_job    
     
cd "${SLURM_SUBMIT_DIR}"

module load anaconda3
source $(conda info --base)/etc/profile.d/conda.sh

conda activate vae_env

# Run script
python analyze_embeddings.py
