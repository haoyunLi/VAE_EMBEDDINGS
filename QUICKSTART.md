# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd VAE_embeddings
python -m venv vae_env
source vae_env/bin/activate  # On Windows: vae_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Your Data
```bash
# Place your gene expression data in the data/ directory
# Expected format: CSV with samples as rows, genes as columns
cp your_gene_expression_data.csv data/filtered_gtex_coding_genes.csv
```

### 3. Train the Model
```bash
# Quick training with default parameters
python run.py --mode train

# Or run directly
python src/training/train_vae.py
```

### 4. Analyze Results
```bash
# Analyze embeddings
python run.py --mode analyze

# Evaluate model performance
python run.py --mode evaluate
```

## 📊 Expected Outputs

After training, you'll find:
- **`models/vae_model.pth`**: Trained model
- **`checkpoints/`**: Training checkpoints
- **`training_plots/`**: Loss curves and metrics
- **`logs/`**: Training logs

## 🔧 Configuration

Edit `config/train_config.py` to customize:
- Model architecture (hidden dimensions, latent size)
- Training parameters (epochs, learning rate)
- KL annealing schedule
- Dynamic masking settings

## 🎯 Key Features Enabled by Default

✅ **ELU activations** - Optimized for gene expression data  
✅ **Beta-VAE KL annealing** - Better disentanglement  
✅ **Dynamic masking** - Improved decoder robustness  
✅ **Early stopping** - Prevents overfitting  
✅ **Checkpointing** - Resume training anytime  

## 💡 Tips

- **GPU Training**: Automatically uses GPU if available
- **Memory Issues**: Reduce `batch_size` in config
- **Poor Convergence**: Adjust KL annealing parameters
- **Custom Data**: Ensure CSV format with numeric values only

## 📈 Monitoring Training

Watch training progress:
```bash
tail -f logs/vae_training.log
```

View loss curves in `training_plots/` directory (updated every 5 epochs).

## 🆘 Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch_size to 64 or 32 |
| Import errors | Run `pip install -r requirements.txt` |
| Poor reconstruction | Check data preprocessing and scaling |
| Slow convergence | Increase learning rate or adjust KL weight |

Ready to explore gene expression embeddings! 🧬 