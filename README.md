# VAE Gene Expression Embeddings

A comprehensive Variational Autoencoder (VAE) implementation for learning latent representations of high-dimensional gene expression data from GTEx dataset.

## 🧬 Project Overview

This project implements a sophisticated VAE architecture specifically designed for gene expression data analysis, featuring:

- **Beta-VAE style KL annealing** for better disentanglement
- **Dynamic masking** for robust decoder training
- **ELU activation functions** optimized for standardized gene expression data
- **Comprehensive evaluation metrics** and visualization tools
- **Contrastive learning** extensions for improved embeddings

## 📁 Project Structure

```
VAE_embeddings/
├── config/                      # Configuration files
│   └── train_config.py          # Centralized training configuration
├── src/                         # Core source code
│   ├── models/                  # Model architectures
│   │   └── vae_model.py         # Main VAE implementation
│   ├── training/                # Training scripts
│   │   ├── train_vae.py         # Standard VAE training
│   │   └── contrastive_vae_training.py  # Contrastive learning
│   ├── evaluation/              # Model evaluation tools
│   │   └── evaluate_vae.py      # Comprehensive evaluation metrics
│   └── utils/                   # Utility functions
│       ├── prepossing.py        # Data preprocessing
│       ├── transpose_data.py    # Data transformation utilities
│       ├── encode.py            # Encoding utilities
│       └── new_pseudobulk.py    # Pseudobulk generation
├── scripts/                     # Execution scripts
│   ├── slurm/                   # SLURM job scripts and batch files
│   │   ├── run_training.sbatch  # Main VAE training job
│   │   ├── run_contrastive.sbatch # Contrastive learning job
│   │   ├── run_embeddings.sbatch # Embedding generation job
│   │   ├── run_analyze.sbatch   # Analysis job
│   │   └── ...                  # Other job scripts
│   └── analysis/                # Analysis scripts
│       ├── analyze_embeddings.py        # Embedding analysis
│       └── generate_contrastive_embeddings.py  # Contrastive embeddings
├── data/                        # Data directory
│   ├── ensg_coding_genes.txt    # Essential gene list for preprocessing
│   ├── gene_names.txt           # Gene name mappings
│   ├── common_gene.txt          # Common gene identifiers
│   └── ...                      # Other data files (not tracked in git)
├── models/                      # Saved model checkpoints
├── checkpoints/                 # Training checkpoints
├── logs/                        # Training and execution logs  
├── training_plots/              # Training visualization outputs
├── notebooks/                   # Jupyter notebooks for exploration
├── docs/                        # Documentation
├── requirements/                # Requirements files
├── run.py                       # Main entry point script
├── setup.py                     # Package setup configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd VAE_embeddings
```

2. **Create and activate virtual environment:**
```bash
python -m venv vae_env
source vae_env/bin/activate  # On Windows: vae_env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Setup project (optional):**
```bash
python setup.py develop
```

## 📋 File Descriptions

### Core Models
- **`src/models/vae_model.py`**: Main VAE architecture with ELU activations, batch normalization, and dropout

### Training Scripts
- **`src/training/train_vae.py`**: Comprehensive training script with KL annealing, early stopping, and dynamic masking
- **`src/training/contrastive_vae_training.py`**: Extended training with contrastive learning objectives

### Configuration
- **`config/train_config.py`**: Centralized configuration management for all training parameters and hyperparameters

### Data Processing Utilities
- **`src/utils/prepossing.py`**: Data preprocessing pipeline for GTEx gene expression data
- **`src/utils/transpose_data.py`**: Matrix transposition utilities for data format conversion
- **`src/utils/encode.py`**: Encoding utilities for generating embeddings from trained models
- **`src/utils/new_pseudobulk.py`**: Pseudobulk data generation utilities

### Evaluation & Analysis
- **`src/evaluation/evaluate_vae.py`**: Comprehensive model evaluation with reconstruction metrics
- **`scripts/analysis/analyze_embeddings.py`**: Embedding quality analysis with PCA, UMAP, t-SNE
- **`scripts/analysis/generate_contrastive_embeddings.py`**: Contrastive embedding generation

### Job Scripts (SLURM)
All SLURM job scripts are organized in `scripts/slurm/`:
- **`run_training.sbatch`**: Main VAE training job
- **`run_contrastive.sbatch`**: Contrastive learning training
- **`run_embeddings.sbatch`**: Embedding generation
- **`run_analyze.sbatch`**: Analysis and visualization
- **`run_preprocessing.sbatch`**: Data preprocessing
- **`run_encode.sbatch`**: Model encoding
- **`run_transpose.sbatch`**: Data transposition

### Essential Data Files
- **`data/ensg_coding_genes.txt`**: Essential gene identifiers list (required for preprocessing)
- **`data/gene_names.txt`**: Gene name mapping file
- **`data/common_gene.txt`**: Common gene identifiers across datasets

## 🚀 Usage

### 1. Using the Main Entry Point
```bash
python run.py
```

### 2. Standard VAE Training
```bash
python src/training/train_vae.py
```

### 3. Contrastive VAE Training
```bash
python src/training/contrastive_vae_training.py
```

### 4. Generate Embeddings
```bash
python src/utils/encode.py
```

### 5. Analyze Results
```bash
python scripts/analysis/analyze_embeddings.py
```

### 6. Using SLURM (on HPC clusters)
```bash
# Submit training job
sbatch scripts/slurm/run_training.sbatch

# Submit analysis job
sbatch scripts/slurm/run_analyze.sbatch
```

## ⚙️ Key Features

### Advanced VAE Architecture
- **Deep encoder/decoder**: 5-layer architecture with gradual dimensionality reduction
- **ELU activations**: Optimized for standardized gene expression data
- **Batch normalization**: Improved training stability
- **Dropout regularization**: Prevents overfitting

### Sophisticated Training
- **Beta-VAE KL annealing**: Cyclic annealing with decay for better disentanglement
- **Dynamic masking**: Trains decoder with varying sparsity levels (5-25%)
- **Early stopping**: Prevents overfitting with patience-based stopping
- **Learning rate scheduling**: Adaptive learning rate reduction

### Comprehensive Evaluation
- **Reconstruction quality**: MSE-based reconstruction error analysis
- **Latent space analysis**: PCA, UMAP, t-SNE visualizations
- **Clustering quality**: Silhouette score and calinski-harabasz index
- **Gene similarity**: Correlation analysis in embedding space

## 📊 Model Configuration

The model configuration is centralized in `config/train_config.py`:

### Architecture Parameters
```python
hidden_dims = [4096, 2048, 1024, 512, 256]  # Encoder/decoder layers
latent_dim = 19797                           # Latent space dimension (matches gene count)
dropout_rate = 0.3                          # Dropout probability
```

### Training Parameters
```python
num_epochs = 300                            # Maximum training epochs
learning_rate = 1e-4                        # Initial learning rate
patience = 30                               # Early stopping patience
target_loss = 0.01                          # Target loss for training completion
```

### KL Annealing Configuration
```python
target_weight = 0.1                         # Final KL weight
annealing_type = 'cosine'                   # Annealing schedule type
beta_vae_style = True                       # Use beta-VAE style annealing
cycle_length = 40                           # Annealing cycle length
```

## 📈 Results

The trained VAE produces:
- **Gene embeddings**: 19,797-dimensional latent representations
- **Reconstruction quality**: Sub-1.0 MSE on standardized data
- **Biological relevance**: Embeddings capture gene functional relationships
- **Visualization**: 2D/3D projections via PCA, UMAP, t-SNE

## 🔬 Data Requirements

- **Input**: Log-transformed gene expression matrix (samples × genes)
- **Format**: CSV with numeric values only
- **Preprocessing**: StandardScaler normalization applied automatically
- **Size**: Handles high-dimensional data (19K+ genes, 17K+ samples)
- **Essential files**: Gene identifier files must be present in `data/` directory

## 🐛 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch_size in `config/train_config.py`
2. **Convergence issues**: Adjust KL annealing parameters in configuration
3. **Poor reconstruction**: Check data preprocessing and scaling
4. **Missing gene files**: Ensure `ensg_coding_genes.txt` and related files are in `data/`

### Performance Tips
- Use GPU when available (automatically detected)
- Monitor training with logging and checkpoints
- Adjust KL weight for better reconstruction/regularization balance
- Use the centralized configuration in `config/train_config.py` for consistent experiments

## 📚 References

- [Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
- [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599)
- [GTEx Consortium Dataset](https://gtexportal.org/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers. 