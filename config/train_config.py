"""
Training Configuration for VAE Gene Expression Analysis
======================================================

Centralized configuration management for all training parameters,
hyperparameters, and experimental settings.

This configuration is optimized for high-dimensional gene expression data
with sophisticated training techniques including beta-VAE style KL annealing
and dynamic masking for improved robustness.
"""

from dataclasses import dataclass
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Data processing configuration."""
    input_file: str = 'data/filtered_gtex_coding_genes.csv'
    batch_size: int = 128
    test_size: float = 0.2
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """VAE model architecture configuration."""
    # Architecture parameters optimized for gene expression data
    hidden_dims: List[int] = None  # [4096, 2048, 1024, 512, 256]
    latent_dim: int = 19797  # Match number of genes
    dropout_rate: float = 0.3
    
    def __post_init__(self):
        if self.hidden_dims is None:
            # More gradual reduction for large input dimension
            self.hidden_dims = [4096, 2048, 1024, 512, 256]


@dataclass
class TrainingConfig:
    """Training process configuration."""
    # Basic training parameters
    num_epochs: int = 300
    learning_rate: float = 1e-4
    patience: int = 30  # Early stopping patience
    target_loss: float = 0.01
    
    # Optimizer settings
    weight_decay: float = 1e-5
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    gradient_clip_norm: float = 1.0


@dataclass
class KLAnnealingConfig:
    """KL annealing configuration - optimized for gene expression data."""
    # KL annealing parameters
    target_weight: float = 0.1  # Final KL weight (for non-beta-VAE modes)
    start_epoch: int = 0  # Start annealing from epoch 0
    end_epoch: int = 50  # Reach target weight by epoch 50
    annealing_type: str = 'cosine'  # Best for gene expression: smooth transitions
    
    # Beta-VAE style configuration (recommended)
    cyclic: bool = False  # Using beta-VAE style instead of cyclic
    cycle_length: int = 40  # Cycle length: allows multiple exploration phases
    cycle_decay: float = 0.7  # Decay factor: gradually reduce cycle intensity
    beta_vae_style: bool = True  # Recommended: beta-VAE style annealing


@dataclass
class MaskingConfig:
    """Dynamic masking configuration for decoder robustness."""
    dynamic_masking: bool = True  # Enable dynamic masking
    mask_min: float = 0.05  # Minimum mask ratio (5%)
    mask_max: float = 0.25  # Maximum mask ratio (25%)


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    # Directories
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    plot_dir: str = 'training_plots'
    
    # Logging settings
    log_level: str = 'INFO'
    save_every: int = 5  # Save plots every N epochs
    
    # Model saving
    save_best_only: bool = False
    save_last: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all components."""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    kl_annealing: KLAnnealingConfig = None
    masking: MaskingConfig = None
    logging: LoggingConfig = None
    
    # Experiment metadata
    experiment_name: str = "vae_gene_expression"
    description: str = "VAE training on GTEx gene expression data"
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.kl_annealing is None:
            self.kl_annealing = KLAnnealingConfig()
        if self.masking is None:
            self.masking = MaskingConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
    
    def create_directories(self):
        """Create necessary directories for training."""
        os.makedirs(self.logging.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logging.log_dir, exist_ok=True)
        os.makedirs(self.logging.plot_dir, exist_ok=True)
    
    def summary(self) -> str:
        """Return a formatted summary of the configuration."""
        summary = f"""
        Experiment Configuration Summary
        ================================
        
        Experiment: {self.experiment_name}
        Description: {self.description}
        
        Data Configuration:
        - Input file: {self.data.input_file}
        - Batch size: {self.data.batch_size}
        - Test split: {self.data.test_size}
        
        Model Architecture:
        - Hidden dimensions: {self.model.hidden_dims}
        - Latent dimension: {self.model.latent_dim}
        - Dropout rate: {self.model.dropout_rate}
        
        Training Configuration:
        - Epochs: {self.training.num_epochs}
        - Learning rate: {self.training.learning_rate}
        - Patience: {self.training.patience}
        - Target loss: {self.training.target_loss}
        
        KL Annealing:
        - Beta-VAE style: {self.kl_annealing.beta_vae_style}
        - Target weight: {self.kl_annealing.target_weight}
        - Annealing type: {self.kl_annealing.annealing_type}
        - Cycle length: {self.kl_annealing.cycle_length}
        
        Dynamic Masking:
        - Enabled: {self.masking.dynamic_masking}
        - Mask range: [{self.masking.mask_min}, {self.masking.mask_max}]
        """
        return summary


# Default configuration instance
default_config = ExperimentConfig() 