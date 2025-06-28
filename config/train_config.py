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
    # Path to the input CSV file containing filtered GTEx coding genes data
    input_file: str = 'data/filtered_gtex_coding_genes.csv'
    # Number of samples processed together in each training batch
    batch_size: int = 128
    # Fraction of data reserved for testing (20% test, 80% train)
    test_size: float = 0.2
    # Number of worker processes for data loading (parallel processing)
    num_workers: int = 4
    # Whether to pin memory to GPU for faster data transfer
    pin_memory: bool = True


@dataclass  # Configuration class for model architecture parameters
class ModelConfig:
    """VAE model architecture configuration."""
    # Architecture parameters optimized for gene expression data
    hidden_dims: List[int] = None  # [4096, 2048, 1024, 512, 256]
    # Dimensionality of the latent space - matches number of genes
    latent_dim: int = 19797  # Match number of genes
    # Dropout probability for regularization during training
    dropout_rate: float = 0.3
    
    # Method called after dataclass initialization
    def __post_init__(self):
        # If hidden_dims not provided, set default architecture
        if self.hidden_dims is None:
            # More gradual reduction for large input dimension
            # Progressive dimensionality reduction: 19797 → 4096 → 2048 → 1024 → 512 → 256
            self.hidden_dims = [4096, 2048, 1024, 512, 256]


@dataclass  # Configuration for the training process
class TrainingConfig:
    """Training process configuration."""
    # Basic training parameters
    # Total number of training epochs (full passes through the dataset)
    num_epochs: int = 300
    # Learning rate for the optimizer (Adam)
    learning_rate: float = 1e-4
    # Number of epochs to wait without improvement before early stopping
    patience: int = 30  # Early stopping patience
    # Target loss value - training can stop early if reached
    target_loss: float = 0.01
    
    # Optimizer settings
    # L2 regularization strength to prevent overfitting
    weight_decay: float = 1e-5
    # Factor by which learning rate is reduced when plateau detected
    lr_scheduler_factor: float = 0.5
    # Number of epochs to wait before reducing learning rate
    lr_scheduler_patience: int = 5
    # Maximum norm for gradient clipping to prevent exploding gradients
    gradient_clip_norm: float = 1.0


@dataclass  # Configuration for KL divergence annealing strategy
class KLAnnealingConfig:
    """KL annealing configuration - optimized for gene expression data."""
    # KL annealing parameters
    # Final weight for KL divergence term in loss function
    target_weight: float = 0.1  # Final KL weight (for non-beta-VAE modes)
    # Epoch to start KL annealing from
    start_epoch: int = 0  # Start annealing from epoch 0
    # Epoch by which target weight should be reached
    end_epoch: int = 50  # Reach target weight by epoch 50
    # Type of annealing schedule: 'linear', 'cosine', or 'exponential'
    annealing_type: str = 'cosine'  # Best for gene expression: smooth transitions
    
    # Beta-VAE style configuration 
    # Whether to use cyclic annealing (periods of high/low KL weight)
    cyclic: bool = False  # Using beta-VAE style instead of cyclic
    # Length of each cycle in epochs (if cyclic is True)
    cycle_length: int = 40  # Cycle length: allows multiple exploration phases
    # Factor by which cycle amplitude decreases over time
    cycle_decay: float = 0.7  # Decay factor: gradually reduce cycle intensity
    # Whether to use beta-VAE style annealing (smooth, non-cyclic)
    beta_vae_style: bool = True  # Recommended: beta-VAE style annealing


@dataclass  # Configuration for dynamic input masking during training
class MaskingConfig:
    """Dynamic masking configuration for decoder robustness."""
    # Whether to enable random masking of input features during training
    dynamic_masking: bool = True  # Enable dynamic masking
    # Minimum fraction of features to mask in each batch
    mask_min: float = 0.05  # Minimum mask ratio (5%)
    # Maximum fraction of features to mask in each batch
    mask_max: float = 0.25  # Maximum mask ratio (25%)


@dataclass  # Configuration for logging, checkpointing, and output management
class LoggingConfig:
    """Logging and checkpointing configuration."""
    # Directory to save model checkpoints
    checkpoint_dir: str = 'checkpoints'
    # Directory to save training logs
    log_dir: str = 'logs'
    # Directory to save training plots and visualizations
    plot_dir: str = 'training_plots'

    # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_level: str = 'INFO'
    # Save training plots every N epochs
    save_every: int = 5  # Save plots every N epochs
    
    # Whether to only save the best model (based on validation loss)
    save_best_only: bool = False
    # Whether to save the model from the last epoch
    save_last: bool = True


@dataclass  # Main configuration class that combines all sub-configurations
class ExperimentConfig:
    """Complete experiment configuration combining all components."""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    kl_annealing: KLAnnealingConfig = None
    masking: MaskingConfig = None
    logging: LoggingConfig = None
    
    # Experiment metadata
    # Name identifier for this experiment
    experiment_name: str = "vae_gene_expression"
    # Readable description of the experiment
    description: str = "VAE training on GTEx gene expression data"
    
    # Method called after dataclass initialization
    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        # Create default DataConfig if not provided
        if self.data is None:
            self.data = DataConfig()
        # Create default ModelConfig if not provided
        if self.model is None:
            self.model = ModelConfig()
        # Create default TrainingConfig if not provided
        if self.training is None:
            self.training = TrainingConfig()
        # Create default KLAnnealingConfig if not provided
        if self.kl_annealing is None:
            self.kl_annealing = KLAnnealingConfig()
        # Create default MaskingConfig if not provided
        if self.masking is None:
            self.masking = MaskingConfig()
        # Create default LoggingConfig if not provided
        if self.logging is None:
            self.logging = LoggingConfig()
    
    # Method to create necessary directories for the experiment
    def create_directories(self):
        """Create necessary directories for training."""
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.logging.checkpoint_dir, exist_ok=True)
        # Create log directory if it doesn't exist
        os.makedirs(self.logging.log_dir, exist_ok=True)
        # Create plot directory if it doesn't exist
        os.makedirs(self.logging.plot_dir, exist_ok=True)
    
    # Method to generate a human-readable summary of all configuration settings
    def summary(self) -> str:
        """Return a formatted summary of the configuration."""
        # Multi-line formatted string with all key configuration parameters
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
        # Return the formatted summary string
        return summary


# Create a default configuration instance that can be imported and used directly
# This provides sensible defaults for all parameters
default_config = ExperimentConfig() 