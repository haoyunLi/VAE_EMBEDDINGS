"""
Variational Autoencoder (VAE) Model for Gene Expression Analysis
================================================================

This module implements a sophisticated VAE architecture specifically designed for 
high-dimensional gene expression data from the GTEx consortium. The model features
ELU activations optimized for standardized biological data, batch normalization
for training stability, and flexible loss computation with masking support.

Key Features:
- Deep encoder-decoder architecture with gradual dimensionality reduction
- ELU activation functions for smooth gradient flow with standardized data
- Batch normalization and dropout for robust training
- Support for masked reconstruction loss (for sparse data evaluation)
- Xavier weight initialization for stable training

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class VAE(nn.Module):
    """
    Variational Autoencoder for Gene Expression Data
    ================================================
    
    A deep VAE implementation optimized for high-dimensional gene expression analysis.
    Uses a symmetric encoder-decoder architecture with sophisticated regularization
    techniques and biologically-aware design choices.
    
    Architecture Details:
    - Encoder: Progressive dimensionality reduction through hidden layers
    - Latent Space: Gaussian latent variables with reparameterization trick
    - Decoder: Progressive dimensionality expansion mirroring encoder
    - Activations: ELU functions for smooth gradients with standardized data
    - Regularization: Batch normalization + dropout for stability
    
    Args:
        input_dim (int): Input feature dimension (number of genes)
        hidden_dims (List[int]): Hidden layer dimensions for encoder/decoder
        latent_dim (int): Latent space dimensionality
        dropout_rate (float): Dropout probability for regularization (default: 0.2)
    
    Example:
        >>> vae = VAE(input_dim=19797, hidden_dims=[4096, 2048, 1024, 512, 256], 
        ...           latent_dim=19797, dropout_rate=0.3)
        >>> reconstruction, mu, log_var = vae(gene_expression_data)
    """
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout_rate: float = 0.2):
        super(VAE, self).__init__()
        
        # Store architecture parameters for checkpointing
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        
        # ==========================================
        # ENCODER ARCHITECTURE
        # ==========================================
        # Progressive dimensionality reduction with regularization
        encoder_layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),           # Linear transformation
                nn.BatchNorm1d(hidden_dim),                # Normalize activations
                nn.ELU(),                                  # Smooth activation for standardized data
                nn.Dropout(dropout_rate)                   # Prevent overfitting
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # ==========================================
        # LATENT SPACE PARAMETERIZATION
        # ==========================================
        # Separate networks for mean and log-variance to ensure proper reparameterization
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)      # Mean of latent distribution
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)     # Log-variance of latent distribution
        
        # ==========================================
        # DECODER ARCHITECTURE  
        # ==========================================
        # Mirror encoder architecture for symmetric reconstruction
        decoder_layers = []
        prev_dim = latent_dim
        
        # Reverse hidden dimensions for symmetric architecture
        for i, hidden_dim in enumerate(reversed(hidden_dims)):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),           # Linear transformation
                nn.BatchNorm1d(hidden_dim),                # Normalize activations
                nn.ELU(),                                  # Consistent activation choice
                nn.Dropout(dropout_rate)                   # Regularization
            ])
            prev_dim = hidden_dim
            
        # Final reconstruction layer (no activation - continuous gene expression values)
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # ==========================================
        # WEIGHT INITIALIZATION
        # ==========================================
        # Xavier initialization for stable gradient flow
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize network weights using Xavier uniform initialization.
        
        This initialization scheme helps maintain gradient magnitudes across layers,
        particularly important for deep architectures like this VAE.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            # Xavier uniform for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                # Zero bias initialization
                nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input data to latent space parameters.
        
        Args:
            x (torch.Tensor): Input gene expression data [batch_size, input_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - mu: Mean of latent distribution [batch_size, latent_dim]
                - log_var: Log-variance of latent distribution [batch_size, latent_dim]
        """
        # Pass through encoder network
        h = self.encoder(x)
        
        # Compute latent distribution parameters
        mu = self.fc_mu(h)        # Mean parameter
        log_var = self.fc_var(h)  # Log-variance (ensures positive variance)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling.
        
        Implements z = μ + σ * ε where ε ~ N(0,1), enabling backpropagation
        through stochastic sampling by making randomness external to the computation graph.
        
        Args:
            mu (torch.Tensor): Mean of latent distribution [batch_size, latent_dim]
            log_var (torch.Tensor): Log-variance of latent distribution [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: Sampled latent variables [batch_size, latent_dim]
        """
        # Compute standard deviation from log-variance
        std = torch.exp(0.5 * log_var)
        
        # Sample noise from standard normal distribution
        eps = torch.randn_like(std)
        
        # Reparameterized sample: z = μ + σ * ε
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent variables to reconstruct input space.
        
        Args:
            z (torch.Tensor): Latent variables [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: Reconstructed data [batch_size, input_dim]
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Complete forward pass through the VAE.
        
        Args:
            x (torch.Tensor): Input gene expression data [batch_size, input_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - reconstruction: Reconstructed data [batch_size, input_dim]
                - mu: Mean of latent distribution [batch_size, latent_dim]  
                - log_var: Log-variance of latent distribution [batch_size, latent_dim]
        """
        # Encode to latent parameters
        mu, log_var = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode to reconstruction
        reconstruction = self.decode(z)
        
        return reconstruction, mu, log_var
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, log_var: torch.Tensor, 
                     kl_weight: float = 0.1, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss with optional masking for sparse evaluation.
        
        Implements the Evidence Lower BOund (ELBO) loss:
        L = E[log p(x|z)] - β * KL(q(z|x) || p(z))
        
        Where:
        - Reconstruction term encourages accurate data reconstruction
        - KL term regularizes latent space to be close to prior N(0,I)
        - β (kl_weight) balances reconstruction vs regularization
        - Optional masking allows evaluation on specific data subsets
        
        Args:
            recon_x (torch.Tensor): Reconstructed data [batch_size, input_dim]
            x (torch.Tensor): Original input data [batch_size, input_dim]
            mu (torch.Tensor): Latent distribution mean [batch_size, latent_dim]
            log_var (torch.Tensor): Latent distribution log-variance [batch_size, latent_dim]
            kl_weight (float): Weight for KL divergence term (β parameter)
            mask (Optional[torch.Tensor]): Binary mask for selective loss computation
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - total_loss: Combined ELBO loss
                - recon_loss: Reconstruction loss component
                - kl_loss: KL divergence loss component
        """
        # ==========================================
        # RECONSTRUCTION LOSS
        # ==========================================
        if mask is not None:
            # Masked reconstruction loss - focus on specific elements
            # Useful for evaluating reconstruction quality on sparse/missing data
            masked_recon = recon_x * mask
            masked_target = x * mask
            recon_loss = F.mse_loss(masked_recon, masked_target, reduction='sum') / (mask.sum() + 1e-8)
        else:
            # Standard reconstruction loss - all elements
            recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # ==========================================
        # KL DIVERGENCE LOSS
        # ==========================================
        # KL divergence between learned distribution q(z|x) and prior p(z) = N(0,I)
        # Analytical solution: KL = 0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # ==========================================
        # TOTAL LOSS (ELBO)
        # ==========================================
        # β-VAE formulation with adjustable KL weight
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def get_latent_representation(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        """
        Extract latent representations for downstream analysis.
        
        Args:
            x (torch.Tensor): Input data [batch_size, input_dim]
            use_mean (bool): If True, return mean of latent distribution,
                           else sample from distribution
                           
        Returns:
            torch.Tensor: Latent representations [batch_size, latent_dim]
        """
        mu, log_var = self.encode(x)
        
        if use_mean:
            # Deterministic encoding using distribution mean
            return mu
        else:
            # Stochastic encoding by sampling
            return self.reparameterize(mu, log_var) 