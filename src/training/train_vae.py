import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..models.vae_model import VAE
from ..evaluation.evaluate_vae import run_evaluation, plot_loss_curves
import logging
import os
import sys

# Add config directory to path for importing configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.train_config import default_config

# Configure logging - will be updated with config in main()
def setup_logging(config):
    """Setup logging with configuration."""
    log_file = os.path.join(config.logging.log_dir, 'vae_training.log')
    logging.basicConfig(
        level=getattr(logging, config.logging.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0

class KLAnnealingScheduler:
    """KL annealing scheduler that gradually increases KL weight from 0 to target."""
    def __init__(self, target_weight=0.1, start_epoch=0, end_epoch=50, annealing_type='linear', 
                 cyclic=False, cycle_length=50, cycle_decay=0.5, beta_vae_style=False):
        self.target_weight = target_weight
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.annealing_type = annealing_type
        self.cyclic = cyclic
        self.cycle_length = cycle_length
        self.cycle_decay = cycle_decay
        self.beta_vae_style = beta_vae_style
        
    def get_weight(self, epoch):
        """Get current KL weight based on epoch."""
        if epoch < self.start_epoch:
            return 0.0
        
        if self.beta_vae_style:
            return self._get_beta_vae_weight(epoch)
        elif self.cyclic:
            return self._get_cyclic_weight(epoch)
        else:
            return self._get_standard_weight(epoch)
    
    def _get_beta_vae_weight(self, epoch):
        """Get weight for beta-VAE style scheduling (ramp to 1.0 then decay per cycle)."""
        cycle_epoch = epoch - self.start_epoch
        cycle_number = cycle_epoch // self.cycle_length
        cycle_progress = (cycle_epoch % self.cycle_length) / self.cycle_length
        
        # In beta-VAE style, we ramp up to 1.0 within each cycle
        if self.annealing_type == 'linear':
            base_weight = 1.0 * cycle_progress  # Ramp to 1.0
        elif self.annealing_type == 'cosine':
            base_weight = 1.0 * (1 - np.cos(cycle_progress * np.pi)) / 2  # Smooth ramp to 1.0
        elif self.annealing_type == 'exponential':
            base_weight = 1.0 * (np.exp(cycle_progress) - 1) / (np.e - 1)  # Exp ramp to 1.0
        else:
            base_weight = 1.0 * cycle_progress
        
        # Apply cycle decay (beta-VAE style: each cycle starts lower)
        decay_factor = self.cycle_decay ** cycle_number
        final_weight = base_weight * decay_factor
        
        # Ensure minimum weight for stability
        return max(final_weight, 0.01)
    
    def _get_standard_weight(self, epoch):
        """Get weight for standard (non-cyclic) annealing."""
        if epoch >= self.end_epoch:
            return self.target_weight
        else:
            # Calculate progress from 0 to 1
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            
            if self.annealing_type == 'linear':
                return self.target_weight * progress
            elif self.annealing_type == 'cosine':
                # Smooth cosine annealing
                return self.target_weight * (1 - np.cos(progress * np.pi)) / 2
            elif self.annealing_type == 'exponential':
                # Exponential annealing
                return self.target_weight * (np.exp(progress) - 1) / (np.e - 1)
            else:
                return self.target_weight * progress
    
    def _get_cyclic_weight(self, epoch):
        """Get weight for cyclic annealing."""
        # Calculate which cycle we're in
        cycle_epoch = epoch - self.start_epoch
        cycle_number = cycle_epoch // self.cycle_length
        cycle_progress = (cycle_epoch % self.cycle_length) / self.cycle_length
        
        # Calculate decay factor for this cycle
        decay_factor = self.cycle_decay ** cycle_number
        
        # Calculate base weight for this cycle
        if self.annealing_type == 'linear':
            base_weight = self.target_weight * cycle_progress
        elif self.annealing_type == 'cosine':
            base_weight = self.target_weight * (1 - np.cos(cycle_progress * np.pi)) / 2
        elif self.annealing_type == 'exponential':
            base_weight = self.target_weight * (np.exp(cycle_progress) - 1) / (np.e - 1)
        else:
            base_weight = self.target_weight * cycle_progress
        
        # Apply decay and ensure we don't go below a minimum
        final_weight = base_weight * decay_factor
        return max(final_weight, self.target_weight * 0.01)  # Minimum 1% of target weight

def save_checkpoint(model, scaler, input_dim, hidden_dims, latent_dim, epoch, loss, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'latent_dim': latent_dim,
        'loss': loss
    }
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Save regular checkpoint
    torch.save(checkpoint, f'checkpoints/vae_epoch_{epoch}.pth')
    
    # Save best model
    if is_best:
        torch.save(checkpoint, 'checkpoints/vae_best.pth')
        logging.info(f"New best model saved with loss: {loss:.2f}")

def load_and_preprocess_data(file_path, batch_size=128, test_size=0.2):
    # Load data
    logging.info("Loading data...")
    data = pd.read_csv(file_path)
    
    # Check for non-numeric columns and remove them
    non_numeric_cols = data.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        logging.info(f"Removing non-numeric columns: {non_numeric_cols}")
        data = data.drop(columns=non_numeric_cols)
    
    # Log data statistics before scaling
    logging.info(f"Data statistics before scaling:")
    logging.info(f"Mean: {data.values.mean():.2f}")
    logging.info(f"Std: {data.values.std():.2f}")
    logging.info(f"Min: {data.values.min():.2f}")
    logging.info(f"Max: {data.values.max():.2f}")
    
    # Convert to numpy array and scale
    logging.info("Preprocessing data...")
    # The data is already log-transformed, so we just need to scale it
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)
    
    # Log statistics after scaling
    logging.info(f"Data statistics after scaling:")
    logging.info(f"Mean: {scaled_data.mean():.2f}")
    logging.info(f"Std: {scaled_data.std():.2f}")
    logging.info(f"Min: {scaled_data.min():.2f}")
    logging.info(f"Max: {scaled_data.max():.2f}")
    
    # Convert to PyTorch tensors
    tensor_data = torch.FloatTensor(scaled_data)
    
    # Create train/test split
    logging.info("Creating train/test split...")
    n_samples = len(tensor_data)
    indices = torch.randperm(n_samples)
    test_size = int(n_samples * test_size)
    train_indices = indices[test_size:]
    test_indices = indices[:test_size]
    
    train_data = tensor_data[train_indices]
    test_data = tensor_data[test_indices]
    
    # Create DataLoaders
    train_dataset = TensorDataset(train_data)
    test_dataset = TensorDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logging.info(f"Data loaded and preprocessed. Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    logging.info(f"Input dimension: {data.shape[1]}")
    return train_loader, test_loader, scaler, data.shape[1]

def train_vae(model, train_loader, test_loader, num_epochs, learning_rate=1e-4, patience=7, scaler=None, input_dim=None, hidden_dims=None, latent_dim=None, target_loss=1000, kl_target_weight=0.1, kl_start_epoch=0, kl_end_epoch=50, kl_annealing_type='linear', kl_cyclic=False, kl_cycle_length=100, kl_cycle_decay=0.5, kl_beta_vae_style=False, dynamic_masking=False, mask_min=0.05, mask_max=0.25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=patience)
    
    # Initialize KL annealing scheduler
    kl_scheduler = KLAnnealingScheduler(
        target_weight=kl_target_weight,
        start_epoch=kl_start_epoch,
        end_epoch=kl_end_epoch,
        annealing_type=kl_annealing_type,
        cyclic=kl_cyclic,
        cycle_length=kl_cycle_length,
        cycle_decay=kl_cycle_decay,
        beta_vae_style=kl_beta_vae_style
    )
    
    # Training history
    history = {
        'train_total_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'test_total_loss': [],
        'test_recon_loss': [],
        'test_kl_loss': [],
        'mask_total_loss': [],
        'mask_recon_loss': [],
        'mask_kl_loss': [],
        'kl_weight': []  # Track KL weight over epochs
    }
    
    # Create directory for training visualizations
    os.makedirs('training_plots', exist_ok=True)
    
    logging.info(f"Training on {device}")
    if kl_beta_vae_style:
        logging.info(f"KL annealing: {kl_annealing_type} BETA-VAE STYLE (ramp to 1.0, decay per cycle) from epoch {kl_start_epoch}, cycle length: {kl_cycle_length}, decay: {kl_cycle_decay}")
    elif kl_cyclic:
        logging.info(f"KL annealing: {kl_annealing_type} CYCLIC from epoch {kl_start_epoch}, cycle length: {kl_cycle_length}, decay: {kl_cycle_decay}, target weight: {kl_target_weight}")
    else:
        logging.info(f"KL annealing: {kl_annealing_type} from epoch {kl_start_epoch} to {kl_end_epoch}, target weight: {kl_target_weight}")
    
    if dynamic_masking:
        logging.info(f"Dynamic masking: U({mask_min}, {mask_max}) - teaches decoder to handle multiple sparsity levels")
    else:
        logging.info(f"Fixed masking: 20% - standard evaluation")
    
    for epoch in range(num_epochs):
        # Get current KL weight
        current_kl_weight = kl_scheduler.get_weight(epoch)
        
        # Training phase
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            data = batch[0].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            
            # Calculate loss with current KL weight
            loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, log_var, kl_weight=current_kl_weight)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.2f}',
                'recon_loss': f'{recon_loss.item():.2f}',
                'kl_loss': f'{kl_loss.item():.2f}',
                'kl_weight': f'{current_kl_weight:.3f}'
            })
        
        # Calculate average training losses
        avg_train_loss = total_loss / len(train_loader)
        avg_train_recon_loss = total_recon_loss / len(train_loader)
        avg_train_kl_loss = total_kl_loss / len(train_loader)
        
        # Evaluation phase
        test_results = run_evaluation(model, test_loader, device, epoch=epoch, kl_weight=current_kl_weight)
        
        # Masking evaluation
        masking_results = evaluate_with_masking(
            model, test_loader, device, 
            kl_weight=current_kl_weight,
            dynamic_masking=dynamic_masking,
            mask_min=mask_min,
            mask_max=mask_max
        )
        
        # Update learning rate
        scheduler.step(test_results['loss'])
        
        # Store history
        history['train_total_loss'].append(avg_train_loss)
        history['train_recon_loss'].append(avg_train_recon_loss)
        history['train_kl_loss'].append(avg_train_kl_loss)
        history['test_total_loss'].append(test_results['loss'])
        history['test_recon_loss'].append(test_results['recon_loss'])
        history['test_kl_loss'].append(test_results['kl_loss'])
        history['mask_total_loss'].append(masking_results['loss'])
        history['mask_recon_loss'].append(masking_results['recon_loss'])
        history['mask_kl_loss'].append(masking_results['kl_loss'])
        history['kl_weight'].append(current_kl_weight)
        
        # Log results
        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'KL Weight: {current_kl_weight:.3f}')
        logging.info(f'Training - Loss: {avg_train_loss:.2f}, Recon: {avg_train_recon_loss:.2f}, KL: {avg_train_kl_loss:.2f}')
        logging.info(f'Testing  - Loss: {test_results["loss"]:.2f}, Recon: {test_results["recon_loss"]:.2f}, KL: {test_results["kl_loss"]:.2f}')
        logging.info(f'Masking  - Loss: {masking_results["loss"]:.2f}, Recon: {masking_results["recon_loss"]:.2f}, KL: {masking_results["kl_loss"]:.2f}')
        
        # Save checkpoint
        save_checkpoint(
            model, scaler, input_dim, hidden_dims, latent_dim,
            epoch, test_results['loss'],
            is_best=(test_results['loss'] == min(history['test_total_loss']))
        )
        
        # Plot and save loss curves every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_loss_curves(history, save_path=f'training_plots/loss_curves_epoch_{epoch+1}.png')
        
        # Check if we've reached target loss
        if test_results['loss'] <= target_loss:
            logging.info(f"Reached target loss of {target_loss} at epoch {epoch+1}")
            break
        
        # Early stopping check
        early_stopping(test_results['loss'], model)
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            # Load best model
            model.load_state_dict(early_stopping.best_model)
            break
    
    # Save final loss curves
    plot_loss_curves(history, save_path='training_plots/final_loss_curves.png')
    
    return history, model

def evaluate_with_masking(model, test_loader, device, mask_ratio=0.2, kl_weight=0.1, dynamic_masking=False, mask_min=0.05, mask_max=0.25):
    """Evaluate model performance with random masking."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0].to(device)
            
            # Create dynamic or fixed mask ratio
            if dynamic_masking:
                current_mask_ratio = np.random.uniform(mask_min, mask_max)
            else:
                current_mask_ratio = mask_ratio
            
            # Create random mask
            mask = torch.rand_like(data) > current_mask_ratio
            masked_data = data * mask
            
            # Forward pass with masked data
            recon_batch, mu, log_var = model(masked_data)
            
            # Calculate loss only on masked elements
            loss, recon_loss, kl_loss = model.loss_function(
                recon_batch, data, mu, log_var, 
                kl_weight=kl_weight,
                mask=~mask  # Invert mask to focus on masked elements
            )
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    return {
        'loss': total_loss / len(test_loader),
        'recon_loss': total_recon_loss / len(test_loader),
        'kl_loss': total_kl_loss / len(test_loader)
    }

def main():
    # Load configuration
    config = default_config
    
    # Create necessary directories
    config.create_directories()
    
    # Setup logging
    setup_logging(config)
    
    # Print configuration summary
    print(config.summary())
    logging.info("Starting VAE training with configuration:")
    logging.info(config.summary())
    
    # Extract configuration values
    input_file = config.data.input_file
    batch_size = config.data.batch_size
    num_epochs = config.training.num_epochs
    learning_rate = config.training.learning_rate
    hidden_dims = config.model.hidden_dims
    latent_dim = config.model.latent_dim
    test_size = config.data.test_size
    patience = config.training.patience
    target_loss = config.training.target_loss
    
    # KL annealing parameters from config
    kl_target_weight = config.kl_annealing.target_weight
    kl_start_epoch = config.kl_annealing.start_epoch
    kl_end_epoch = config.kl_annealing.end_epoch
    kl_annealing_type = config.kl_annealing.annealing_type
    kl_cyclic = config.kl_annealing.cyclic
    kl_cycle_length = config.kl_annealing.cycle_length
    kl_cycle_decay = config.kl_annealing.cycle_decay
    kl_beta_vae_style = config.kl_annealing.beta_vae_style
    
    # Dynamic masking parameters from config
    dynamic_masking = config.masking.dynamic_masking
    mask_min = config.masking.mask_min
    mask_max = config.masking.mask_max
    
    # Log GPU memory before training
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logging.info(f"Initial GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # Log KL annealing configuration
    logging.info("KL Annealing Configuration (BETA-VAE STYLE FOR HIGH-DIMENSIONAL DATA):")
    logging.info(f"  Target weight: {kl_target_weight}")
    logging.info(f"  Start epoch: {kl_start_epoch}")
    logging.info(f"  End epoch: {kl_end_epoch}")
    logging.info(f"  Annealing type: {kl_annealing_type} (smooth transitions)")
    logging.info(f"  Beta-VAE style: {kl_beta_vae_style} (ramp to 1.0, decay per cycle)")
    logging.info(f"  Cycle length: {kl_cycle_length} (exploration phases)")
    logging.info(f"  Cycle decay: {kl_cycle_decay} (gradual regularization)")
    logging.info(f"  Benefits: Better disentanglement, prevents posterior collapse")
    
    # Log dynamic masking configuration
    logging.info("Dynamic Masking Configuration (IMPROVES DECODER ROBUSTNESS):")
    logging.info(f"  Dynamic masking: {dynamic_masking}")
    logging.info(f"  Mask range: U({mask_min}, {mask_max})")
    logging.info(f"  Benefits: Teaches decoder to handle multiple sparsity levels")
    
    # Load and preprocess data
    train_loader, test_loader, scaler, input_dim = load_and_preprocess_data(
        input_file, 
        batch_size, 
        test_size=test_size
    )
    
    # Initialize model with config dropout rate
    model = VAE(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, dropout_rate=config.model.dropout_rate)
    
    # Train model
    history, best_model = train_vae(
        model, 
        train_loader, 
        test_loader, 
        num_epochs, 
        learning_rate,
        patience=patience,
        scaler=scaler,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        target_loss=target_loss,
        kl_target_weight=kl_target_weight,
        kl_start_epoch=kl_start_epoch,
        kl_end_epoch=kl_end_epoch,
        kl_annealing_type=kl_annealing_type,
        kl_cyclic=kl_cyclic,
        kl_cycle_length=kl_cycle_length,
        kl_cycle_decay=kl_cycle_decay,
        kl_beta_vae_style=kl_beta_vae_style,
        dynamic_masking=dynamic_masking,
        mask_min=mask_min,
        mask_max=mask_max
    )
    
    # Save final model
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'scaler': scaler,
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'latent_dim': latent_dim,
        'kl_annealing_config': {
            'target_weight': kl_target_weight,
            'start_epoch': kl_start_epoch,
            'end_epoch': kl_end_epoch,
            'annealing_type': kl_annealing_type,
            'cyclic': kl_cyclic,
            'cycle_length': kl_cycle_length,
            'cycle_decay': kl_cycle_decay,
            'beta_vae_style': kl_beta_vae_style
        },
        'dynamic_masking_config': {
            'enabled': dynamic_masking,
            'mask_min': mask_min,
            'mask_max': mask_max
        }
    }, 'vae_model.pth')
    
    # Run final evaluation
    final_results = run_evaluation(best_model, test_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), kl_weight=kl_target_weight)
    logging.info("Final Evaluation Results:")
    logging.info(f"Total Loss: {final_results['loss']:.2f}")
    logging.info(f"Reconstruction Loss: {final_results['recon_loss']:.2f}")
    logging.info(f"KL Loss: {final_results['kl_loss']:.2f}")
    
    # Run masking evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masking_results = evaluate_with_masking(
        best_model, test_loader, device, 
        kl_weight=kl_target_weight,
        dynamic_masking=dynamic_masking,
        mask_min=mask_min,
        mask_max=mask_max
    )
    logging.info("\nMasking Evaluation Results (Dynamic masking):")
    logging.info(f"Total Loss: {masking_results['loss']:.2f}")
    logging.info(f"Reconstruction Loss: {masking_results['recon_loss']:.2f}")
    logging.info(f"KL Loss: {masking_results['kl_loss']:.2f}")
    
    # Log final GPU memory usage
    if torch.cuda.is_available():
        logging.info(f"Final GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    logging.info("Training completed and model saved!")

if __name__ == "__main__":
    main() 