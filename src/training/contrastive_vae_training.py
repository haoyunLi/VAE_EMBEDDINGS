import torch
import torch.nn as nn
import numpy as np
import logging
from vae_model import VAE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_model_and_data(model_path, data_path):
    """Load the trained model and original data."""
    # Load model with weights_only=False since this is our own checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    model = VAE(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        latent_dim=checkpoint['latent_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data with first column as index
    data = pd.read_csv(data_path, index_col=0)
    
    # Convert all columns to float type
    data = data.astype(float)
    
    # Verify the number of features matches the scaler's expectation
    if data.shape[1] != checkpoint['input_dim']:
        raise ValueError(f"Number of features in data ({data.shape[1]}) does not match model's input dimension ({checkpoint['input_dim']})")
    
    scaler = checkpoint['scaler']
    
    return model, data, scaler

class ContrastiveVAETrainer:
    def __init__(self, vae_model, embedding_dim, temperature=0.07, learning_rate=1e-4):
        """
        Initialize the contrastive VAE trainer.
        
        Args:
            vae_model: pre-trained VAE model
            embedding_dim: Dimension of the latent space
            temperature: Temperature parameter for contrastive loss
            learning_rate: Learning rate for optimization
        """
        self.vae_model = vae_model
        self.temperature = temperature
        self.embedding_dim = embedding_dim
        
        # Initialize projection heads for both modalities
        self.pseudobulk_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.celltype_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.pseudobulk_projection.parameters()) + 
            list(self.celltype_projection.parameters()),
            lr=learning_rate
        )
        
        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_model = self.vae_model.to(self.device)
        self.pseudobulk_projection = self.pseudobulk_projection.to(self.device)
        self.celltype_projection = self.celltype_projection.to(self.device)
        
        # Freeze VAE model parameters
        for param in self.vae_model.parameters():
            param.requires_grad = False

    def compute_contrastive_loss(self, pseudobulk_embeddings, celltype_embeddings, pseudobulk_donors):
        """
        Compute contrastive loss between pseudobulk and grouped celltype embeddings.
        """
        # Project embeddings
        pseudobulk_proj = self.pseudobulk_projection(pseudobulk_embeddings)
        
        # Project and average celltype embeddings for each donor
        celltype_proj = self.celltype_projection(celltype_embeddings)
        celltype_proj = celltype_proj.mean(dim=1)  # Average celltype projections for each donor
        
        # Normalize projections
        pseudobulk_proj = nn.functional.normalize(pseudobulk_proj, dim=1)
        celltype_proj = nn.functional.normalize(celltype_proj, dim=1)
        
        # Compute similarity matrix with numerical stability
        similarity_matrix = torch.matmul(pseudobulk_proj, celltype_proj.T) / self.temperature
        
        # Create positive mask (same donor)
        positive_mask = torch.zeros_like(similarity_matrix)
        for i, p_donor in enumerate(pseudobulk_donors):
            for j, c_donor in enumerate(pseudobulk_donors):
                if p_donor == c_donor:
                    positive_mask[i, j] = 1
        
        # Add numerical stability to exponential calculations
        similarity_matrix = similarity_matrix - similarity_matrix.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(similarity_matrix)
        
        # Compute loss for pseudobulk direction with numerical stability
        positive_sim = torch.sum(exp_sim * positive_mask, dim=1)
        negative_sim = torch.sum(exp_sim * (1 - positive_mask), dim=1)
        
        # Add small epsilon to prevent log(0)
        pseudobulk_loss = -torch.mean(torch.log(positive_sim / (positive_sim + negative_sim + 1e-8) + 1e-8))
        
        # Compute loss for celltype direction with numerical stability
        positive_sim = torch.sum(exp_sim * positive_mask, dim=0)
        negative_sim = torch.sum(exp_sim * (1 - positive_mask), dim=0)
        
        # Add small epsilon to prevent log(0)
        celltype_loss = -torch.mean(torch.log(positive_sim / (positive_sim + negative_sim + 1e-8) + 1e-8))
        
        # Check for NaN values and handle them
        if torch.isnan(pseudobulk_loss) or torch.isnan(celltype_loss):
            logging.warning("NaN detected in loss computation. Using fallback values.")
            return torch.tensor(0.0, device=pseudobulk_loss.device, requires_grad=True)
        
        return pseudobulk_loss + celltype_loss

    def train_step(self, pseudobulk_batch, celltype_batch, pseudobulk_donors):
        """
        Perform a single training step with grouped celltype data.
        """
        self.optimizer.zero_grad()
        
        # Get embeddings from VAE
        with torch.no_grad():
            pseudobulk_embeddings = self.vae_model.encode(pseudobulk_batch)[0]
            
            # Reshape celltype batch for VAE processing
            batch_size, num_celltypes, feature_dim = celltype_batch.shape
            celltype_reshaped = celltype_batch.view(-1, feature_dim)  # Flatten to (batch_size * num_celltypes, feature_dim)
            celltype_embeddings = self.vae_model.encode(celltype_reshaped)[0]
            celltype_embeddings = celltype_embeddings.view(batch_size, num_celltypes, -1)  # Reshape back to (batch_size, num_celltypes, embedding_dim)
        
        # Compute contrastive loss
        loss = self.compute_contrastive_loss(
            pseudobulk_embeddings,
            celltype_embeddings,
            pseudobulk_donors
        )
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, train_loader, num_epochs, save_dir='checkpoints', patience=10, min_delta=1e-4):
        """
        Train the model for specified number of epochs with early stopping.
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Maximum number of epochs to train
            save_dir: Directory to save checkpoints
            patience: Number of epochs to wait for improvement before early stopping
            min_delta: Minimum change in loss to be considered as improvement
        """
        os.makedirs(save_dir, exist_ok=True)
        best_loss = float('inf')
        history = {
            'train_loss': [],
            'best_loss': []
        }
        
        # Early stopping variables
        no_improvement_count = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                pseudobulk_data, celltype_data, pseudobulk_donors = batch
                
                # Move data to device
                pseudobulk_data = pseudobulk_data.to(self.device)
                celltype_data = celltype_data.to(self.device)
                
                # Training step
                loss = self.train_step(
                    pseudobulk_data,
                    celltype_data,
                    pseudobulk_donors
                )
                
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            history['train_loss'].append(avg_loss)
            history['best_loss'].append(min(avg_loss, best_loss))
            
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            # Early stopping check
            if avg_loss < best_loss - min_delta:
                best_loss = avg_loss
                no_improvement_count = 0
                best_model_state = {
                    'pseudobulk_projection_state_dict': self.pseudobulk_projection.state_dict(),
                    'celltype_projection_state_dict': self.celltype_projection.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                self.save_checkpoint(os.path.join(save_dir, 'best_model.pth'))
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    # Restore best model
                    if best_model_state is not None:
                        self.pseudobulk_projection.load_state_dict(best_model_state['pseudobulk_projection_state_dict'])
                        self.celltype_projection.load_state_dict(best_model_state['celltype_projection_state_dict'])
                        self.optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
                    break
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Plot training history
        self.plot_training_history(history, save_dir)
        
        return history

    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        """
        torch.save({
            'pseudobulk_projection_state_dict': self.pseudobulk_projection.state_dict(),
            'celltype_projection_state_dict': self.celltype_projection.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(path)
        self.pseudobulk_projection.load_state_dict(checkpoint['pseudobulk_projection_state_dict'])
        self.celltype_projection.load_state_dict(checkpoint['celltype_projection_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def plot_training_history(self, history, save_dir):
        """
        Plot training history.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['best_loss'], label='Best Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        plt.close()

def load_and_preprocess_data(pseudobulk_path, celltype_path, scaler, batch_size=32):
    """
    Load and preprocess both pseudobulk and celltype-specific data.
    Groups celltype samples by donor and creates aligned datasets.
    """
    # Load data
    pseudobulk_df = pd.read_csv(pseudobulk_path, index_col=0)
    celltype_df = pd.read_csv(celltype_path, index_col=0)
    
    # Convert all columns to float type
    pseudobulk_df = pseudobulk_df.astype(float)
    celltype_df = celltype_df.astype(float)
    
    # Get donor IDs
    pseudobulk_donors = pseudobulk_df.index.tolist()
    celltype_donors = [idx.split('|')[1] for idx in celltype_df.index]
    
    # Create a mapping of donor to celltype indices
    donor_to_celltype_indices = {}
    for i, donor in enumerate(celltype_donors):
        if donor not in donor_to_celltype_indices:
            donor_to_celltype_indices[donor] = []
        donor_to_celltype_indices[donor].append(i)
    
    # Filter celltype data to only include donors that exist in pseudobulk data
    valid_donors = set(pseudobulk_donors) & set(donor_to_celltype_indices.keys())
    logging.info(f"Found {len(valid_donors)} common donors between pseudobulk and celltype data")
    
    # Create aligned datasets
    aligned_pseudobulk_data = []
    aligned_celltype_groups = []  # List of lists, each inner list contains celltype data for one donor
    aligned_donor_indices = []
    
    for donor in valid_donors:
        # Get pseudobulk data for this donor
        pseudobulk_idx = pseudobulk_donors.index(donor)
        pseudobulk_data = pseudobulk_df.iloc[pseudobulk_idx].values
        
        # Get all celltype data for this donor
        celltype_indices = donor_to_celltype_indices[donor]
        celltype_group = [celltype_df.iloc[idx].values for idx in celltype_indices]
        
        aligned_pseudobulk_data.append(pseudobulk_data)
        aligned_celltype_groups.append(celltype_group)
        aligned_donor_indices.append(pseudobulk_idx)
    
    # Convert to numpy arrays
    aligned_pseudobulk_data = np.array(aligned_pseudobulk_data)
    
    # Scale data
    pseudobulk_scaled = scaler.transform(aligned_pseudobulk_data)
    
    # Convert to tensors
    pseudobulk_tensor = torch.FloatTensor(pseudobulk_scaled)
    donor_indices_tensor = torch.LongTensor(aligned_donor_indices)
    
    # Create custom dataset
    class GroupedDataset(torch.utils.data.Dataset):
        def __init__(self, pseudobulk_data, celltype_groups, donor_indices, scaler):
            self.pseudobulk_data = pseudobulk_data
            self.celltype_groups = celltype_groups
            self.donor_indices = donor_indices
            self.scaler = scaler
        
        def __len__(self):
            return len(self.pseudobulk_data)
        
        def __getitem__(self, idx):
            # Get pseudobulk data
            pseudobulk = self.pseudobulk_data[idx]
            
            # Get and scale celltype group data
            celltype_group = self.celltype_groups[idx]
            celltype_scaled = self.scaler.transform(np.array(celltype_group))
            celltype_tensor = torch.FloatTensor(celltype_scaled)
            
            return pseudobulk, celltype_tensor, self.donor_indices[idx]
    
    dataset = GroupedDataset(pseudobulk_tensor, aligned_celltype_groups, donor_indices_tensor, scaler)
    logging.info(f"Created dataset with {len(dataset)} donor groups")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('contrastive_training.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Load VAE model and data
        model_path = 'checkpoints/vae_best.pth'
        pseudobulk_path = 'data/processed_weighted_pseudobulk_expression.csv'
        celltype_path = 'data/processed_celltype_specific_2d_matrix.csv'
        
        logging.info("Loading model and data...")
        vae_model, _, scaler = load_model_and_data(model_path, celltype_path)
        
        # Get latent dimension from checkpoint
        checkpoint = torch.load(model_path, weights_only=False)
        latent_dim = checkpoint['latent_dim']
        logging.info(f"Model loaded successfully with latent dimension: {latent_dim}")
        
        # Initialize trainer with more stable hyperparameters
        trainer = ContrastiveVAETrainer(
            vae_model=vae_model,
            embedding_dim=latent_dim,
            temperature=0.2,  
            learning_rate=1e-4  
        )
        
        # Load and preprocess data
        logging.info("Loading and preprocessing data...")
        train_loader = load_and_preprocess_data(
            pseudobulk_path,
            celltype_path,
            scaler,
            batch_size=32  
        )
        logging.info("Data loaded and preprocessed successfully")
        
        # Create checkpoints directory
        os.makedirs('checkpoints', exist_ok=True)
        
        # Train the model with early stopping
        logging.info("Starting training...")
        history = trainer.train(
            train_loader, 
            num_epochs=200,
            patience=20,
            min_delta=1e-4
        )
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 