import torch
import pandas as pd
import numpy as np
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from models.vae_model import VAE
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

def generate_embeddings(model, data, scaler, device):
    """Generate embeddings for the data."""
    # Get feature names and values separately
    feature_names = data.columns
    data_values = data.values
    
    # Scale the data using values only
    scaled_data = scaler.transform(data_values)
    
    # Convert to PyTorch tensor
    tensor_data = torch.FloatTensor(scaled_data).to(device)
    
    # Generate embeddings
    with torch.no_grad():
        mu, _ = model.encode(tensor_data)
    
    return mu.cpu().numpy()

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('contrastive_embeddings.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        # Load VAE model and data
        model_path = 'checkpoints/vae_best.pth'
        pseudobulk_path = 'data/processed_weighted_pseudobulk_expression.csv'
        celltype_path = 'data/processed_celltype_specific_2d_matrix.csv'
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        # Load model and data
        logging.info("Loading model and data...")
        vae_model, _, scaler = load_model_and_data(model_path, celltype_path)
        vae_model = vae_model.to(device)
        
        # Load and process pseudobulk data
        logging.info("Processing pseudobulk data...")
        pseudobulk_data = pd.read_csv(pseudobulk_path, index_col=0)
        pseudobulk_data = pseudobulk_data.astype(float)
        
        # Generate embeddings for pseudobulk data
        pseudobulk_embeddings = generate_embeddings(vae_model, pseudobulk_data, scaler, device)
        
        # Load and process celltype data
        logging.info("Processing celltype data...")
        celltype_data = pd.read_csv(celltype_path, index_col=0)
        celltype_data = celltype_data.astype(float)
        
        # Generate embeddings for celltype data
        celltype_embeddings = generate_embeddings(vae_model, celltype_data, scaler, device)
        
        # Save embeddings
        np.save('data/pseudobulk_trained_embeddings.npy', pseudobulk_embeddings)
        np.save('data/celltype_trained_embeddings.npy', celltype_embeddings)
        
        # Save as CSV with sample names
        pd.DataFrame(pseudobulk_embeddings, index=pseudobulk_data.index).to_csv('data/pseudobulk_trained_embeddings.csv')
        pd.DataFrame(celltype_embeddings, index=celltype_data.index).to_csv('data/celltype_trained_embeddings.csv')
        
        logging.info("Embeddings generated and saved successfully!")
        logging.info(f"Pseudobulk embeddings shape: {pseudobulk_embeddings.shape}")
        logging.info(f"Celltype embeddings shape: {celltype_embeddings.shape}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 