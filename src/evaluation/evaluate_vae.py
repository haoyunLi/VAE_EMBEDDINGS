import torch
import numpy as np
import logging
from vae_model import VAE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_model(model_path, device):
    """Load a trained VAE model."""
    checkpoint = torch.load(model_path, map_location=device)
    model = VAE(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        latent_dim=checkpoint['latent_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint['scaler']

def load_and_preprocess_data(file_path, scaler, batch_size=128):
    """Load and preprocess data for evaluation."""
    data = pd.read_csv(file_path)
    scaled_data = scaler.transform(data.values)
    tensor_data = torch.FloatTensor(scaled_data)
    dataset = TensorDataset(tensor_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def evaluate_model(model, test_loader, device):
    """Evaluate model performance on test set."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    all_latents = []
    all_reconstructions = []
    all_originals = []
    
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0].to(device)
            recon_batch, mu, log_var = model(data)
            loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, log_var)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            all_latents.append(mu.cpu().numpy())
            all_reconstructions.append(recon_batch.cpu().numpy())
            all_originals.append(data.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    avg_recon_loss = total_recon_loss / len(test_loader)
    avg_kl_loss = total_kl_loss / len(test_loader)
    
    all_latents = np.concatenate(all_latents, axis=0)
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    all_originals = np.concatenate(all_originals, axis=0)
    
    return {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'kl_loss': avg_kl_loss,
        'latent_representations': all_latents,
        'reconstructions': all_reconstructions,
        'originals': all_originals
    }

def plot_loss_curves(history, save_path='loss_curves.png'):
    """Plot training and validation loss curves."""
    # Determine number of subplots based on available data
    num_plots = 3
    if 'kl_weight' in history:
        num_plots = 4
    
    plt.figure(figsize=(12, 2 * num_plots))
    
    # Plot total loss
    plt.subplot(num_plots, 1, 1)
    plt.plot(history['train_total_loss'], label='Training')
    plt.plot(history['test_total_loss'], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot reconstruction loss
    plt.subplot(num_plots, 1, 2)
    plt.plot(history['train_recon_loss'], label='Training')
    plt.plot(history['test_recon_loss'], label='Validation')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot KL loss
    plt.subplot(num_plots, 1, 3)
    plt.plot(history['train_kl_loss'], label='Training')
    plt.plot(history['test_kl_loss'], label='Validation')
    plt.title('KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot KL weight if available
    if 'kl_weight' in history:
        plt.subplot(num_plots, 1, 4)
        plt.plot(history['kl_weight'], label='KL Weight', color='red')
        plt.title('KL Weight Annealing')
        plt.xlabel('Epoch')
        plt.ylabel('Weight')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_evaluation(model, test_loader, device, epoch=None, kl_weight=0.1):
    """Run evaluation on test set and return metrics."""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    all_originals = []
    all_reconstructions = []
    
    with torch.no_grad():
        for batch in test_loader:
            data = batch[0].to(device)
            recon_batch, mu, log_var = model(data)
            loss, recon_loss, kl_loss = model.loss_function(recon_batch, data, mu, log_var, kl_weight=kl_weight)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Store original and reconstructed data for accuracy metrics
            all_originals.append(data.cpu().numpy())
            all_reconstructions.append(recon_batch.cpu().numpy())
    
    # Calculate average losses
    avg_loss = total_loss / len(test_loader)
    avg_recon_loss = total_recon_loss / len(test_loader)
    avg_kl_loss = total_kl_loss / len(test_loader)
    
    # Calculate accuracy metrics
    all_originals = np.concatenate(all_originals, axis=0)
    all_reconstructions = np.concatenate(all_reconstructions, axis=0)
    
    # Mean Squared Error
    mse = np.mean((all_originals - all_reconstructions) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(all_originals - all_reconstructions))
    
    # R-squared score
    ss_res = np.sum((all_originals - all_reconstructions) ** 2)
    ss_tot = np.sum((all_originals - np.mean(all_originals)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    if epoch is not None:
        logging.info(f'Epoch {epoch} - Test Loss: {avg_loss:.2f}, Recon: {avg_recon_loss:.2f}, KL: {avg_kl_loss:.2f}')
        logging.info(f'Accuracy Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2_score:.4f}')
    
    return {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'kl_loss': avg_kl_loss,
        'mse': mse,
        'mae': mae,
        'r2_score': r2_score
    }

def plot_accuracy_curves(history, save_path='accuracy_curves.png'):
    """Plot accuracy metrics curves."""
    plt.figure(figsize=(12, 8))
    
    # Plot MSE
    plt.subplot(3, 1, 1)
    plt.plot(history['test_mse'], label='Test MSE')
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    # Plot MAE
    plt.subplot(3, 1, 2)
    plt.plot(history['test_mae'], label='Test MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # Plot R-squared
    plt.subplot(3, 1, 3)
    plt.plot(history['test_r2_score'], label='Test R²')
    plt.title('R-squared Score')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('vae_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    
    # Load model and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler = load_model('vae_model.pth', device)
    
    # Load test data
    test_loader = load_and_preprocess_data('data/gtex_RSEM_gene_tpm_transposed.csv', scaler)
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device)
    
    # Log results
    logging.info("Model Evaluation Results:")
    logging.info(f"Total Loss: {results['loss']:.2f}")
    logging.info(f"Reconstruction Loss: {results['recon_loss']:.2f}")
    logging.info(f"KL Loss: {results['kl_loss']:.2f}")
    
    # Calculate and log accuracy metrics
    mse = np.mean((results['originals'] - results['reconstructions']) ** 2)
    mae = np.mean(np.abs(results['originals'] - results['reconstructions']))
    ss_res = np.sum((results['originals'] - results['reconstructions']) ** 2)
    ss_tot = np.sum((results['originals'] - np.mean(results['originals'])) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    logging.info("Accuracy Metrics:")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"R² Score: {r2_score:.4f}")
    
    # Create accuracy history for plotting
    accuracy_history = {
        'test_mse': [mse],
        'test_mae': [mae],
        'test_r2_score': [r2_score]
    }
    
    # Plot accuracy curves
    plot_accuracy_curves(accuracy_history, save_path='evaluation_accuracy_curves.png')
    
    logging.info("Accuracy plots saved as 'evaluation_accuracy_curves.png'")

if __name__ == "__main__":
    main() 