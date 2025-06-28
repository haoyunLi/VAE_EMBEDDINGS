import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import umap
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from models.vae_model import VAE
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_model_and_data(model_path, data_path):
    """Load the trained model and original data."""
    # Load model with weights_only=False since this is own checkpoint
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

def generate_embeddings(model, data, scaler):
    """Generate embeddings for the data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Store original column names (gene names) and index (sample names)
    gene_names = data.columns.tolist() if isinstance(data, pd.DataFrame) else None
    sample_names = data.index.tolist() if isinstance(data, pd.DataFrame) else None
    
    # Preprocess data
    if isinstance(data, pd.DataFrame):
        non_numeric_cols = data.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            data = data.drop(columns=non_numeric_cols)
        data = data.values
    
    # Scale data
    scaled_data = scaler.transform(data)
    tensor_data = torch.FloatTensor(scaled_data)
    
    # Generate embeddings
    with torch.no_grad():
        mu, _ = model.encode(tensor_data.to(device))
    
    return mu.cpu().numpy(), gene_names, sample_names

def analyze_reconstruction_quality(model, data, scaler):
    """Analyze how well the model reconstructs the input data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Preprocess data
    if isinstance(data, pd.DataFrame):
        non_numeric_cols = data.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            data = data.drop(columns=non_numeric_cols)
        data = data.values
    
    # Scale data
    scaled_data = scaler.transform(data)
    tensor_data = torch.FloatTensor(scaled_data)
    
    # Reconstruct data
    with torch.no_grad():
        recon_data, _, _ = model(tensor_data.to(device))
    
    # Calculate reconstruction error for each sample
    recon_error = np.mean((recon_data.cpu().numpy() - scaled_data) ** 2, axis=1)
    
    # Plot reconstruction error distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(recon_error, bins=50)
    plt.title('Distribution of Reconstruction Error')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Count')
    plt.savefig('data/reconstruction_error_distribution.png')
    plt.close()    
    return recon_error

def analyze_latent_space(embeddings):
    """Analyze the structure of the latent space."""
    # Calculate pairwise distances
    distances = pdist(embeddings)
    
    # Plot distance distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(distances, bins=50)
    plt.title('Distribution of Pairwise Distances in Latent Space')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Count')
    plt.savefig('data/latent_space_distances.png')
    plt.close()
    
    # Calculate and plot correlation between latent dimensions
    corr_matrix = np.corrcoef(embeddings.T)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title('Correlation Between Latent Dimensions')
    plt.savefig('data/latent_dimensions_correlation.png')
    plt.close()
    
    return corr_matrix

def visualize_embeddings(embeddings, method='tsne', n_components=2):
    """Visualize embeddings using dimensionality reduction."""
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        title = 't-SNE visualization of embeddings'
    elif method == 'pca':
        reducer = PCA(n_components=n_components)
        title = 'PCA visualization of embeddings'
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        title = 'UMAP visualization of embeddings'
    
    # Reduce dimensionality
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Plot with smaller points and more transparency
    plt.figure(figsize=(12, 10))  # Increased figure size
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
               alpha=0.3,  # More transparent
               s=8,       # Smaller points
               c='lightblue')   # Single color for clarity
    plt.title(title, fontsize=14)
    plt.xlabel(f'{method.upper()} 1', fontsize=12)
    plt.ylabel(f'{method.upper()} 2', fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(f'data/embeddings_{method}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return reduced_embeddings

def analyze_clustering(embeddings, n_clusters_range=range(2, 11)): 
    """Analyze clustering quality for different numbers of clusters."""
    silhouette_scores = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(score)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(n_clusters_range), silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Clustering Quality Analysis')
    plt.savefig('data/clustering_quality.png')
    plt.close()
    
    return silhouette_scores

def main():
    # Load model and data
    model_path = 'checkpoints/vae_best.pth'  
    #data_path = 'data/processed_weighted_pseudobulk_expression.csv'
    data_path = 'data/processed_celltype_specific_2d_matrix.csv'
    
    logging.info("Loading model and data...")
    model, data, scaler = load_model_and_data(model_path, data_path)
    
    # Generate embeddings
    logging.info("Generating embeddings...")
    embeddings, gene_names, sample_names = generate_embeddings(model, data, scaler)
    
    # Create DataFrame with embeddings
    # Each row is a sample, each column is a gene
    embeddings_df = pd.DataFrame(embeddings, 
                               index=sample_names if sample_names is not None else None,
                               columns=gene_names if gene_names is not None else None)
    
    # Save embeddings
    #embeddings_df.to_csv('data/weighted_pseudobulk_gene_embeddings.csv')
    embeddings_df.to_csv('data/celltype_specific_gene_embeddings.csv')
    
    # Save embeddings with metadata
    embeddings_dict = {
        'data': {
            'embeddings': embeddings,
            'gene_names': gene_names,
            'sample_names': sample_names
        },
        'metadata': {
            'n_samples': len(embeddings),
            'n_genes': len(gene_names) if gene_names is not None else None,
            'embedding_dim': embeddings.shape[1]
        }
    }
    #np.save('data/weighted_pseudobulk_gene_embeddings.npy', embeddings_dict)
    np.save('data/celltype_specific_gene_embeddings.npy', embeddings_dict)
    
    # Generate summary report
    logging.info("\nAnalysis Summary:")
    logging.info(f"Number of samples: {len(embeddings)}")
    logging.info(f"Number of genes: {len(gene_names) if gene_names is not None else 'unknown'}")
    logging.info(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Visualize embeddings using t-SNE
    logging.info("Creating t-SNE visualization...")
    visualize_embeddings(embeddings, method='tsne')
    
    logging.info("Analysis completed. Check data/gene_embeddings.csv for the embeddings.")

if __name__ == "__main__":
    main() 