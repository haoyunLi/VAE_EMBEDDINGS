"""
Main execution script for VAE Gene Expression Analysis
=====================================================

Provides a unified interface to run different components of the pipeline:
- Data preprocessing
- VAE training (standard and contrastive)
- Model evaluation
- Embedding analysis

Usage:
    python run.py --mode train                    # Train VAE (includes data loading)
    python run.py --mode train_contrastive        # Train contrastive VAE
    python run.py --mode evaluate                 # Evaluate trained model
    python run.py --mode analyze                  # Analyze embeddings
    python run.py --mode preprocess               # Preprocess data

"""

import argparse
import sys
import os
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(
        description="VAE Gene Expression Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'train_contrastive', 'evaluate', 'analyze', 'preprocess'],
        required=True,
        help='Mode of operation'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.py',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/filtered_gtex_coding_genes.csv',
        help='Path to input data file'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/vae_model.pth',
        help='Path to saved model file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f" VAE Gene Expression Analysis Pipeline")
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_path}")
    print("-" * 50)
    
    try:
        if args.mode == 'train':
            print(" Starting VAE training...")
            from src.training.train_vae import main as train_main
            train_main()
            
        elif args.mode == 'train_contrastive':
            print(" Starting contrastive VAE training...")
            from src.training.contrastive_vae_training import main as contrastive_main
            contrastive_main()
            
        elif args.mode == 'evaluate':
            print(" Evaluating trained model...")
            from src.evaluation.evaluate_vae import main as eval_main
            eval_main()
            
        elif args.mode == 'analyze':
            print(" Analyzing embeddings...")
            from scripts.analysis.analyze_embeddings import main as analyze_main
            analyze_main()
            
        elif args.mode == 'preprocess':
            print(" Preprocessing data...")
            from src.utils.prepossing import main as preprocess_main
            preprocess_main()
            
    except ImportError as e:
        print(f" Import Error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f" Error during execution: {e}")
        sys.exit(1)
    
    print(" Pipeline completed successfully!")

if __name__ == "__main__":
    main() 