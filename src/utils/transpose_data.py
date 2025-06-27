import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def convert_and_transpose(input_txt_file, csv_file=None, transposed_csv_file=None):
    """
    Reads a whitespace-delimited gene expression file, saves as CSV, then transposes and saves the transposed CSV.
    
    Parameters:
    -----------
    input_txt_file : str
        Path to the original whitespace-delimited file
    csv_file : str, optional
        Path to save the intermediate CSV file. If None, will use input_txt_file with '.csv' extension
    transposed_csv_file : str, optional
        Path to save the transposed CSV file. If None, will use input_txt_file with '_transposed.csv' extension
    """
    logging.info(f"Loading whitespace-delimited data from {input_txt_file}...")
    # Read the whitespace-delimited file
    data = pd.read_csv(input_txt_file, sep=None, engine='python', index_col=0)
    logging.info(f"Original data shape: {data.shape}")

    # Save as CSV
    if csv_file is None:
        base_name = os.path.basename(input_txt_file)
        csv_file = os.path.join('data', os.path.splitext(base_name)[0] + '.csv')
    logging.info(f"Saving as CSV to {csv_file}...")
    data.to_csv(csv_file)

    # Transpose
    logging.info("Transposing data...")
    transposed_data = data.T
    logging.info(f"Transposed data shape: {transposed_data.shape}")

    # Save transposed CSV
    if transposed_csv_file is None:
        base_name = os.path.basename(input_txt_file)
        transposed_csv_file = os.path.join('data', os.path.splitext(base_name)[0] + '_transposed.csv')
    logging.info(f"Saving transposed data to {transposed_csv_file}...")
    transposed_data.to_csv(transposed_csv_file)
    logging.info("Done!")

if __name__ == "__main__":
    input_txt_file = 'data/gtex_RSEM_gene_tpm'
    convert_and_transpose(input_txt_file) 