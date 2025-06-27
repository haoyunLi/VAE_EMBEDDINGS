import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def extract_and_save():
    # Source file paths
    gencode_file = 'data/gencode.v23.annotation.gtf'
    gtex_file = 'data/gtex_RSEM_gene_tpm_transposed.csv'
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    try:
        # Read gencode file and extract protein-coding gene IDs and names
        coding_gene_info = {}  # Dictionary to store gene_id -> gene_name mapping
        with open(gencode_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                if 'gene_id' in line and 'gene_type "protein_coding"' in line:
                    # Extract gene ID and gene name from the line
                    gene_id = line.split('gene_id "')[1].split('"')[0]
                    gene_name = line.split('gene_name "')[1].split('"')[0]
                    coding_gene_info[gene_id] = gene_name
        
        logging.info(f"Number of unique protein-coding genes in gencode: {len(coding_gene_info)}")
        
        # Read GTEx data
        gtex_data = pd.read_csv(gtex_file)
        logging.info(f"Original GTEx data dimensions: {gtex_data.shape}")
        
        # Get the first column name (sample names)
        sample_column = gtex_data.columns[0]
        
        # Filter GTEx data to keep only protein-coding genes (columns)
        # Keep the sample column and add the filtered gene columns
        filtered_columns = [sample_column] + [col for col in gtex_data.columns if col in coding_gene_info]
        filtered_data = gtex_data[filtered_columns]
        
        logging.info(f"Filtered GTEx data dimensions: {filtered_data.shape}")
        
        # Create a new dataframe with gene names instead of ENSG IDs
        gene_name_columns = [sample_column] + [coding_gene_info[col] for col in filtered_columns[1:]]
        filtered_data_with_names = filtered_data.copy()
        filtered_data_with_names.columns = gene_name_columns
        
        # Save filtered data with gene names
        filtered_data_with_names.to_csv('data/filtered_gtex_coding_genes.csv', index=False)
        logging.info("Successfully saved filtered protein-coding genes data with gene names")
        
        # Save ENSG ID list
        ensg_list = filtered_columns[1:]  # Exclude the sample column
        with open('data/ensg_coding_genes.txt', 'w') as f:
            f.write('\n'.join(ensg_list))
        logging.info("Successfully saved ENSG ID list")
        
        # Save gene name list
        gene_name_list = [coding_gene_info[ensg] for ensg in ensg_list]
        with open('data/gene_names.txt', 'w') as f:
            f.write('\n'.join(gene_name_list))
        logging.info("Successfully saved gene name list")
        
    except Exception as e:
        logging.error(f"Error occurred while processing files: {str(e)}")

if __name__ == "__main__":
    extract_and_save()
