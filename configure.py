import scanpy as sc
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Process neural network parameters.')
    
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='Learning rate for the optimizer')
    parser.add_argument('--hidden_dim', type=int, default=700,
                        help='Hidden dimension for the neural network')
    parser.add_argument('--train_decoder_epochs', type=int, default=5000,
                        help='Number of epochs for training scDecoder')
    parser.add_argument('--bulk_epochs', type=int, default=3000,
                        help='Number of epochs for bulk Encoder')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate. Default is 0.1')
    parser.add_argument('--cell_proportions', nargs='+', type=float,
                        default=[0.108, 0.232, 0.170  , 0.060, 0.162, 0.046, 0.020, 0.073, 0.016, 0.112],
                        help='Cell type proportions. Order must match the order in the mapping_dict')

    return parser


def load_data(data_dir, barcode_path):
    adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)

    barcodes_with_labels = pd.read_csv(barcode_path, sep=',', header=None)
    barcodes_with_labels.columns = ['barcodes', 'labels']

    # Step 2: Clean the labels data
    # Remove rows with 'unknown' or NaN labels
    barcodes_with_labels = barcodes_with_labels[
        # (barcodes_with_labels['labels'].notna()) &
        (barcodes_with_labels['labels'] != 'Unknown')
    ]

    # Here's the cleaned labels after filtering
    labels = barcodes_with_labels['labels'].values

    # Step 3: Filter the adata object
    # Retain only the observations that match the filtered barcodes
    filtered_barcodes = barcodes_with_labels['barcodes'].values
    adata = adata[adata.obs.index.isin(filtered_barcodes)]

    adata.obs['barcodes'] = adata.obs.index

    adata.obs = adata.obs.reset_index(drop=True)
    adata.obs = adata.obs.merge(barcodes_with_labels, on='barcodes', how='left')
    adata.X = adata.X.toarray()
    adata.obs.index = adata.obs.index.astype(str)

    mapping_dict = {
        'CD8+ NKT-like cells': '0',
        'Classical Monocytes': '1',
        'Effector CD4+ T cells': '2',
        'Macrophages': '3',
        'Myeloid Dendritic cells': '4',
        'Naive B cells': '5',
        'Naive CD4+ T cells': '6',
        'Naive CD8+ T cells': '7',
        'Natural killer  cells': '8',
        'Non-classical monocytes': '9',
        'Plasma B cells': '10',
        'Plasmacytoid Dendritic cells': '11',
        'Pre-B cells': '12',
        'Memory CD4+ T cells': '13',
        'Platelets': '14',
        'Unknown': '15'
    }

    color_map = {
        'CD8+ NKT-like cells': 'pink',
        'Classical Monocytes': 'orange',
        'Effector CD4+ T cells': 'grey',
        'Macrophages': 'tan',
        'Myeloid Dendritic cells': 'green',
        'Naive B cells': 'red',
        'Naive CD4+ T cells': 'slateblue',
        'Naive CD8+ T cells': 'blue',
        'Natural killer  cells': 'cyan',
        'Non-classical monocytes': 'black',
        'Plasma B cells': 'purple',
        'Plasmacytoid Dendritic cells': 'lime',
        'Pre-B cells': 'cornflowerblue',
        'Memory CD4+ T cells': 'magenta',
        'Platelets': 'yellow',
        'Unknown': 'teal'
    }

    adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict)
    adata.obs['labels'] = adata.obs['labels'].astype('category')

    labels = adata.obs['labels'].cat.codes.values
    labels = torch.LongTensor(labels)
    cell_types_tensor = labels

    X_tensor = torch.Tensor(adata.X)
    dataset = TensorDataset(X_tensor, labels)

    # Cell type fractions
    cell_type_fractions = np.unique(adata.obs['labels'].values, return_counts=True)[1]/len(adata.obs['labels'].values)
    
    return dataset, X_tensor, cell_types_tensor, cell_type_fractions, mapping_dict, color_map


def get_saved_GMM_params(mus_path, vars_path):
    gmm_mus_celltypes = torch.load(mus_path).squeeze().T # num_celltypes x latent_dim
    gmm_vars_celltypes = torch.load(vars_path).squeeze().T # num_celltypes x latent_dim
    return gmm_mus_celltypes, gmm_vars_celltypes


def configure(data_dir, barcode_path, mus_path=None, vars_path=None):
    dataset, X_tensor, cell_types_tensor, cell_type_fractions, mapping_dict, color_map = load_data(
                                                                                            data_dir=data_dir,
                                                                                            barcode_path=barcode_path,
                                                                                            )
    num_cells = X_tensor.shape[0]
    num_genes = X_tensor.shape[1]
    parser = get_arguments()
    
    parser.add_argument('--batch_size', type=int, default=num_cells,
                        help='Batch size for the DataLoader')
    
    parser.add_argument('--input_dim', type=int, default=num_genes,
                        help='Input dimension for the neural network')
    
    args = parser.parse_args()
    
    dataloader = DataLoader(dataset, batch_size=num_cells, shuffle=True)

    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"

    args.dataloader = dataloader
    args.cell_types_tensor = cell_types_tensor
    args.cell_type_fractions = cell_type_fractions
    args.mapping_dict = mapping_dict
    args.color_map = color_map
    args.z_dim = len(np.unique(cell_types_tensor))
    args.X_tensor = X_tensor
    label_map = {str(v): k for k, v in mapping_dict.items()}
    args.label_map = label_map

    if mus_path and vars_path:
        gmm_mus_celltypes, gmm_vars_celltypes = get_saved_GMM_params(mus_path, vars_path)
        args.gmm_mus_celltypes = gmm_mus_celltypes
        args.gmm_vars_celltypes = gmm_vars_celltypes
            
    print('Configuration is complete.')
    return args
