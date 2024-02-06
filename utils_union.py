import scanpy as sc
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import tqdm

# Set random seed.
np.random.seed(4)
# Set torch seed.
torch.manual_seed(4)

def get_celltype2int_dict():
    mapping_dict = {
        'Naive B cells': 0, 'Non-classical monocytes': 1, 'Classical Monocytes': 2, 'Natural killer  cells': 3,
        'CD8+ NKT-like cells': 4, 'Memory CD4+ T cells': 5, 'Naive CD8+ T cells': 6, 'Platelets': 7, 'Pre-B cells':8,
        'Plasmacytoid Dendritic cells':9, 'Effector CD4+ T cells':10, 'Macrophages':11, 'Myeloid Dendritic cells':12,
        'Effector CD8+ T cells':13, 'Plasma B cells': 14, 'Memory B cells': 15, "Naive CD4+ T cells": 16,
        'Progenitor cells':17, 'γδ-T cells':18, 'Eosinophils': 19, 'Neutrophils': 20, 'Basophils': 21, 'Mast cells': 22,
        'Intermediate monocytes': 23, 'Megakaryocyte': 24, 'Endothelial': 25, 'Erythroid-like and erythroid precursor cells': 26,
        'HSC/MPP cells': 27, 'Granulocytes': 28, 'ISG expressing immune cells': 29, 'Cancer cells': 30, "Memory CD8+ T cells": 31,
        "Pro-B cells": 32, "Immature B cells": 33
    }
    return mapping_dict


def get_celltype2strint_dict():
    mapping_dict = {
        'Naive B cells': '0', 'Non-classical monocytes': '1', 'Classical Monocytes': '2', 'Natural killer  cells': '3',
        'CD8+ NKT-like cells': '4', 'Memory CD4+ T cells': '5', 'Naive CD8+ T cells': '6', 'Platelets': '7', 'Pre-B cells': '8',
        'Plasmacytoid Dendritic cells': '9', 'Effector CD4+ T cells': '10', 'Macrophages': '11', 'Myeloid Dendritic cells': '12',
        'Effector CD8+ T cells': '13', 'Plasma B cells': '14', 'Memory B cells': '15', "Naive CD4+ T cells": "16",
        'Progenitor cells':'17', 'γδ-T cells':'18', 'Eosinophils': '19', 'Neutrophils': '20', 'Basophils': '21', 'Mast cells': '22',
        'Intermediate monocytes': '23', 'Megakaryocyte': '24', 'Endothelial': '25', 'Erythroid-like and erythroid precursor cells': '26',
        'HSC/MPP cells': '27', 'Granulocytes': '28', 'ISG expressing immune cells': '29', 'Cancer cells': '30', "Memory CD8+ T cells": "31",
        "Pro-B cells": "32", "Immature B cells": "33"
        }
    return mapping_dict


def get_colormap():
    color_map = {
        'Naive B cells': 'cyan', 'Non-classical monocytes': 'black', 'Classical Monocytes': 'magenta', 'Natural killer  cells': 'orange',
        'CD8+ NKT-like cells': 'blue', 'Memory CD4+ T cells': 'darkblue', 'Naive CD8+ T cells': 'lime', 'Platelets': 'pink', 'Pre-B cells':'red',
        'Plasmacytoid Dendritic cells':'yellow', 'Effector CD4+ T cells':'grey', 'Macrophages':'tan', 'Myeloid Dendritic cells':'green',
        'Effector CD8+ T cells':'brown', 'Plasma B cells': 'purple', "Memory B cells": "darkred", "Naive CD4+ T cells": "cornflowerblue",
        'Progenitor cells':'darkgreen', 'γδ-T cells':'darkcyan', 'Eosinophils': 'darkolivegreen', 'Neutrophils': 'darkorchid', 'Basophils': 'darkred',
        'Mast cells': 'darkseagreen', 'Intermediate monocytes': 'darkslateblue', 'Megakaryocyte': 'darkslategrey', 'Endothelial': 'darkturquoise',
        'Erythroid-like and erythroid precursor cells': 'darkviolet', 'HSC/MPP cells': 'deeppink', 'Granulocytes': 'deepskyblue',
        'ISG expressing immune cells': 'dimgray', 'Cancer cells': 'dodgerblue', 'Memory CD8+ T cells': 'darkkhaki', 'Pro-B cells': 'darkorange',
        'Immature B cells': 'darkgoldenrod'
        # 'CD4+ NKT-like cells': 'darkmagenta',
    }
    return color_map
# def get_colormap():
#     color_map = {
#         'Naive B cells': 'red', 'Non-classical monocytes': 'black', 'Classical Monocytes': 'orange', 'Natural killer  cells': 'cyan',
#         'CD8+ NKT-like cells': 'pink', 'Memory CD4+ T cells': 'magenta', 'Naive CD8+ T cells': 'blue', 'Platelets': 'yellow', 'Pre-B cells':'cornflowerblue',
#         'Plasmacytoid Dendritic cells':'lime', 'Effector CD4+ T cells':'grey', 'Macrophages':'tan', 'Myeloid Dendritic cells':'green',
#         'Effector CD8+ T cells':'brown', 'Plasma B cells': 'purple', "Memory B cells": "darkred", "Naive CD4+ T cells": "darkblue",
#         'Progenitor cells':'darkgreen', 'γδ-T cells':'darkcyan', 'Eosinophils': 'darkolivegreen', 'Neutrophils': 'darkorchid', 'Basophils': 'darkred',
#         'Mast cells': 'darkseagreen', 'Intermediate monocytes': 'darkslateblue', 'Megakaryocyte': 'darkslategrey', 'Endothelial': 'darkturquoise',
#         'Erythroid-like and erythroid precursor cells': 'darkviolet', 'HSC/MPP cells': 'deeppink', 'Granulocytes': 'deepskyblue',
#         'ISG expressing immune cells': 'dimgray', 'Cancer cells': 'dodgerblue', 'Memory CD8+ T cells': 'darkkhaki', 'Pro-B cells': 'darkorange',
#         'Immature B cells': 'darkgoldenrod'
#         # 'CD4+ NKT-like cells': 'darkmagenta',
#     }
#     return color_map

def load_data(data_dir, barcode_path):
    # from scipy.sparse import csr_matrix
    # data_dir = "/u/hc2kc/scVAE/pbmc3k/data/"
    # barcode_path = data_dir+'barcode_to_celltype.tsv'
    # adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
    # pbmc3k_genes = pd.read_csv("/u/hc2kc/scVAE/pbmc3k/data/genes.tsv", sep='\t', header=None, names=["gene_ids", "gene_names"])
    # adata.var_names = pbmc3k_genes['gene_ids']

    # sorted_unique_gene_names = pd.read_csv("/u/hc2kc/scVAE/bulk2sc/unique_gene_names.csv", header=0)
    
    
    # adata.obs_names = adata.obs_names.astype(str)
    # num_cells = adata.n_obs
    # num_genes = len(sorted_unique_gene_names)
    # new_data_matrix = csr_matrix((num_cells, num_genes), dtype=np.float32)

    # # Create a new AnnData object with this matrix
    # new_adata = sc.AnnData(new_data_matrix)
    # new_adata.obs_names = adata.obs_names
    # new_adata.var_names = sorted_unique_gene_names['gene_ids'].astype(str)  # Convert to string explicitly

    # # Map the old gene indices to the new indices and fill in the data
    # old_gene_indices = {gene: i for i, gene in enumerate(adata.var_names)}
    # new_gene_indices = {gene: i for i, gene in enumerate(new_adata.var_names)}

    # for old_gene, old_index in old_gene_indices.items():
    #     new_index = new_gene_indices[old_gene]
    #     new_adata.X[:, new_index] = adata.X[:, old_index]
    # new_adata.write("/u/hc2kc/scVAE/pbmc3k/data/matrix.h5ad")
    # adata = sc.read_h5ad("/u/hc2kc/scVAE/pbmc3k/data/matrix.h5ad")

    # ------------------------------------------------
    # data_dir = "/u/hc2kc/scVAE/pbmc10k/data/"
    # barcode_path = data_dir+'barcode_to_celltype.csv'

    # adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
    # pbmc10k_genes = pd.read_csv("/u/hc2kc/scVAE/pbmc10k/data/features.tsv", sep='\t', header=None, names=["gene_ids", "gene_names", "type"])
    # adata.var_names = pbmc10k_genes['gene_ids']

    # sorted_unique_gene_names = pd.read_csv("/u/hc2kc/scVAE/bulk2sc/unique_gene_names.csv", header=0)
    

    
    # adata.obs_names = adata.obs_names.astype(str)
    # num_cells = adata.n_obs
    # num_genes = len(sorted_unique_gene_names)
    # new_data_matrix = csr_matrix((num_cells, num_genes), dtype=np.float32)

    # # Create a new AnnData object with this matrix
    # new_adata = sc.AnnData(new_data_matrix)
    # new_adata.obs_names = adata.obs_names
    # new_adata.var_names = sorted_unique_gene_names['gene_ids'].astype(str)  # Convert to string explicitly

    # # Map the old gene indices to the new indices and fill in the data
    # old_gene_indices = {gene: i for i, gene in enumerate(adata.var_names)}
    # new_gene_indices = {gene: i for i, gene in enumerate(new_adata.var_names)}

    # for old_gene, old_index in old_gene_indices.items():
    #     new_index = new_gene_indices[old_gene]
    #     new_adata.X[:, new_index] = adata.X[:, old_index]

    # # Save the new AnnData object as mtx file.
    # # sc.write("pbmc10k_new.mtx", new_adata)

    # new_adata.write("/u/hc2kc/scVAE/pbmc10k/data/pbmc10k_new.h5ad")

    # print("Saved new 10k AnnData object as mtx file.")

    # ------------------------------------------------

    # data_dir = "/u/hc2kc/scVAE/pbmc68k/data/"
    # barcode_path = data_dir+'barcode_to_celltype.csv'

    # adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)
    # pbmc68k_genes = pd.read_csv("/u/hc2kc/scVAE/pbmc68k/data/genes.tsv", sep='\t', header=None, names=["gene_ids", "gene_names"])
    # adata.var_names = pbmc68k_genes['gene_ids']

    # sorted_unique_gene_names = pd.read_csv("/u/hc2kc/scVAE/bulk2sc/unique_gene_names.csv", header=0)

    
    # adata.obs_names = adata.obs_names.astype(str)
    # num_cells = adata.n_obs
    # num_genes = len(sorted_unique_gene_names)
    # new_data_matrix = csr_matrix((num_cells, num_genes), dtype=np.float32)

    # # Create a new AnnData object with this matrix
    # new_adata = sc.AnnData(new_data_matrix)
    # new_adata.obs_names = adata.obs_names
    # new_adata.var_names = sorted_unique_gene_names['gene_ids'].astype(str)  # Convert to string explicitly

    # # Map the old gene indices to the new indices and fill in the data
    # old_gene_indices = {gene: i for i, gene in enumerate(adata.var_names)}
    # new_gene_indices = {gene: i for i, gene in enumerate(new_adata.var_names)}

    # for old_gene, old_index in old_gene_indices.items():
    #     new_index = new_gene_indices[old_gene]
    #     new_adata.X[:, new_index] = adata.X[:, old_index]

    # # Save the new AnnData object as mtx file.
    # sc.write("pbmc68k_new.mtx", new_adata)
    # print("Saved new 68k AnnData object as mtx file.")

    # # Load the new AnnData object.
    # new_adata.write("/u/hc2kc/scVAE/pbmc68k/data/pbmc68k_new.h5ad")
    # import pdb; pdb.set_trace()
    mapping_dict = get_celltype2strint_dict()

    adata_3k = sc.read_h5ad("/u/hc2kc/scVAE/pbmc3k/data/matrix.h5ad")
    data_dir = "/u/hc2kc/scVAE/pbmc3k/data/"
    barcode_path = data_dir+'barcode_to_celltype.tsv'
    # Read barcodes_with_labels tsv file.
    barcodes_with_labels_3k = pd.read_csv(barcode_path, sep=',', header=None).iloc[1:]
    barcodes_with_labels_3k.columns = ['barcodes', 'labels']

    # Remove rows with 'unknown' or NaN labels
    barcodes_with_labels_3k = barcodes_with_labels_3k[(barcodes_with_labels_3k['labels'] != 'Unknown')]

    # Cleaned labels after filtering
    labels_3k = barcodes_with_labels_3k['labels'].values

    filtered_barcodes = barcodes_with_labels_3k['barcodes'].values
    adata_3k = adata_3k[adata_3k.obs.index.isin(filtered_barcodes)]

    adata_3k.obs['barcodes'] = adata_3k.obs.index

    adata_3k.obs = adata_3k.obs.reset_index(drop=True)
    adata_3k.obs = adata_3k.obs.merge(barcodes_with_labels_3k, on='barcodes', how='left')
    adata_3k.X = adata_3k.X.toarray()
    adata_3k.obs.index = adata_3k.obs.index.astype(str)

    # Map labels to integers as done before
    adata_3k.obs['labels'] = adata_3k.obs['labels'].replace(mapping_dict)
    adata_3k.obs['labels'] = adata_3k.obs['labels'].astype('category')
    

    adata_10k = sc.read_h5ad("/u/hc2kc/scVAE/pbmc10k/data/matrix.h5ad")
    data_dir = "/u/hc2kc/scVAE/pbmc10k/data/"
    barcode_path = data_dir+'barcode_to_celltype.csv'
    # Read barcodes_with_labels tsv file.
    barcodes_with_labels_10k = pd.read_csv(barcode_path, sep=',', header=None).iloc[1:]
    barcodes_with_labels_10k.columns = ['barcodes', 'labels']

    # Remove rows with 'unknown' or NaN labels
    barcodes_with_labels_10k = barcodes_with_labels_10k[(barcodes_with_labels_10k['labels'] != 'Unknown')]

    # Cleaned labels after filtering
    labels_10k = barcodes_with_labels_10k['labels'].values

    filtered_barcodes = barcodes_with_labels_10k['barcodes'].values
    adata_10k = adata_10k[adata_10k.obs.index.isin(filtered_barcodes)]

    adata_10k.obs['barcodes'] = adata_10k.obs.index

    adata_10k.obs = adata_10k.obs.reset_index(drop=True)
    adata_10k.obs = adata_10k.obs.merge(barcodes_with_labels_10k, on='barcodes', how='left')
    adata_10k.X = adata_10k.X.toarray()
    adata_10k.obs.index = adata_10k.obs.index.astype(str)

    # Map labels to integers as done before
    adata_10k.obs['labels'] = adata_10k.obs['labels'].replace(mapping_dict)
    adata_10k.obs['labels'] = adata_10k.obs['labels'].astype('category')


    adata_68k = sc.read_h5ad("/u/hc2kc/scVAE/pbmc68k/data/matrix.h5ad")
    data_dir = "/u/hc2kc/scVAE/pbmc68k/data/"
    barcode_path = data_dir+'barcode_to_celltype.csv'
    # Read barcodes_with_labels tsv file.
    barcodes_with_labels_68k = pd.read_csv(barcode_path, sep=',', header=None).iloc[1:]
    barcodes_with_labels_68k.columns = ['barcodes', 'labels']

    # Remove rows with 'unknown' or NaN labels
    barcodes_with_labels_68k = barcodes_with_labels_68k[(barcodes_with_labels_68k['labels'] != 'Unknown')]

    # Cleaned labels after filtering
    labels_68k = barcodes_with_labels_68k['labels'].values

    filtered_barcodes = barcodes_with_labels_68k['barcodes'].values
    adata_68k = adata_68k[adata_68k.obs.index.isin(filtered_barcodes)]

    adata_68k.obs['barcodes'] = adata_68k.obs.index

    adata_68k.obs = adata_68k.obs.reset_index(drop=True)
    adata_68k.obs = adata_68k.obs.merge(barcodes_with_labels_68k, on='barcodes', how='left')
    adata_68k.X = adata_68k.X.toarray()
    adata_68k.obs.index = adata_68k.obs.index.astype(str)

    # Map labels to integers as done before
    adata_68k.obs['labels'] = adata_68k.obs['labels'].replace(mapping_dict)
    adata_68k.obs['labels'] = adata_68k.obs['labels'].astype('category')

    adata_3k.X = adata_3k.X.toarray() if not isinstance(adata_3k.X, np.ndarray) else adata_3k.X
    adata_10k.X = adata_10k.X.toarray() if not isinstance(adata_10k.X, np.ndarray) else adata_10k.X
    adata_68k.X = adata_68k.X.toarray() if not isinstance(adata_68k.X, np.ndarray) else adata_68k.X
    
    adata = sc.concat([adata_3k, adata_10k, adata_68k], join='outer', index_unique='-')
    # Merge labels_3k, labels_10k, and labels_68k.
    labels = np.concatenate([labels_3k, labels_10k, labels_68k])
    

    adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict)
    adata.obs['labels'] = adata.obs['labels'].astype('category')
    
    labels = adata.obs['labels']

    label_to_int = {}
    max_int_label = -1

    for label in labels.unique():
        if label.isdigit():
            int_val = int(label)
            label_to_int[label] = int_val
            max_int_label = max(max_int_label, int_val)
        else:
            if label not in label_to_int:
                max_int_label += 1
                label_to_int[label] = max_int_label

    # Map the labels in your DataFrame to integers
    int_labels = labels.map(label_to_int)

    # Convert the Series of integers to a LongTensor
    labels = torch.LongTensor(int_labels.values)

    print(labels)
    X_tensor = torch.Tensor(adata.X)
    # randomly sample 80% of the data.
    rand_index = np.random.choice(X_tensor.shape[0], int(0.8*X_tensor.shape[0]), replace=False)
    # Select those not in the random index to be the test set.
    X_tensor = X_tensor[rand_index]
    labels = labels[rand_index]

    mask = np.isin(np.arange(X_tensor.shape[0]), rand_index, invert=True)
    X_tensor = X_tensor[mask]
    labels = labels[mask]
    print(f"Shape of X_tensor: {X_tensor.shape}")
    print(f"Shape of labels: {labels.shape}")

    dataset = TensorDataset(X_tensor, labels)

    # Cell type fractions
    cell_type_fractions = np.unique(adata.obs['labels'].values, return_counts=True)[1]/len(adata.obs['labels'].values)
    
    return dataset, X_tensor, labels, cell_type_fractions, mapping_dict


def get_saved_GMM_params(mus_path, vars_path):
    gmm_mus_celltypes = torch.load(mus_path).squeeze().T
    gmm_vars_celltypes = torch.load(vars_path).squeeze().T
    return gmm_mus_celltypes, gmm_vars_celltypes


def configure(data_dir, barcode_path):
    dataset, X_tensor, cell_types_tensor, cell_type_fractions, mapping_dict = load_data(
                                                                                        data_dir=data_dir,
                                                                                        barcode_path=barcode_path,
                                                                                        )
    num_cells = X_tensor.shape[0]
    num_genes = X_tensor.shape[1]

    parser = argparse.ArgumentParser(description='Process neural network parameters.')
    
    args = parser.parse_args()
    args.num_cells = num_cells
    # args.learning_rate = 1e-3
    args.hidden_dim = 600
    args.latent_dim = 300
    args.train_GMVAE_epochs = 400
    args.bulk_encoder_epochs = 1000
    # args.dropout = 0.05
    args.batch_size = num_cells
    args.input_dim = num_genes

    dataloader = DataLoader(dataset, batch_size=num_cells//7, shuffle=True, drop_last=True)

    args.dataloader = dataloader
    args.cell_types_tensor = cell_types_tensor

    args.mapping_dict = mapping_dict
    args.color_map = get_colormap()
    args.K = 34
    unique_cell_types = np.unique(cell_types_tensor)
    cell_type_fractions_ = []

    # Create a dictionary mapping from cell type to its fraction
    cell_type_to_fraction = {cell_type: fraction for cell_type, fraction in zip(unique_cell_types, cell_type_fractions)}
    

    for i in range(args.K):
        # Append the fraction if the cell type is present, else append 0
        cell_type_fractions_.append(cell_type_to_fraction.get(i, 0))
    
    args.cell_type_fractions = torch.FloatTensor(np.array(cell_type_fractions_))
    print(args.cell_type_fractions)
    print("@@")
    
    args.X_tensor = X_tensor
    label_map = {str(v): k for k, v in mapping_dict.items()}
    args.label_map = label_map

    print('Configuration is complete.')
    return args
