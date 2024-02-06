import scanpy as sc
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import matplotlib.pyplot as plt


# Set random seed.
# np.random.seed(4)
# Set torch seed.
# torch.manual_seed(4)

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
        'Naive B cells': 'red', 'Non-classical monocytes': 'black', 'Classical Monocytes': 'orange', 'Natural killer  cells': 'cyan',
        'CD8+ NKT-like cells': 'pink', 'Memory CD4+ T cells': 'magenta', 'Naive CD8+ T cells': 'blue', 'Platelets': 'yellow', 'Pre-B cells':'cornflowerblue',
        'Plasmacytoid Dendritic cells':'lime', 'Effector CD4+ T cells':'grey', 'Macrophages':'tan', 'Myeloid Dendritic cells':'green',
        'Effector CD8+ T cells':'brown', 'Plasma B cells': 'purple', "Naive CD4+ T cells": "darkblue", "Memory B cells": "darkred",
        'Progenitor cells':'darkgreen', 'γδ-T cells':'darkcyan', 'Pro-B cells': 'darkorange', 'Immature B cells': 'darkgoldenrod',
        'Memory CD8+ T cells': 'darkkhaki', 'CD4+ NKT-like cells': 'darkmagenta', 'Eosinophils': 'darkolivegreen', 'Neutrophils': 'darkorchid',
        'Basophils': 'darkred', 'Mast cells': 'darkseagreen', 'Intermediate monocytes': 'darkslateblue', 'Megakaryocyte': 'darkslategrey',
        'Endothelial': 'darkturquoise', 'Erythroid-like and erythroid precursor cells': 'darkviolet', 'HSC/MPP cells': 'deeppink',
        'Granulocytes': 'deepskyblue', 'ISG expressing immune cells': 'dimgray', 'Cancer cells': 'dodgerblue'
    }
    return color_map


def get_colormap():
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

    color_map = {
        'Naive B cells': rgb_to_hex((0, 110, 169)), #0
        'Non-classical monocytes': rgb_to_hex((255, 117, 39)), #1
        'Classical Monocytes': rgb_to_hex((43, 162, 116)), #2
        'Natural killer  cells': rgb_to_hex((218, 35, 42)), #3
        'CD8+ NKT-like cells': rgb_to_hex((168, 64, 246)), #4
        'Memory CD4+ T cells': rgb_to_hex((113, 76, 67)), #5
        'Naive CD8+ T cells': rgb_to_hex((233, 127, 193)), #6
        "Platelets": rgb_to_hex((173, 182, 97)), #7
        'Pre-B cells': rgb_to_hex((0, 183, 199)), #8
        'Plasmacytoid Dendritic cells': rgb_to_hex((161, 192, 227)), #9
        'Effector CD4+ T cells': rgb_to_hex((255, 183, 124)), #10
        'Macrophages': rgb_to_hex((133, 218, 132)), #11
        'Myeloid Dendritic cells': rgb_to_hex((255, 187, 186)), #12
        'Effector CD8+ T cells': rgb_to_hex((191, 167, 206)), #13
        'Plasma B cells': rgb_to_hex((0, 0, 0)), #14
        "Memory B cells": rgb_to_hex((0, 0, 0)), #15
        "Naive CD4+ T cells": 'black', #16",
        'Progenitor cells':'black', #17
        'γδ-T cells':'black', #18
        'Eosinophils':'black', #19
        'Neutrophils':'black', #20
        'Basophils':'black', #21
        'Mast cells':'black', #22
        'Intermediate monocytes': 'black', #23
        'Megakaryocyte': 'black', #24
        'Endothelial': 'black', #25
        'Erythroid-like and erythroid precursor cells': 'black', #26
        'HSC/MPP cells': 'black', #27
        'Granulocytes': 'black', #28
        'ISG expressing immune cells': 'black', #29
        'Cancer cells': 'black', #30
        "Memory CD8+ T cells": "black", #31
        "Pro-B cells": "black", #32
        "Immature B cells": "black" #33
    }
    return color_map


def load_data(data_dir):
    # adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)    

    # sc.tl.pca(adata, svd_solver='arpack')
    # sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    # sc.tl.umap(adata)
    # sc.tl.leiden(adata, resolution=0.25)
    
    # # adata.obs['leiden'].values.astype(int).to_csv('leiden_clusters.csv')
    # adata.obs['leiden'].astype(int).to_csv('leiden_clusters.csv', header=True)

    # sc.pl.umap(adata, color=['leiden'], legend_loc='on data', title='', frameon=False, save='_leiden_notnorm.png')
    # plt.savefig('umap_plot.png', dpi=300)

    # labels = adata.obs.leiden.values
    # save labels to file.
    # pd.DataFrame(labels).to_csv('/u/hc2kc/scVAE/paired/data/leiden_clusters.csv', header=True)

    # from scipy.sparse import csr_matrix
    # import tqdm
    # sorted_unique_gene_names = pd.read_csv("/u/hc2kc/scVAE/bulk2sc/unique_gene_names.csv", header=0)
    
    # bulk_genes = pd.read_csv("/u/hc2kc/scVAE/bulk2sc/genes.tsv", header=None, sep='\t').values[:,0]
    
    # adata.obs_names = adata.obs_names.astype(str)
    # adata.var_names = bulk_genes.astype(str)
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
    
    # for old_gene, old_index in tqdm.tqdm(old_gene_indices.items()):
    #     new_index = new_gene_indices[old_gene]
    #     new_adata.X[:, new_index] = adata.X[:, old_index]
    # new_adata.write("/u/hc2kc/scVAE/paired/data/matrix.h5ad")
    adata = sc.read_h5ad("/u/hc2kc/scVAE/paired/data/matrix.h5ad")
    labels = pd.read_csv("/u/hc2kc/scVAE/paired/data/leiden_clusters.csv", header=0, index_col=0)#.values

    unique_values = labels['0'].unique()
    mapping_dict = get_celltype2strint_dict()

    label_to_int = {}
    max_int_label = -1  # Assuming max_int_label is initialized somewhere before this loop

    # for label in unique_values:
    #     import pdb; pdb.set_trace()
    #     if label.isdigit():  # Check if the label is numeric
    #         int_val = int(label)
    #         label_to_int[label] = int_val
    #         max_int_label = max(max_int_label, int_val)
    #     else:
    #         if label not in label_to_int:
    #             max_int_label += 1
    #             label_to_int[label] = max_int_label
    # int_labels = labels.map(label_to_int)
    # labels = torch.LongTensor(int_labels.codes)
    for label in unique_values:
        int_val = int(label)
        label_to_int[label] = int_val
        max_int_label = max(max_int_label, int_val)

    int_labels = labels['0'].replace(label_to_int)

    labels = torch.LongTensor(int_labels)

    adata.obs['labels'] = labels

    X_tensor = torch.Tensor(adata.X.toarray())

    rand_index = np.random.choice(X_tensor.shape[0], int(0.8*X_tensor.shape[0]), replace=False)
    # Select those not in the random index to be the test set.
    X_tensor = X_tensor[rand_index]
    labels = labels[rand_index]

    # mask = np.isin(np.arange(X_tensor.shape[0]), rand_index, invert=True)
    # X_tensor = X_tensor[mask]
    # labels = labels[mask]
    
    print(f"Shape of X_tensor: {X_tensor.shape}")
    print(f"Shape of labels: {labels.shape}")

    dataset = TensorDataset(X_tensor, labels)

    cell_type_fractions = np.unique(adata.obs['labels'].values, return_counts=True)[1]/len(adata.obs['labels'].values)
    return dataset, X_tensor, labels, cell_type_fractions, mapping_dict

def get_saved_GMM_params(mus_path, vars_path):
    gmm_mus_celltypes = torch.load(mus_path).squeeze().T
    gmm_vars_celltypes = torch.load(vars_path).squeeze().T
    return gmm_mus_celltypes, gmm_vars_celltypes


def configure(data_dir, barcode_path=None):
    dataset, X_tensor, cell_types_tensor, cell_type_fractions, mapping_dict = load_data(data_dir=data_dir)
    num_cells = X_tensor.shape[0]
    num_genes = X_tensor.shape[1]

    parser = argparse.ArgumentParser(description='Process neural network parameters.')
    
    args = parser.parse_args()
    args.num_cells = num_cells

    args.hidden_dim = 600
    args.latent_dim = 300
    args.train_GMVAE_epochs = 100
    args.bulk_encoder_epochs = 100

    args.batch_size = num_cells
    args.input_dim = num_genes
    
    dataloader = DataLoader(dataset, batch_size=num_cells//25, shuffle=True, drop_last=True)#

    args.dataloader = dataloader
    args.cell_types_tensor = cell_types_tensor

    args.mapping_dict = mapping_dict
    args.color_map = get_colormap()
    args.K = 34
    unique_cell_types = np.unique(cell_types_tensor)
    cell_type_fractions_ = []

    cell_type_to_fraction = {cell_type: fraction for cell_type, fraction in zip(unique_cell_types, cell_type_fractions)}
    
    for i in range(args.K):
        cell_type_fractions_.append(cell_type_to_fraction.get(i, 0))
    
    args.cell_type_fractions = torch.FloatTensor(np.array(cell_type_fractions_))
    print(args.cell_type_fractions)
    print("@@")
    
    args.X_tensor = X_tensor
    # import pdb; pdb.set_trace()
    label_map = {str(v): k for k, v in mapping_dict.items()}
    args.label_map = label_map

    print('Configuration is complete.')
    return args
