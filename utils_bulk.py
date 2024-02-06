import scanpy as sc
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import matplotlib.pyplot as plt
import tqdm

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


# def get_colormap():
#     color_map = {
#         'Naive B cells': 'red', 'Non-classical monocytes': 'black', 'Classical Monocytes': 'orange', 'Natural killer  cells': 'cyan',
#         'CD8+ NKT-like cells': 'pink', 'Memory CD4+ T cells': 'magenta', 'Naive CD8+ T cells': 'blue', 'Platelets': 'yellow', 'Pre-B cells':'cornflowerblue',
#         'Plasmacytoid Dendritic cells':'lime', 'Effector CD4+ T cells':'grey', 'Macrophages':'tan', 'Myeloid Dendritic cells':'green',
#         'Effector CD8+ T cells':'brown', 'Plasma B cells': 'purple', "Naive CD4+ T cells": "darkblue", "Memory B cells": "darkred",
#         'Progenitor cells':'darkgreen', 'γδ-T cells':'darkcyan', 'Pro-B cells': 'darkorange', 'Immature B cells': 'darkgoldenrod',
#         'Memory CD8+ T cells': 'darkkhaki', 'CD4+ NKT-like cells': 'darkmagenta', 'Eosinophils': 'darkolivegreen', 'Neutrophils': 'darkorchid',
#         'Basophils': 'darkred', 'Mast cells': 'darkseagreen', 'Intermediate monocytes': 'darkslateblue', 'Megakaryocyte': 'darkslategrey',
#         'Endothelial': 'darkturquoise', 'Erythroid-like and erythroid precursor cells': 'darkviolet', 'HSC/MPP cells': 'deeppink',
#         'Granulocytes': 'deepskyblue', 'ISG expressing immune cells': 'dimgray', 'Cancer cells': 'dodgerblue'
#     }
#     return color_map


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


def load_data(data_dir, barcode_path=None):
    # from scipy.sparse import csr_matrix
    import scipy.sparse as sp

    # Load the sorted unique gene names
    sorted_unique_gene_names = pd.read_csv("/u/hc2kc/scVAE/bulk2sc/unique_gene_names.csv", header=0, squeeze=True)

    # Load the data
    adata = pd.read_csv("/u/hc2kc/scVAE/paired/data/GSE132044_HEK293_PBMC_TPM_bulk.tsv", sep="\t", index_col=0)

    # Extract the PBMC1 column
    pbmc_data = adata['PBMC2']/3000

    # Initialize a new DataFrame with zeros for all sorted unique gene names
    new_data = pd.DataFrame(0, index=sorted_unique_gene_names, columns=['PBMC2'])

    # Update this new DataFrame with existing values from pbmc1_data, if present
    for gene in pbmc_data.index:
        gene_id = gene.split("_")[0]  # Assuming gene IDs need to be extracted this way
        if gene_id in new_data.index:
            new_data.loc[gene_id, 'PBMC2'] = pbmc_data.loc[gene]

    # Now new_data contains all genes from sorted_unique_gene_names with values from PBMC1 where available and 0 elsewhere
    # Convert this to an AnnData object for further analysis with Scanpy
    new_adata = sc.AnnData(new_data)

    # new_adata.write("/u/hc2kc/scVAE/bulk2sc/bulk_matrix.h5ad")

    # import pdb; pdb.set_trace()
    adata = new_adata


    mapping_dict = get_celltype2strint_dict()

   
    X_tensor = torch.Tensor(adata.X).reshape(1,-1)
    
    print(f"Shape of X_tensor: {X_tensor.shape}")
    dataset = TensorDataset(X_tensor)
    labels = None
    cell_type_fractions = None
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
    # args.learning_rate = 1e-3
    args.hidden_dim = 600
    args.latent_dim = 300
    args.train_GMVAE_epochs = 500
    args.bulk_encoder_epochs = 2000
    # args.dropout = 0.05
    args.batch_size = num_cells
    args.input_dim = num_genes
    
    dataloader = DataLoader(dataset, batch_size=num_cells, shuffle=True, drop_last=True)#//25

    args.dataloader = dataloader
    args.cell_types_tensor = cell_types_tensor

    args.mapping_dict = mapping_dict
    args.color_map = get_colormap()
    args.K = 34
    unique_cell_types = np.unique(cell_types_tensor)

    args.X_tensor = X_tensor
    label_map = {str(v): k for k, v in mapping_dict.items()}
    args.label_map = label_map

    print('Configuration is complete.')
    return args
