import torch
import umap
import matplotlib.pyplot as plt
import numpy as np

def generate_(encoder, GMVAE_model, dataloader, device='cuda'):
    generated_list = []
    labels_list = []
    # Generate one cell per batch.
    for _, (data, labels) in enumerate(dataloader):
        data = data.to(device)

        bulk_data = data.sum(dim=0)
        bulk_data = bulk_data.unsqueeze(0)

        # Forward pass
        mus, logvars, pis = encoder(bulk_data)
        mus = mus.squeeze()
        logvars = logvars.squeeze()
        pis = pis.squeeze()
 #        pis = torch.FloatTensor([5.7450e-02, 2.6600e-02, 3.6010e-01, 1.2500e-03, 4.7200e-02,2.9185e-01,
 # 9.0000e-04, 2.0460e-01, 3.0000e-04, 1.7000e-03, 1.5000e-04, 0,0,1.0000e-04,0,0,
 # 7.5000e-03,0,5.0000e-05,0,1.0000e-04,0,0,0,1.0000e-04,0,0,0,0,0,0,5.0000e-05,0,0]).to(device)

        generated, k = GMVAE_model.module.decode_bulk(mus, logvars, pis)
        
        generated_list.append(generated)
        labels_list.append(k.item())
    
    generated_tensor = torch.stack(generated_list)
    
    return generated_tensor, labels_list


def generate(encoder, GMVAE_model, dataloader, num_cells, mapping_dict, color_map, device='cuda'):
    encoder.eval()
    GMVAE_model.eval()
    encoder = encoder.to(device)
    GMVAE_model = GMVAE_model.to(device)
        
    generated_aggregate = []
    sampled_celltypes = []
    
    print(f"Generating {num_cells} cells...")

    for i in range(0):
        if (i + 1) % 100 == 0:
            print(f"Generating {i + 1}th cell...")
        gt, label = generate_(encoder, GMVAE_model, dataloader, device=device)
        
        # Append gt to generated_aggregate without changing its type
        generated_aggregate.append(gt)
        for l in label:
            sampled_celltypes.append(l)
        
        if (i + 1) % 500 == 0:
            # Process and save the data every 100 iterations

            # Convert the list of tensors to a single tensor
            generated_aggregate_tensor = torch.stack(generated_aggregate)
            generated_aggregate_tensor = generated_aggregate_tensor.squeeze()
            generated_aggregate_tensor = generated_aggregate_tensor.cpu()
            
            # Convert sampled_celltypes to a tensor
            sampled_celltypes_tensor = torch.LongTensor(sampled_celltypes)
            
            # Reshape generated_aggregate_tensor
            input_dim = generated_aggregate_tensor.shape[-1]
            generated_aggregate_tensor = generated_aggregate_tensor.reshape(-1, input_dim)
            
            # Save tensors
            torch.save(generated_aggregate_tensor, f"saved_files/generated_aggregate_tensor.pt")
            torch.save(sampled_celltypes_tensor, f"saved_files/sampled_celltypes.pt")
            
            print(f"{i + 1}th generated_aggregate_tensor saved.")


    generated_aggregate_tensor = torch.load("saved_files/generated_aggregate_tensor.pt")
    sampled_celltypes = torch.load("saved_files/sampled_celltypes.pt")
    sampled_celltypes = torch.LongTensor(sampled_celltypes)
    
    print("Generated cell type proportion:")
    print(np.unique(sampled_celltypes, return_counts=True)[0])
    print(np.unique(sampled_celltypes, return_counts=True)[1]/len(sampled_celltypes))

    

    reducer = umap.UMAP()

    X_2d = reducer.fit_transform(generated_aggregate_tensor.detach().numpy())
    plt.figure(figsize=(6, 5))

    # set label_map as the inverse of mapping_dict
    label_map = {v: k for k, v in mapping_dict.items()}

    # Assuming X_2d, sampled_celltypes, label_map, and color_map are defined
    unique_celltypes, counts = torch.unique(sampled_celltypes, return_counts=True)
    proportions = counts.float() / len(sampled_celltypes)

    for i in range(len(list(label_map.keys()))):
        if i in unique_celltypes:
            if proportions[unique_celltypes.tolist().index(i)] > 0.001:
                label = i
                plt.scatter(
                    X_2d[sampled_celltypes==label, 0],
                    X_2d[sampled_celltypes==label, 1],
                    c=color_map[label_map[str(label)]],
                    label=label_map[str(label)]
                )
    # plt.legend("")
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')


    # remove x and y ticks.
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Generated Cells")
    plt.savefig(f"saved_files/generated_umap.png")
    plt.close()


