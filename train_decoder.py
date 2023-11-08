import torch
import torch.nn as nn
import torch.nn.functional as F
from models import scDecoder


def loss_function(x_reconst, original_x):
    MSE = F.mse_loss(x_reconst, original_x, reduction='sum')
    L1 = F.l1_loss(x_reconst, original_x, reduction='sum')
    return MSE + L1


def train(epoch, model, optimizer, dataloader, cell_types_tensor, gmm_mus_celltypes, gmm_vars_celltypes, device, max_epochs):

    model.train()

    for batch_idx, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        
        zs = []
        for i in range(len(data)):
            si = cell_types_tensor[i]
            mu = gmm_mus_celltypes[si]
            var = gmm_vars_celltypes[si]
            z = mu + torch.sqrt(var) * torch.randn_like(var)
            zs.append(z)

        zs_stacked = torch.stack(zs).squeeze()
        reconstructed = model(zs_stacked)
        
        loss = loss_function(reconstructed, data)

    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss
        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch}/{max_epochs}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    from configure import configure
    data_dir = "/u/hc2kc/scVAE/pbmc1k/data/"
    barcode_path = data_dir+"barcode_to_celltype.csv"

    gmm_mus_path = "./gmm_parameters/gmm_mu_c.pt"
    gmm_vars_path = "./gmm_parameters/gmm_var_c.pt"

    args = configure(data_dir, barcode_path, gmm_mus_path, gmm_vars_path)

    dataloader = args.dataloader
    cell_types_tensor = args.cell_types_tensor
    gmm_mus_celltypes = args.gmm_mus_celltypes
    gmm_vars_celltypes = args.gmm_vars_celltypes
    device = args.device
    input_dim = args.X_tensor.shape[-1]
    h_dim = args.hidden_dim
    z_dim = args.gmm_mus_celltypes.shape[-1]
    
    # Create VAE and optimizer
    model = scDecoder(input_dim, h_dim, z_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Weight initialization
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight) 
            m.bias.data.fill_(0.01)

    model.apply(init_weights) 
    model = model.to(args.device)

    epochs = args.train_decoder_epochs
    for epoch in range(-1, epochs):
        train(epoch, model, optimizer, dataloader, cell_types_tensor, gmm_mus_celltypes, gmm_vars_celltypes, device, epochs)
    torch.save(model.state_dict(), 'scDecoder_model.pt')