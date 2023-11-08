import torch
import torch.nn as nn
import torch.nn.functional as F
from models import bulkEncoder


def train(optimizer, model, train_loader, gmm_mus_celltypes, gmm_vars_celltypes, cell_proportions, device):

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        iter_loss = 0
        bulk_data = data.sum(dim=0)
        bulk_data = bulk_data.unsqueeze(0)

        # Forward pass
        mus, vars, pis = model(bulk_data)
        mus = mus.squeeze()
        vars = vars.squeeze()
        pis = pis.squeeze()
        mus_loss = F.mse_loss(mus, gmm_mus_celltypes) + F.l1_loss(mus, gmm_mus_celltypes)
        vars_loss = F.mse_loss(vars, gmm_vars_celltypes) + F.l1_loss(mus, gmm_mus_celltypes)
        pis_loss = F.mse_loss(pis, cell_proportions)

        loss = mus_loss + vars_loss + pis_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print('Epoch: {} \tLoss: {:.6f}'.format(epoch+1, loss.item()))
        print('Mus loss: {:.6f}'.format(mus_loss.item()))
        print('Vars loss: {:.6f}'.format(vars_loss.item()))
        print('Pis loss: {:.6f}'.format(pis_loss.item()))
        print('------------------------')
    

if __name__ == "__main__":
    from configure import configure
    data_dir = "/u/hc2kc/scVAE/pbmc1k/data/"
    barcode_path = data_dir+"barcode_to_celltype.csv"

    gmm_mus_path = "./gmm_parameters/gmm_mu_c.pt"
    gmm_vars_path = "./gmm_parameters/gmm_var_c.pt"

    args = configure(data_dir, barcode_path, gmm_mus_path, gmm_vars_path)

    input_dim = args.X_tensor.shape[-1]
    h_dim = args.hidden_dim
    z_dim = args.gmm_mus_celltypes.shape[-1]
    latent_dim =  args.gmm_mus_celltypes.shape[-1]

    model = bulkEncoder(input_dim,h_dim, latent_dim, z_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    dataloader = args.dataloader
    gmm_mus_celltypes = args.gmm_mus_celltypes
    gmm_vars_celltypes = args.gmm_vars_celltypes
    device = args.device
    cell_proportions = torch.FloatTensor(args.cell_proportions).to(device)

    # Weight initialization
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight) 
            m.bias.data.fill_(0.01)

    model.apply(init_weights) 
    model = model.to(args.device)

    epochs = args.bulk_epochs
    for epoch in range(-1, epochs):
        train(optimizer, model, dataloader, gmm_mus_celltypes, gmm_vars_celltypes, cell_proportions, device)
    
    # Save the model checkpoint.
    torch.save(model.state_dict(), "bulkEncoder_model.pt")
