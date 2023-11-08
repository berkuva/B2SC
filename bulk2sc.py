import torch
import torch.nn as nn
import torch.nn.functional as F
from models import bulkEncoder, scDecoder




def generate(encoder, decoder, date_loader):
    generated_list = []
    labels_list = []
    for batch_idx, (data, labels) in enumerate(date_loader):
        data = data.to(device)
        iter_loss = 0
        bulk_data = data.sum(dim=0)
        bulk_data = bulk_data.unsqueeze(0)


        # Forward pass
        mus, vars, pis = encoder(bulk_data)
        mus = mus.squeeze()
        vars = vars.squeeze()
        pis = pis.squeeze()

        sampled_indices = torch.multinomial(pis, 1).squeeze()
        mu = mus[sampled_indices]
        var = vars[sampled_indices]
        z = decoder.reparameterize(mu, var)
        generated = decoder(z)
        
        generated_list.append(generated)
        labels_list.append(sampled_indices.item())
    
    generated_tensor = torch.stack(generated_list)[0]
    
    return generated_tensor, labels_list


if __name__ == "__main__":
    from configure import configure
    data_dir = "./sample_data/"
    barcode_path = data_dir+'barcode_to_celltype.csv'

    args = configure(data_dir, barcode_path)

    device = args.device
    
    input_dim = args.X_tensor.shape[-1]
    h_dim = args.hidden_dim
    z_dim = args.gmm_mus_celltypes.shape[-1]
    latent_dim =  args.gmm_mus_celltypes.shape[-1]
    
    # Create VAE and optimizer
    decoder = scDecoder(input_dim, h_dim, z_dim)
    encoder = bulkEncoder(input_dim, h_dim, latent_dim, z_dim)

    # Instantiate the combined model
    decoder_state_dict = torch.load("scDecoder_model.pt")
    decoder.load_state_dict(decoder_state_dict, strict=True)

    encoder_state_dict = torch.load("bulkEncoder_model.pt")
    encoder.load_state_dict(encoder_state_dict, strict=True)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    dataloader = args.dataloader

    batch_size = args.batch_size

    generated_aggregate=[]
    y = []
    for i in range(batch_size): # Each batch creates one cell.
        if (i+1) % 100 == 0:
            print(f"Generating {i+1}th cell...")
        gt, label = generate(encoder, decoder, dataloader)
        generated_aggregate.append(gt)
        for l in label:
            y.append(l)

