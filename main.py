import torch
import torch.nn as nn
from utils import configure
import os

# Train GMVAE. Refer to train_GMVAE.py for the implementation and model checkpoint path.
def train_model_GMVAE(max_epochs,
                      dataloader,
                      proportion_tensor,
                      mapping_dict,
                      color_map,
                      model_param_tuple,
                      device='cuda'):

    # Check if pre-trained weights are available.
    if os.path.exists('saved_files/GMVAE_mus2.pt') and os.path.exists('saved_files/GMVAE_logvars.pt') and os.path.exists('saved_files/GMVAE_pis.pt'):
        print("Pre-trained GMVAE_mus and GMVAE_logvars EXIST. Skipping training.")
        return 0
    else:
        print(f"Pre-trained GMVAE_mus and GMVAE_logvars DO NOT EXIST. Training for {max_epochs} epochs.")

        from models import GaussianMixtureVAE
        from train_GMVAE import train_GMVAE

        input_dim, hidden_dim, latent_dim, K = model_param_tuple
        GMVAE_model = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim, K)
        optimizer = torch.optim.Adam(GMVAE_model.parameters(), lr=1e-3)
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Wrap the model with nn.DataParallel
        GMVAE_model = nn.DataParallel(GMVAE_model)
        try:
            # Load the state dict (assuming it was saved from a model wrapped with nn.DataParallel)
            gmvae_state_dict = torch.load("saved_files/GMVAE_model.pt")
            GMVAE_model.load_state_dict(gmvae_state_dict, strict=True)
            print("Loaded existing GMVAE_model.pt")
        except:
            # Initialize weights.
            for m in GMVAE_model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            print("Initialized GMVAE_model")


        kl_weight = 0.0
        kl_weight_max = 1.0
        losses = []
        
        for epoch in range(0, max_epochs):
            kl_weight_increment = kl_weight_max / (100000)
            
            if kl_weight < kl_weight_max:
                kl_weight += kl_weight_increment
                kl_weight = min(kl_weight, kl_weight_max)
            # Train model.
            total_loss = train_GMVAE(GMVAE_model, epoch, dataloader, optimizer, proportion_tensor, kl_weight, mapping_dict, color_map, max_epochs, device)
            losses.append(total_loss)
 


# Train BulkEncoder. Refer to train_bulkEncoder.py for the implementation and model checkpoint path.
def train_model_BulkEncoder(max_epochs,
                            dataloader,
                            model_param_tuple,
                            device='cuda',
                            train_more=False):
    # Check if pre-trained weights are available.
    if os.path.exists('saved_files/bulkEncoder_model.pt'):
        if train_more:
            print(f"Pre-trained bulkEncoder_model EXIST. Additionally training for {max_epochs} epochs.")
        else:
            print("Pre-trained bulkEncoder_model EXIST. Skipping training.")
            return 0
    else:
        print(f"Pre-trained bulkEncoder_model DOES NOT exist. Training for {max_epochs} epochs.")

    from models import GaussianMixtureVAE, bulkEncoder
    from train_bulkEncoder import train_BulkEncoder

    scMus = torch.load('saved_files/GMVAE_mus.pt').to(device).detach().requires_grad_(False)
    scLogVars = torch.load('saved_files/GMVAE_logvars.pt').to(device).detach().requires_grad_(False)
    scPis = torch.load('saved_files/GMVAE_pis.pt').to(device).detach().requires_grad_(False)

    input_dim, hidden_dim, latent_dim, K = model_param_tuple

    bulkEncoder_model = bulkEncoder(input_dim, hidden_dim, latent_dim, K)
    
    if os.path.exists('saved_files/bulkEncoder_model.pt'):
        encoder_state_dict = torch.load( "saved_files/bulkEncoder_model.pt")
        bulkEncoder_model.load_state_dict(encoder_state_dict, strict=True)
    else:
        # Initialize weights.
        for m in bulkEncoder_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    optimizer = torch.optim.Adam(bulkEncoder_model.parameters(), lr=1e-3)

    GMVAE_model = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim, K)

    GMVAE_model = nn.DataParallel(GMVAE_model)

    # Load the state dict (assuming it was saved from a model wrapped with nn.DataParallel)
    gmvae_state_dict = torch.load("saved_files/GMVAE_model.pt")
    GMVAE_model.load_state_dict(gmvae_state_dict, strict=True)

    bulkEncoder_model = bulkEncoder_model.to(device)

    for epoch in range(0, max_epochs):
        train_BulkEncoder(epoch,
                            bulkEncoder_model,
                            GMVAE_model,
                            max_epochs,
                            optimizer,
                            dataloader,
                            scMus,
                            scLogVars,
                            scPis,
                            device)


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    ############################ 0. Prepare args
    data_dir = "/u/hc2kc/scVAE/paired/data/"
    barcode_path = data_dir+'barcode_to_celltype.csv'
    args = configure(data_dir,barcode_path)

    ############################ 1. Train GMVAE for scMu and scLogVar.
    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    K = args.K

    train_model_GMVAE(
        max_epochs=args.train_GMVAE_epochs,
        dataloader=args.dataloader,
        proportion_tensor=args.cell_type_fractions,
        mapping_dict=args.mapping_dict,
        color_map=args.color_map,
        model_param_tuple=(input_dim, hidden_dim, latent_dim, K),
        device=device
    )
    
    ############################ 2. Train scDecoder for reconstruction using trained scMu and scLogVar.
    train_model_BulkEncoder(
        max_epochs=args.bulk_encoder_epochs,
        dataloader=args.dataloader,
        model_param_tuple=(input_dim, hidden_dim, latent_dim, K),
        device=device,
        train_more=True
    )

    ############################ 3. Generate. Refer to generate.py for the implementation and data save path.
    from models import GaussianMixtureVAE, bulkEncoder
    from generate import generate

    num_cells = args.num_cells
    GMVAE_model = GMVAE_model = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim, K)
    bulkEncoder_model = bulkEncoder(input_dim, hidden_dim, latent_dim, K)
    
    encoder_state_dict = torch.load("saved_files/bulkEncoder_model.pt")
    gmvae_state_dict = torch.load("saved_files/GMVAE_model.pt")

    bulkEncoder_model.load_state_dict(encoder_state_dict, strict=True)

    GMVAE_model = nn.DataParallel(GMVAE_model)

    # Load the state dict (assuming it was saved from a model wrapped with nn.DataParallel)
    gmvae_state_dict = torch.load("saved_files/GMVAE_model.pt")
    GMVAE_model.load_state_dict(gmvae_state_dict, strict=True)

    generate(bulkEncoder_model, GMVAE_model, args.dataloader, num_cells, args.mapping_dict, args.color_map, device=device)


