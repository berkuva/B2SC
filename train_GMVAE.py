import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
import torch.nn.functional as F



def zinb_loss(y_true, y_pred, pi, r, eps=1e-10):
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    # Negative binomial part
    nb_case = -torch.lgamma(r + eps) + torch.lgamma(y_true + r + eps) - torch.lgamma(y_true + 1.0) \
              + r * torch.log(pi + eps) + y_true * torch.log(1.0 - (pi + eps))
    
    # Zero-inflated part
    zero_nb = torch.pow(pi, r)
    zero_case = torch.where(y_true < eps, -torch.log(zero_nb + (1.0 - zero_nb) * torch.exp(-r * torch.log(1.0 - pi + eps))), torch.tensor(0.0, device=y_true.device))
    
    return -torch.mean(zero_case + nb_case)


def train_GMVAE(model, epoch, dataloader, optimizer, proportion_tensor, kl_weight, mapping_dict, color_map, max_epochs, device='cuda'):
    model.train()
    total_loss = 0
    model = model.to(device)

    for idx, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        reconstructed, mus, logvars, pis, zs = model(data, labels)
        

        proportion_tensor_reshaped = proportion_tensor.to(pis.device)
        # import pdb; pdb.set_trace()
        fraction_loss =  F.mse_loss(pis.mean(0), proportion_tensor_reshaped)
        loss_recon = F.mse_loss(reconstructed, data)

        zinb_loss_val = zinb_loss(data, reconstructed, model.module.prob_extra_zero, model.module.over_disp)

        loss_kl = 0.5 * torch.sum(-1 - logvars + mus.pow(2) + logvars.exp())/1e10
        loss_kl = loss_kl * kl_weight
        loss_kl = 1 if loss_kl > 1 else loss_kl
        
        loss = loss_recon + loss_kl + fraction_loss + zinb_loss_val
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch+1} KL Loss: {loss_kl:.4f} Recon Loss: {loss_recon:.4f} Total Loss: {total_loss:.4f} Fraction Loss: {fraction_loss:.4f} ZINB Loss: {zinb_loss_val:.4f}')


    if (epoch+1) % 100 == 0:
        # Save reconstructed.
        torch.save(reconstructed, 'saved_files/GMVAE_reconstructed.pt')

        mus = mus.mean(0)
        logvars = logvars.mean(0)
        pis = pis.mean(0)

        # Save the mean, logvar, and pi.
        torch.save(mus, 'saved_files/GMVAE_mus.pt')
        torch.save(logvars, 'saved_files/GMVAE_logvars.pt')
        torch.save(pis, 'saved_files/GMVAE_pis.pt')
        print("GMVAE mu & var & pi saved.")

        model.eval()

        k = labels.cpu().detach().numpy()
        
        # Generate QQ plot for reconstructed data.
        reconstructed = reconstructed.cpu().detach().numpy()

        z = zs.cpu().detach().numpy()

        # Convert all_labels to colors using the color_map
        label_map = {v: k for k, v in mapping_dict.items()}
        mean_colors = [color_map[label_map[str(label.item())]] for label in k]
        z_colors = [color_map[label_map[str(label.item())]] for label in k]

        # UMAP transformation of recon
        reducer = umap.UMAP()
        embedding_z = reducer.fit_transform(z)
        embedding_recon = reducer.fit_transform(reconstructed)
 

        plt.figure(figsize=(12, 10))
        plt.scatter(embedding_z[:, 0], embedding_z[:, 1], c=z_colors, s=5)
        # Remove ticks
        plt.xticks([])
        plt.yticks([])
        # Name the axes.
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.title('UMAP of reparameterized z')
        plt.savefig('saved_files/umap_latent.png')
        plt.close()

        plt.figure(figsize=(12, 10))
        plt.scatter(embedding_recon[:, 0], embedding_recon[:, 1], c=mean_colors, s=5)
        # Remove ticks
        plt.xticks([])
        plt.yticks([])
        # Name the axes.
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.title('UMAP of Reconstructed Data')
        plt.savefig('saved_files/umap_recon.png')
        plt.close()


        torch.save(model.state_dict(), 'saved_files/GMVAE_model.pt')
        print("GMVAE Model saved.")
        
    
    return total_loss