import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMixtureVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_components):
        super(GaussianMixtureVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_components = n_components
        
        # Encoder
        self.fc0 = nn.Linear(input_dim, hidden_dim//2)
        self.fc1 = nn.Linear(hidden_dim//2, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, n_components * latent_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim, n_components * latent_dim)  # logvar layer
        # Mixture weights
        self.fc_pi = nn.Linear(hidden_dim, n_components)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.05)

        # Parameters for ZINB
        self.prob_extra_zero = nn.Parameter(torch.rand(1))  # Initialize to a random value
        self.over_disp = nn.Parameter(torch.rand(1))   # Initialize to a random value

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim//2)
        # self.h0_attention = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.fc4 = nn.Linear(hidden_dim//2, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)


    def encode(self, x):
        h0 = F.relu(self.fc0(x))
        h0 = self.dropout(h0)
        # h0_sum = h0.mean(dim=0)
        h1 = F.relu(self.fc1(h0))
        h1 = self.dropout(h1)

        mu = self.fc21(h1)  # Shape: (latent_dim)
        logvar = self.fc22(h1)  # Shape: (latent_dim)

        logits = self.fc_pi(h1)  # These are logits for the softmax
        logits = logits - logits.max(dim=-1, keepdim=True).values  # Normalize for numerical stability
        pis = F.softmax(logits, dim=-1)

        mu = mu.reshape(-1, self.n_components, self.latent_dim)
        logvar = logvar.reshape(-1, self.n_components, self.latent_dim)

        return mu, logvar, pis#, h0


    def reparameterize_with_labels(self, mus, logvars, labels):
        zs = []
        for label in labels:
            k = label.item()
            # Extract the mu and logvar for the specified cluster k
            # Shapes of selected_mus and selected_logvars: (batch_size, latent_dim)
            selected_mus = mus[:, k, :]
            selected_logvars = logvars[:, k, :]

            # Representative value for the cluster (e.g., mean across the batch)
            mean_mus = selected_mus.mean(dim=0)  # Shape: (latent_dim,)
            mean_logvars = selected_logvars.mean(dim=0)  # Shape: (latent_dim,)

            # Reparameterization trick
            std = torch.exp(0.5 * mean_logvars)
            eps = torch.randn_like(std)
            z = mean_mus + eps * std  # Shape: (latent_dim,)
            zs.append(z)
        zs = torch.stack(zs)

        return zs
    

    def decode(self, zs):
        h3 = F.relu(self.fc3(zs))
        h3 = self.dropout(h3)

        # h1 = self.h0_attention(h0)
        # h1 = h1.squeeze()
        # import pdb; pdb.set_trace()

        # hh = torch.cat([h1, h3], dim=-1)

        h4 = F.relu(self.fc4(h3))
        h4 = self.dropout(h4)
        
        reconstructed = F.relu(self.fc5(h4))
        return reconstructed


    def forward(self, x, labels):
        mus, logvars, pis = self.encode(x.view(-1, self.input_dim))
        zs = self.reparameterize_with_labels(mus, logvars, labels)
        reconstructed = self.decode(zs)
        return reconstructed, mus, logvars, pis, zs


    def reparameterize_with_proportion(self, mus, logvars, pi):
        """Given a bulk RNA-seq, reparameterize mus and logvars into one z."""
        # Randomly sample a cluster from the multinomial distribution.
        # Shape of sampled_indices: (batch_size,)
        k = torch.multinomial(pi, 1).squeeze()
        # Reparameterize the mus and logvars.
        # import pdb; pdb.set_trace()
        selected_mus = mus[k, :]
        selected_logvars = logvars[k, :]

        # # Representative value for the cluster (e.g., mean across the batch)
        # mean_mus = selected_mus.mean(dim=0)  # Shape: (latent_dim,)
        # mean_logvars = selected_logvars.mean(dim=0)  # Shape: (latent_dim,)

        # Reparameterization trick
        std = torch.exp(0.5 * selected_logvars)
        eps = torch.randn_like(std)
        z = selected_mus + eps * std  # Shape: (latent_dim,)
        return z, k


    def decode_bulk(self, mus, logvars, pis):
        z, k = self.reparameterize_with_proportion(mus, logvars, pis)
        reconstructed = self.decode(z)
        return reconstructed, k


class bulkEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_components=10):
        super(bulkEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_components = n_components

        self.fc0 = nn.Linear(input_dim, hidden_dim//2)
        # self.h0_attention = nn.Linear(hidden_dim//2, hidden_dim//2)
        # self.h0_attention2 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.fc1 = nn.Linear(hidden_dim//2, hidden_dim)

        self.dropout = nn.Dropout(p=0.05)

        # The output layers for mu, logvar, and pi
        self.fc_mu = nn.Linear(hidden_dim, n_components * latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, n_components * latent_dim)
        self.fc_pi = nn.Linear(hidden_dim, n_components)
        
    def forward(self, x):
        h0 = F.relu(self.fc0(x))
        h0 = self.dropout(h0)
        
        # h9 = F.relu(self.h0_attention(h0))
        # h9 = self.h0_attention2(h9)


        h1 = F.relu(self.fc1(h0))
        h = self.dropout(h1)
        
        mu = self.fc_mu(h).view(-1, self.n_components, self.latent_dim)
        logvar = self.fc_logvar(h).view(-1, self.n_components, self.latent_dim)
        pi_logits = self.fc_pi(h)
        pi = F.softmax(pi_logits, dim=-1)
        
        return mu, logvar, pi