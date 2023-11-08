import torch
import torch.nn as nn
import torch.nn.functional as F


class scDecoder(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim):
        super(scDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, h_dim//2)
        self.fc2 = nn.Linear(h_dim//2, h_dim)
        self.fc3 = nn.Linear(h_dim, input_dim)


    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps*std
    

    def decode(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return h
    
    def forward(self, z):
        x_reconst = self.decode(z)
        return x_reconst
    


class bulkEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_clusters):
        super(bulkEncoder, self).__init__()
        
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # The output layers for mu, var, and pi
        self.fc_mu = nn.Linear(hidden_dim, n_clusters * latent_dim)
        self.fc_var = nn.Linear(hidden_dim, n_clusters * latent_dim)
        self.fc_pi = nn.Linear(hidden_dim, n_clusters)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        
        mu = self.fc_mu(h).view(-1, self.n_clusters, self.latent_dim)
        var = self.fc_var(h).view(-1, self.n_clusters, self.latent_dim)
        pi_logits = self.fc_pi(h)
        pi = F.softmax(pi_logits, dim=-1)
        
        return mu, var, pi