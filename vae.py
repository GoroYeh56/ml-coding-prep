import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_var = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        # Reparameterization Trick to allow gradients to flow backward
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        mu, log_var = self.fc_mu(x), self.fc_var(x)
        # Sample
        z = self.reparameterize(mu, log_var)
        # Decode
        recon_x = self.decoder(z)
        
        return recon_x, mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    # 1. Reconstruction Loss (BCE)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # 2. KL Divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total ELBO Loss
    return recon_loss + kl_div

# Standard Backward Pass Example:
recon_x, mu, log_var = model(x)
loss = vae_loss(recon_x, x, mu, log_var)
loss.backward()
