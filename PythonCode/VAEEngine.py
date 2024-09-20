# VAEEngine.py
import torch
from torch import nn
from FolderHandlers import data_chunk_generator

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.input_dim = input_dim  # Add this line
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvariance layer

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):

        # Ensure the values are in the [0, 1] range
        assert recon_x.min() >= 0 and recon_x.max() <= 1, f"Values should be in [0, 1], but got min={recon_x.min()}, max={recon_x.max()}"
        assert x.min() >= 0 and x.max() <= 1, f"Values of x should be in [0, 1], but got min={x.min()}, max={x.max()}"

        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 240), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

def encode_data_in_chunks(vae, all_data, chunk_size, device):
    data_generator = data_chunk_generator(all_data, chunk_size)
    for data_chunk in data_generator:
        data_chunk = torch.tensor(data_chunk, dtype=torch.float32).unsqueeze(0).to(device)
        encoded_data_chunk, _ = vae.encode(data_chunk)
        yield encoded_data_chunk