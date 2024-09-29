# VAEEngine.py
import os

import torch
from torch import nn, optim

import WavHandler
from FolderHandlers import data_chunk_generator
import numpy as np

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
        data_chunk = np.array(data_chunk)  # Convert list of numpy arrays to a single numpy array
        data_chunk = torch.tensor(data_chunk, dtype=torch.float32).unsqueeze(0).to(device)
        encoded_data_chunk, _ = vae.encode(data_chunk)
        yield encoded_data_chunk

def retrain_VAE(ninp, hidden_layer_constant, latent_dim, device, do_train_VAE, num_epochs_VAE, train_percentage, model_folder):
    # Initialize VAE
    vae = VAE(input_dim=ninp, hidden_dim=hidden_layer_constant, latent_dim=latent_dim).to(device)
    vae_optimizer = optim.Adam(vae.parameters())

    print("Training VAE.")
    if do_train_VAE:
        # Train VAE on audio clips
        for epoch in range(num_epochs_VAE):
            for data in WavHandler.load_samples_generator(percentage=train_percentage):
                data = np.array(data)
                data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
                vae_optimizer.zero_grad()
                reconstructed_data, mu, logvar = vae(data)
                loss = VAE.loss_function(reconstructed_data, data, mu, logvar)
                loss.backward()
                vae_optimizer.step()

    # Save VAE
    torch.save(vae.state_dict(), 'vae.pth')

    # Encode all data and save encoded values
    print("Encoding music clip data.")
    all_data = WavHandler.load_samples_generator(percentage=100)
    encoded_clips = []
    for i, data_clip in enumerate(all_data):
        data_clip = np.array(data_clip)
        data_clip = torch.tensor(data_clip, dtype=torch.float32).unsqueeze(0).to(device)
        encoded_data_clip, _ = vae.encode(data_clip)
        encoded_clips.append(encoded_data_clip.detach().cpu().numpy())

        # Save every 1000 clips in a single file
        if i % 1000 == 0 and i > 0:
            np.save(os.path.join(model_folder, f'encoded_data_{i // 1000}.npy'),
                    np.array(encoded_clips))  # Save as numpy array
            encoded_clips = []

        # Save remaining clips
    if encoded_clips:
        np.save(os.path.join(model_folder, f'encoded_data_{i // 1000 + 1}.npy'),
                np.array(encoded_clips))  # Save as numpy array

    print("Finished encoding music clip data.")
