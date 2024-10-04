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

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc21 = nn.Linear(hidden_dim // 4, latent_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim // 4, latent_dim)  # logvariance layer
        self.dropout = nn.Dropout(p=0.3)

        # Decoder
        self.fc4 = nn.Linear(latent_dim, hidden_dim // 4)
        self.fc5 = nn.Linear(hidden_dim // 4, hidden_dim // 2)
        self.fc6 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = self.dropout(torch.relu(self.fc3(h2)))
        return self.fc21(h3).view(-1, self.latent_dim), self.fc22(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = torch.relu(self.fc4(z))
        h5 = torch.relu(self.fc5(h4))
        h6 = self.dropout(torch.relu(self.fc6(h5)))
        return torch.sigmoid(self.fc7(h6))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, input_dim):
        # Ensure the values are in the [0, 1] range
        assert recon_x.min() >= 0 and recon_x.max() <= 1, f"Values should be in [0, 1], but got min={recon_x.min()}, max={recon_x.max()}"
        assert x.min() >= 0 and x.max() <= 1, f"Values of x should be in [0, 1], but got min={x.min()}, max={x.max()}"

        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

def encode_data_in_chunks(vae, all_data, chunk_size, device):
    data_generator = data_chunk_generator(all_data, chunk_size)
    for data_chunk in data_generator:
        data_chunk = np.array(data_chunk)  # Convert list of numpy arrays to a single numpy array
        data_chunk = torch.tensor(data_chunk, dtype=torch.float32).unsqueeze(0).to(device)
        encoded_data_chunk, _ = vae.encode(data_chunk)
        yield encoded_data_chunk

def retrain_VAE(ninp, hidden_layer_constant, latent_dim, device, do_train_VAE, num_epochs_VAE, train_percentage, model_folder, batch_size=24):
    # Initialize VAE
    vae = VAE(input_dim=ninp, hidden_dim=hidden_layer_constant, latent_dim=latent_dim).to(device)
    vae_optimizer = optim.Adam(vae.parameters())

    print("Training VAE.")
    if do_train_VAE:
        # Train VAE on audio clips
        for epoch in range(num_epochs_VAE):
            for batch in WavHandler.load_samples_generator_batches(percentage=train_percentage, batch_size=batch_size):
                batch = np.array(batch)
                batch = torch.tensor(batch, dtype=torch.float32).unsqueeze(0).to(device)
                vae_optimizer.zero_grad()
                reconstructed_batch, mu, logvar = vae(batch)
                loss = VAE.loss_function(reconstructed_batch, batch, mu, logvar, input_dim=ninp)
                loss.backward()
                vae_optimizer.step()

    # Save VAE
    torch.save(vae.state_dict(), os.path.join(model_folder, 'vae.pth'))  # Save to the model_folder directory

    # Encode all data and save encoded values
    print("Encoding music clip data.")
    all_data = WavHandler.load_samples_generator_batches(percentage=100, batch_size=batch_size)
    save_after_batches = round(1000 / batch_size)
    batch_counter = 0
    encoded_clips = []

    # Ensure the directory exists before saving the file
    os.makedirs(model_folder, exist_ok=True)

    for i, batch in enumerate(all_data):
        batch = np.array(batch)
        batch = torch.tensor(batch, dtype=torch.float32).unsqueeze(0).to(device)
        encoded_batch, _ = vae.encode(batch)
        encoded_clips.extend(encoded_batch.detach().cpu().numpy())

        batch_counter += 1

        # Save every 'save_after_batches' batches in a single file
        if batch_counter >= save_after_batches:
            np.save(os.path.join(model_folder, f'encoded_data_{i}.npy'),
                    np.array(encoded_clips))  # Save as numpy array
            encoded_clips = []
            batch_counter = 0  # Reset the counter

    # Save remaining clips
    if encoded_clips:
        # Pad the sequences in encoded_clips
        encoded_clips_padded = WavHandler.pad_sequences(encoded_clips)

        # Now you can save it as a numpy array
        np.save(os.path.join(model_folder, f'encoded_data_{i}.npy'),
                np.array(encoded_clips_padded))

    print("Finished encoding music clip data.")
