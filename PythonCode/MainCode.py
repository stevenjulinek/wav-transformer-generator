import platform
#set 1 when debugging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import FolderHandlers
import WavHandler
import NeuralNetwork
from VAEEngine import VAE, encode_data_in_chunks
import torch
from torch import optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

# Parameters
do_slice = False
do_train_VAE = True
num_epochs_VAE = 1
do_train_transformer = True
num_epochs_transformer = 1
do_generate = False
train_percentage = 20
vae_chunk_size = 1000  # Adjust this value based on your memory capacity
clip_output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"
generated_output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Generated"
model_folder = "D:\\University\\Thesis\\FittedModel"

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use the first GPU for processing
    print("Running on the following GPU: " + str(torch.cuda.get_device_name()))
else:
    device = torch.device("cpu")  # Use the CPU for processing
    print("Running on the following CPU: " + str(platform.processor()))

# Prepare network variables
ntoken = int(FolderHandlers.count_wavs_in_folder(clip_output_folder) * train_percentage / 100)
ninp = WavHandler.length_of_a_clip(clip_output_folder)
hidden_layer_constant = 1

# Slice audio clips, train VAE and save encoded data
if do_slice:
    WavHandler.prepare_slices(0.01, 0, 24000)

    # Initialize VAE
    vae = VAE(input_dim=ninp, hidden_dim=hidden_layer_constant, latent_dim=ntoken).to(device)
    vae_optimizer = optim.Adam(vae.parameters())

    if do_train_VAE:
        # Train VAE on audio clips
        for epoch in range(num_epochs_VAE):
            for data in WavHandler.load_unquantised_samples_generator(percentage=train_percentage):
                data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
                vae_optimizer.zero_grad()
                reconstructed_data, mu, logvar = vae(data)
                loss = VAE.loss_function(reconstructed_data, data, mu, logvar)
                loss.backward()
                vae_optimizer.step()

    # Save VAE
    torch.save(vae.state_dict(), 'vae.pth')

    # Encode all data and save encoded values
    all_data = WavHandler.load_unquantised_samples_generator(percentage=100)
    encoded_data_generator = encode_data_in_chunks(vae, all_data, vae_chunk_size, device)

    for i, encoded_data_chunk in enumerate(encoded_data_generator):
        torch.save(encoded_data_chunk, f'encoded_data_chunk_{i}.pth')


elif do_train_VAE:
    # Initialize VAE
    vae = VAE(input_dim=ninp, hidden_dim=hidden_layer_constant, latent_dim=ntoken).to(device)
    vae_optimizer = optim.Adam(vae.parameters())

    # Train VAE on audio clips
    for epoch in range(num_epochs_VAE):
        for data in WavHandler.load_unquantised_samples_generator(percentage=train_percentage):
            data = torch.tensor(data[0]).to(device)
            vae_optimizer.zero_grad()
            reconstructed_data, mu, logvar = vae(data)
            loss = VAE.loss_function(reconstructed_data, data, mu, logvar)
            loss.backward()
            vae_optimizer.step()

    # Save VAE
    torch.save(vae.state_dict(), 'vae.pth')

    # Encode all data and save encoded values
    all_data = WavHandler.load_unquantised_samples_generator(percentage=100)
    encoded_data_generator = encode_data_in_chunks(vae, all_data, vae_chunk_size, device)

    for i, encoded_data_chunk in enumerate(encoded_data_generator):
        torch.save(encoded_data_chunk, f'encoded_data_chunk_{i}.pth')

if do_train_transformer:
    # Load encoded data
    encoded_data = torch.load('encoded_data.pth')

    transformer_model = NeuralNetwork.transformer_model(ntoken, ninp, 2, hidden_layer_constant, 1)
    optimizer = optim.Adam(transformer_model.parameters())

    # Move your model to the chosen device
    model = transformer_model.to(device)

    history = {'loss': []}

    for epoch in range(num_epochs_transformer):
        print(f"Starting epoch {epoch + 1}")
        # Prepare training data
        quantised_samples_generator = WavHandler.load_quantised_samples_generator(percentage=train_percentage)

        for encoded_train_data, encoded_train_labels in encoded_data:
            # Reshape data if necessary, for example if your model expects a certain shape
            data = torch.tensor(encoded_train_data, dtype=torch.float32).unsqueeze(0).to(device)
            labels = torch.tensor(encoded_train_labels, dtype=torch.float32).unsqueeze(0).to(device)

            # Move your data to the chosen device
            data = data.to(device)
            labels = labels.to(device)

            # Train the model on the data
            optimizer.zero_grad()
            outputs = transformer_model(data)
            outputs = outputs[:, -1, :]
            loss = F.mse_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # Append the loss of the current epoch to the history
            history['loss'].append(loss.item())

        # Save the model
        FolderHandlers.save_model_with_version(model=transformer_model, directory=model_folder,
                                               base_filename="fitted_model")
        NeuralNetwork.save_training_plot(history=history, model_folder=model_folder, do_show=False)

    # Save the model
    FolderHandlers.save_model_with_version(model=transformer_model, directory=model_folder,
                                           base_filename="fitted_model")

    NeuralNetwork.save_training_plot(history=history, model_folder=model_folder, do_show=False)

trained_model = FolderHandlers.get_latest_model(directory=model_folder, base_filename="fitted_model")

if(do_generate):
    print("Generating output.")

    # Load VAE
    vae = VAE(input_dim=ninp, hidden_dim=hidden_layer_constant, latent_dim=ntoken).to(device)
    vae.load_state_dict(torch.load('vae.pth'))

    # Prepare test data
    test_data = WavHandler.load_unquantised_samples_generator(percentage=10)

    # Encode test data using VAE
    encoded_test_data = vae.encode(torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).to(device))

    # Generate output
    generated_sequence = NeuralNetwork.generate_sequence(model=trained_model, start_sequence=encoded_test_data,
                                                         length=400,
                                                         lookback=500)

    # Decode generated sequence using VAE
    decoded_generated_sequence = vae.decode(generated_sequence)

    WavHandler.create_dequantised_output(quantised_sequence=decoded_generated_sequence,
                                         directory=generated_output_folder,
                                         file_name="generated_output", num_bins=65535)
    print("Generated output saved.")
