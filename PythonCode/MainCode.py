import platform
import os

import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"  # Set to "1" to debug CUDA errors

import VAEEngine
import FolderHandlers
import WavHandler
import NeuralNetwork
from VAEEngine import VAE
import torch
from torch import optim

''' PARAMETERS '''
# Input clip settings
do_slice = False
do_train_VAE = True
num_epochs_VAE = 1
do_train_transformer = True

# Transformer settings
num_epochs_transformer = 1
do_generate = False

# Common training
train_percentage = 20
hidden_layer_constant = 1
latent_dim_vae = 40

# Output paths
clip_output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"
generated_output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Generated"
model_folder = "D:\\University\\Thesis\\FittedModel"

''' ENVIRONMENT SETUPS '''

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

''' CLIP PREPARATION AND TRAINING '''

# Slice audio clips, train VAE and save encoded data
if do_slice:
    WavHandler.prepare_slices(0.01, 0, 24000)
    VAEEngine.retrain_VAE(ninp, hidden_layer_constant, latent_dim_vae, device, do_train_VAE, num_epochs_VAE,
                          train_percentage, model_folder)
elif do_train_VAE:
    VAEEngine.retrain_VAE(ninp, hidden_layer_constant, ntoken, device, do_train_VAE, num_epochs_VAE, train_percentage,
                          model_folder)

''' TRANSFORMER TRAINING '''

# Train transformer model
if do_train_transformer:
    # Load encoded data
    encoded_data_files = sorted(
        [f for f in os.listdir(model_folder) if f.startswith('encoded_data_') and f.endswith('.npy')])

    # Initialize transformer model
    transformer_model = NeuralNetwork.transformer_model(ntoken, latent_dim_vae, 2, hidden_layer_constant, 1)
    optimizer = optim.Adam(transformer_model.parameters())

    # Move your model to the chosen device
    model = transformer_model.to(device)

    history = {'loss': []}

    # Train the model
    for epoch in range(num_epochs_transformer):
        print(f"Starting epoch {epoch + 1}")

        # Load encoded data in chunks
        for encoded_data_file in encoded_data_files:
            # Load encoded data
            encoded_data_chunks = np.load(os.path.join(model_folder, encoded_data_file))

            # Train the model on each clip
            for encoded_data_chunk in encoded_data_chunks:
                NeuralNetwork.single_epoch_train(encoded_data_chunk, transformer_model, optimizer, history, device,
                                                 epoch)

        # Save the model and plot after each epoch
        FolderHandlers.save_model_with_version(model=transformer_model, directory=model_folder,
                                               base_filename="fitted_model")
        NeuralNetwork.save_training_plot(history=history, model_folder=model_folder, do_show=False)

    # Save the model and plot after the last epoch
    FolderHandlers.save_model_with_version(model=transformer_model, directory=model_folder,
                                           base_filename="fitted_model")

    NeuralNetwork.save_training_plot(history=history, model_folder=model_folder, do_show=False)

''' GENERATE OUTPUT '''

trained_model = FolderHandlers.get_latest_model(directory=model_folder, base_filename="fitted_model")

if do_generate:
    print("Generating output.")

    # Load VAE
    vae = VAE(input_dim=ninp, hidden_dim=hidden_layer_constant, latent_dim=ntoken).to(device)
    vae.load_state_dict(torch.load('vae.pth'))

    # Prepare test data
    test_data = WavHandler.load_samples_generator(percentage=10)

    # Encode test data using VAE
    encoded_test_data = vae.encode(torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).to(device))

    # Generate output
    generated_sequence = NeuralNetwork.generate_sequence(model=trained_model, start_sequence=encoded_test_data,
                                                         length=400,
                                                         lookback=500, vae_model=vae)

    # Decode generated sequence using VAE
    decoded_generated_sequence = vae.decode(generated_sequence)

    WavHandler.create_dequantised_output(quantised_sequence=decoded_generated_sequence,
                                         directory=generated_output_folder,
                                         file_name="generated_output", num_bins=65535)
    
    print("Generated output saved to " + generated_output_folder)
