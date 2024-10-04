import platform
import os

import numpy as np
from tqdm import tqdm

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
do_slice = True
do_train_VAE = True
num_epochs_VAE = 4
do_train_transformer = True

# Transformer settings
num_epochs_transformer = 4

do_generate = True
attention_heads = 4
transformer_layers = 8

# Common training
train_percentage = 60
hidden_layer_constant = 256
latent_dim_vae = 240

# Generation settings
generated_length = 200
lookback = 500

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
    WavHandler.prepare_slices(0.1, 0, 24000)
    VAEEngine.retrain_VAE(ninp=ninp, hidden_layer_constant=hidden_layer_constant, latent_dim=latent_dim_vae,
                          device=device,
                          do_train_VAE=do_train_VAE, num_epochs_VAE=num_epochs_VAE, train_percentage=train_percentage,
                          model_folder=model_folder)
elif do_train_VAE:
    VAEEngine.retrain_VAE(ninp=ninp, hidden_layer_constant=hidden_layer_constant, latent_dim=latent_dim_vae,
                          device=device,
                          do_train_VAE=do_train_VAE, num_epochs_VAE=num_epochs_VAE, train_percentage=train_percentage,
                          model_folder=model_folder)

''' TRANSFORMER TRAINING '''

# Train transformer model
if do_train_transformer:
    # Load encoded data
    encoded_data_files = sorted(
        [f for f in os.listdir(model_folder) if f.startswith('encoded_data_') and f.endswith('.npy')])

    # Initialize transformer model
    transformer_model = NeuralNetwork.transformer_model(ntoken=ntoken, ninp=latent_dim_vae, nhead=attention_heads,
                                                        nhid=hidden_layer_constant, nlayers=transformer_layers)
    optimizer = optim.Adam(transformer_model.parameters())

    # Move your model to the chosen device
    model = transformer_model.to(device)

    history = {'loss': []}

    # Train the model
    for epoch in range(num_epochs_transformer):
        print(f"Starting epoch {epoch + 1}")

        # Load encoded data in chunks
        for i in tqdm(range(len(encoded_data_files) - 1),
                      bar_format='\033[37m{l_bar}{bar:40}{r_bar}\033[0m'):  # Subtract 1 to avoid going out of bounds
            # Load encoded data
            encoded_data_chunk = np.load(os.path.join(model_folder, encoded_data_files[i]))
            # Load next encoded data for label
            encoded_data_label = np.load(os.path.join(model_folder, encoded_data_files[i + 1]))

            # Train the model on each clip
            for index, encoded_data in enumerate(encoded_data_chunk):
                NeuralNetwork.single_epoch_train(encoded_data, encoded_data_label, transformer_model, optimizer,
                                                 history, device,
                                                 index)

        # Save the model and plot after each epoch
        FolderHandlers.save_model_with_version(model=transformer_model, directory=model_folder,
                                               base_filename="fitted_model")
        NeuralNetwork.save_training_plot(history=history, model_folder=model_folder, do_show=False)

    # Save the model and plot after the last epoch
    FolderHandlers.save_model_with_version(model=transformer_model, directory=model_folder,
                                           base_filename="fitted_model")

    NeuralNetwork.save_training_plot(history=history, model_folder=model_folder, do_show=False)

''' GENERATE OUTPUT '''

# Initialize the model
transformer_model = NeuralNetwork.transformer_model(ntoken=ntoken, ninp=latent_dim_vae, nhead=attention_heads,
                                                    nhid=hidden_layer_constant, nlayers=transformer_layers)

# Load the state dictionary into the model instance
trained_model = FolderHandlers.get_latest_model(directory=model_folder, base_filename="fitted_model")
transformer_model.load_state_dict(trained_model)

# Move the model to the device
transformer_model = transformer_model.to(device)

# Now you can use transformer_model as your model
if do_generate:
    print("Generating output.")

    # Load VAE
    vae = VAE(input_dim=ninp, hidden_dim=hidden_layer_constant, latent_dim=latent_dim_vae).to(device)
    vae.load_state_dict(torch.load(os.path.join(model_folder, 'vae.pth'), weights_only=True))

    # Prepare test data
    test_data = WavHandler.load_samples(num_samples=lookback, directory=clip_output_folder)
    test_data = np.array(test_data)

    # Encode test data using VAE
    encoded_test_data, _ = vae.encode(torch.tensor(test_data, dtype=torch.float32).unsqueeze(0).to(device))

    # Generate output
    generated_sequence = NeuralNetwork.generate_sequence(model=transformer_model, start_sequence=encoded_test_data,
                                                         length=generated_length,
                                                         lookback=lookback, vae_model=vae)

    WavHandler.save_generated_output(sequence=generated_sequence,
                                     directory=generated_output_folder,
                                     file_name="generated_output")

    print("Generated output saved to " + generated_output_folder)
