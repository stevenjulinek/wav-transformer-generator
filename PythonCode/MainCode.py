import librosa

import FolderHandlers
import WavHandler
import NeuralNetwork
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

do_slice = True
do_train = True
do_generate = True
train_percentage = 20
clip_output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"
generated_output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Generated"
model_folder = "D:\\University\\Thesis\\FittedModel"

if (do_slice):
    WavHandler.prepare_slices(0.1, 0, 24000)

# Prepare model
ntoken = int(FolderHandlers.count_wavs_in_folder(clip_output_folder) * train_percentage / 100)
ninp = WavHandler.length_of_a_clip(clip_output_folder)
hidden_layer_constant = 3
transformer_model = NeuralNetwork.transformer_model(ntoken, ninp, 18, hidden_layer_constant * ninp, 18)
transformer_model.compile(optimizer='adam', loss='mse')

if do_train:
    num_epochs = 10
    history = {'loss': []}
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}")
        # Prepare training data
        quantised_samples_generator = WavHandler.load_quantised_samples_generator(percentage=train_percentage)
        for quantised_train_data, quantised_train_labels in quantised_samples_generator:
            # Reshape data if necessary, for example if your model expects a certain shape
            train_data = np.reshape(quantised_train_data, (1, ninp))
            train_labels = np.reshape(quantised_train_labels, (1, ninp))

            # Train the model on the data
            hist = transformer_model.fit(train_data, train_labels, epochs=1, verbose=0)

            # Append the loss of the current epoch to the history
            history['loss'].append(hist.history['loss'][0])

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

    # Prepare test data
    test_data = WavHandler.load_quantised_samples(percentage=10)
    # Generate output
    generated_sequence = NeuralNetwork.generate_sequence(model=trained_model, start_sequence=test_data, length=400,
                                                        lookback=500)

    WavHandler.create_dequantised_output(quantised_sequence=generated_sequence, directory=generated_output_folder,
                                        file_name="generated_output", num_bins=65535)
    print("Generated output saved.")
