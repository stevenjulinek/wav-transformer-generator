import FolderHandlers
import WavHandler
import NeuralNetwork
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

do_train = True
do_slice = False
clip_output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"
generated_output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Generated"
model_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\PythonCode\\FittedModel\\"

if (do_slice):
    print("Starting slicing of input samples. This may take a while.")
    WavHandler.prepare_slices(0.01, 0, 24000)
    print("Finished slicing samples.")

# Prepare training data
train_data = WavHandler.return_slices(percentage=100)
# Quantise the waveform values into 256 discrete values
quantised_train_data = np.digitize(train_data, np.linspace(-1.0, 1.0, 65535)) - 1
# Shift the data by one timestep for labels
# Remove the last sample in train_data and the first sample in train_labels
quantised_train_labels = quantised_train_data[1:]
quantised_train_data = quantised_train_data[:-1]

ntoken = len(quantised_train_data)
ninp = WavHandler.length_of_a_clip(clip_output_folder)
hidden_layer_constant = 3
transformer_model = NeuralNetwork.transformer_model(ntoken, ninp, 14, hidden_layer_constant * ninp, 14)
transformer_model.compile(optimizer='adam', loss='mse')

# Convert your data to TensorFlow tensors
train_data = tf.convert_to_tensor(quantised_train_data)
train_labels = tf.convert_to_tensor(quantised_train_labels)

if do_train:
    # Fit the model
    history = transformer_model.fit(quantised_train_data, quantised_train_labels, epochs=10, batch_size=4)

    # Save the model
    FolderHandlers.save_model_with_version(model=transformer_model, directory=model_folder,
                                           base_filename="fitted_model")

    # Plot the training loss
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    # plt.show()

    # Save the plot
    plt.savefig(model_folder + 'loss_plot.png')

trained_model = FolderHandlers.get_latest_model(directory=model_folder, base_filename="fitted_model")

generated_sequence = NeuralNetwork.generate_sequence(model=trained_model, start_sequence=train_data, length=1000,
                                                     lookback=500)

WavHandler.create_dequantised_output(quantised_sequence=generated_sequence, directory=generated_output_folder,
                                     file_name="generated_output", num_bins=65535)

print("Finished generating output.")
