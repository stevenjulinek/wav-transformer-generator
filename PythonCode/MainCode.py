import FolderHandlers
import WavHandler
import NeuralNetwork
import tensorflow as tf
import numpy as np
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"

# WavHandler.prepare_slices(0.01, 0.001, 24000)

# Prepare training data
train_data = WavHandler.return_slices(percentage=100)
# Quantise the waveform values into 256 discrete values
quantised_train_data = np.digitize(train_data, np.linspace(-1.0, 1.0, 256)) - 1
# Shift the data by one timestep for labels
# Remove the last sample in train_data and the first sample in train_labels
quantised_train_labels = quantised_train_data[1:]
quantised_train_data = quantised_train_data[:-1]


ntoken = len(quantised_train_data)
ninp = WavHandler.length_of_a_clip(output_folder)
hidden_layer_constant = 3
embed_dim = 1024
num_heads = 8
transformer_model = NeuralNetwork.transformer_model(ntoken, ninp, 16, hidden_layer_constant*ninp, 16)
transformer_model.compile(optimizer='adam', loss='mse')

# Convert your data to TensorFlow tensors
train_data = tf.convert_to_tensor(quantised_train_data)
train_labels = tf.convert_to_tensor(quantised_train_labels)

# Fit the model
history = transformer_model.fit(quantised_train_data, quantised_train_labels, epochs=10, batch_size=4)

# Save the model
folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\PythonCode\\FittedModel\\"

transformer_model.save('C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\PythonCode\\FittedModel\\model')
