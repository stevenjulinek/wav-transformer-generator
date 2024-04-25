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
# Shift the data by one timestep for labels
# Remove the last sample in train_data and the first sample in train_labels
train_labels = train_data[1:]
train_data = train_data[:-1]


ntoken = len(train_data)
ninp = WavHandler.length_of_a_clip(output_folder)
hidden_layer_constant = 3
embed_dim = 1024
num_heads = 8
transformer_model = NeuralNetwork.transformer_model(ntoken, ninp, 16, hidden_layer_constant*ninp, 16)
transformer_model.compile(optimizer='adam', loss='mse')

# Convert your data to TensorFlow tensors
train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)

# Fit the model
history = transformer_model.fit(train_data, train_labels, epochs=10, batch_size=4)
