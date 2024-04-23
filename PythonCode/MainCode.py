import FolderHandlers
import WavHandler
import NeuralNetwork
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

output_folder = "C:\\Users\\STEVE\\OneDrive\\Documents\\University\\Diploma work\\Code\\MusicData\\Clips"

#WavHandler.prepare_slices(0.1, 0.01, 24000)

slices = WavHandler.return_slices(percentage=10)
ntoken = FolderHandlers.count_wavs_in_folder(output_folder)
ninp = WavHandler.length_of_a_clip(output_folder)
hidden_layer_constant = 3
embed_dim = 1024
num_heads = 8
transformer_model = NeuralNetwork.transformer_model(ntoken, ninp, 24, hidden_layer_constant*ninp, 24)
transformer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

