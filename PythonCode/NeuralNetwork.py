import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

import FolderHandlers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="sigmoid"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Define the model
def transformer_model(ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
    inputs = layers.Input(shape=(None,))
    x = layers.Embedding(ntoken, ninp)(inputs)

    # Use the TransformerBlock here
    for _ in range(nlayers):
        x = TransformerBlock(ninp, nhead, nhid, dropout)(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(ninp)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def generate_sequence(model, start_sequence, length, lookback):
    """Generate a sequence using a model's predictions."""
    print("Generating sequence.")

    # Starting timer for the generation
    start_time = time.time()

    sequence = start_sequence[-lookback:]
    # Convert the sequence list to a numpy array
    sequence = np.array(sequence)

    # Generate the desired number of outputs
    for _ in tqdm(range(length), bar_format='\033[37m{l_bar}{bar:40}{r_bar}\033[0m'):
        # Make a prediction based on the current sequence
        prediction = model.predict(sequence, verbose=0, batch_size=1)

        # Convert the prediction to integer
        prediction = prediction.astype(int)

        prediction = tf.convert_to_tensor(prediction)

        # Reshape the last prediction to a 1D array with a single element
        prediction = np.reshape(prediction[-1], (1, 240))

        # Append the prediction to the sequence
        sequence = np.concatenate((sequence, prediction), axis=0)

        # Use only the last part of the sequence for the next input
        sequence = sequence[-lookback:]

    # Print the time taken for the generation
    print("Time taken for generation: %s seconds" % (time.time() - start_time))

    return sequence[-length:]

def save_training_plot(history, model_folder, do_show=False):
    import matplotlib.pyplot as plt
    # Plot the training loss
    plt.plot(history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    if do_show:
        plt.show()

    # Save the plot
    next_version = FolderHandlers.find_highest_version(base_filename="loss_plot", directory=model_folder) + 1
    plt.savefig(model_folder + 'loss_plot_' + str(next_version) + '.png')
