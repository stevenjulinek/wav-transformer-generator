import time

import numpy as np
from tqdm import tqdm

import FolderHandlers


import torch
from torch import nn
from torch.nn import functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, inputs):
        attn_output, _ = self.att(inputs, inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output))

class transformer_model(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(transformer_model, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(ninp, nhead, nhid, dropout) for _ in range(nlayers)])
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(nhid, ninp)

    def forward(self, x):
        x = self.embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.avg_pooling(x)
        x = self.output(x)
        return x[:, -240:]  # return only the last 240 samples


def generate_sequence(model, start_sequence, length, lookback, vae_model):
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
        prediction = model(sequence, verbose=0, batch_size=1)

        # Convert the prediction to integer
        prediction = prediction.astype(int)

        prediction = torch.tensor(prediction)

        # Reshape the last prediction to a 1D array with a single element
        prediction = np.reshape(prediction[-1], (1, 240))

        # Append the prediction to the sequence
        sequence = np.concatenate((sequence, prediction), axis=0)

        # Use only the last part of the sequence for the next input
        sequence = sequence[-lookback:]

    # Decode the generated sequence using the VAE
    decoded_sequence = vae_model.decode(sequence)

    # Print the time taken for the generation
    print("Time taken for generation: %s seconds" % (time.time() - start_time))

    return decoded_sequence[-length:]

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
