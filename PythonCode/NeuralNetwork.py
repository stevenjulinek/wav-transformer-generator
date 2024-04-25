import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
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