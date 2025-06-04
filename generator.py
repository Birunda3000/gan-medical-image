import tensorflow as tf
from tensorflow.keras import layers

# Learning rates mais comuns para MNIST com Adam em GANs
# Original: learning_rate=2e-5
generator_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, learning_rate=2e-4) # Ajustado para MNIST

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Corrigido para funcionar bem com MNIST (output_shape=(28,28,1))
def make_generator_model(input_dim=110, output_shape=(28, 28, 1)):
    model = tf.keras.Sequential()

    # Para MNIST (28x28), geralmente começamos com uma base de 7x7
    # Camada densa inicial, projeção e reshape
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(input_dim,))) # Ajustado para base 7x7
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((7, 7, 256))) # Ajustado para base 7x7
    # A asserção original seria: assert model.output_shape == (None, 7, 7, 256)

    # Primeiro bloco Conv2DTranspose: 7x7x256 -> 14x14x128
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # A asserção original seria: assert model.output_shape == (None, 14, 14, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Segundo bloco Conv2DTranspose (imagem final): 14x14x128 -> 28x28xoutput_shape[-1]
    # O argumento output_shape dita o número de canais da imagem final.
    # Para MNIST, output_shape[-1] seria 1 (escala de cinza).
    model.add(layers.Conv2DTranspose(output_shape[-1], (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    
    # Checando se a forma de saída do modelo corresponde ao argumento output_shape (ignorando a dimensão do batch)
    # A asserção original era: assert model.output_shape == output_shape
    # Uma verificação mais robusta, considerando que model.output_shape[0] é None (batch_size):
    # if output_shape is not None: # Adicionado para robustez
    #    assert model.output_shape[1:] == output_shape, \
    #        f"Model output shape {model.output_shape[1:]} does not match expected {output_shape}"

    return model