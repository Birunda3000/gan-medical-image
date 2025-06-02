import tensorflow as tf
from tensorflow.keras import layers


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
generator_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, learning_rate=1e-5)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def make_generator_model(input_dim=110):
    model = tf.keras.Sequential()

    # Inicia o gerador com uma camada densa que mapeia o ruído + label para um tensor 4x4x512
    # Isso fornece uma base rica em features para o upsampling.
    model.add(layers.Dense(4*4*512, use_bias=False, input_shape=(input_dim,)))
    model.add(layers.BatchNormalization()) # BatchNormalization é crucial para estabilidade [cite: 1]
    model.add(layers.LeakyReLU(alpha=0.2)) # Usar alpha para LeakyReLU [cite: 1]

    # Reshape para a primeira camada convolucional transposta (deconvolucional)
    model.add(layers.Reshape((4, 4, 512)))
    assert model.output_shape == (None, 4, 4, 512)

    # --- Bloco de Camadas Convolucionais Transpostas (Upsampling) ---
    # Cada camada dobra a dimensão espacial (strides=(2,2))

    # Camada 1: 4x4x512 -> 8x8x256
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Camada 2: 8x8x256 -> 16x16x128
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Camada 3: 16x16x128 -> 32x32x64
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Camada 4: 32x32x64 -> 64x64x32
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # Camada 5 (Final): 64x64x32 -> 128x128x1
    # A última camada de Conv2DTranspose tem 1 filtro para imagem em escala de cinza.
    # A ativação 'tanh' é padrão para saída de geradores, normalizando pixels para [-1, 1].
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 128, 128, 1)

    return model