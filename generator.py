import tensorflow as tf
from tensorflow.keras import layers

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-4,
    decay_steps=28,
    decay_rate=0.5,
    staircase=True) 

generator_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, learning_rate=2e-4)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def make_generator_model(input_dim=110, output_shape=(128, 128, 1)): # Default para 128x128x1
    model = tf.keras.Sequential()

    # Camada densa inicial, projeção para uma base espacial pequena com muitos filtros
    # Base 4x4 para 5 etapas de upsampling (4 -> 8 -> 16 -> 32 -> 64 -> 128)
    model.add(layers.Dense(4*4*512, use_bias=False, input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((4, 4, 512)))
    assert model.output_shape == (None, 4, 4, 512)  # Verifica a saída da camada reshape
    # Output esperado: (None, 4, 4, 512)

    # Bloco de Upsampling 1: 4x4x512 -> 8x8x256
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # Output esperado: (None, 8, 8, 256)

    # Bloco de Upsampling 2: 8x8x256 -> 16x16x128
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 16, 16, 128)  # Verifica a saída da camada
    # Output esperado: (None, 16, 16, 128)

    # Bloco de Upsampling 3: 16x16x128 -> 32x32x64
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # Output esperado: (None, 32, 32, 64)

    # Bloco de Upsampling 4: 32x32x64 -> 64x64x32
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # Output esperado: (None, 64, 64, 32)

    # Bloco de Upsampling 5 (Camada de Saída): 64x64x32 -> 128x128xN_CANAIS
    # O número de canais é definido por output_shape[-1] (1 para escala de cinza)
    model.add(layers.Conv2DTranspose(output_shape[-1], (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, output_shape[-3], output_shape[-2], output_shape[-1])  # Verifica a saída da camada final
    # Output esperado: (None, 128, 128, output_shape[-1])

    return model




'''
# Corrigido para funcionar bem com MNIST (output_shape=(28,28,1))
def make_generator_model(input_dim=110, output_shape=(128, 128, 1)): # Default para 128x128x1
    model = tf.keras.Sequential()

    # Camada densa inicial, projeção para uma base espacial pequena com muitos filtros
    # Base 4x4 para 5 etapas de upsampling (4 -> 8 -> 16 -> 32 -> 64 -> 128)
    model.add(layers.Dense(4*4*512, use_bias=False, input_shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((4, 4, 512)))
    assert model.output_shape == (None, 4, 4, 512)  # Verifica a saída da camada reshape
    # Output esperado: (None, 4, 4, 512)

    # Bloco de Upsampling 1: 4x4x512 -> 8x8x256
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # Output esperado: (None, 8, 8, 256)

    # Bloco de Upsampling 2: 8x8x256 -> 16x16x128
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    assert model.output_shape == (None, 16, 16, 128)  # Verifica a saída da camada
    # Output esperado: (None, 16, 16, 128)

    # Bloco de Upsampling 3: 16x16x128 -> 32x32x64
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # Output esperado: (None, 32, 32, 64)

    # Bloco de Upsampling 4: 32x32x64 -> 64x64x32
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    # Output esperado: (None, 64, 64, 32)

    # Bloco de Upsampling 5 (Camada de Saída): 64x64x32 -> 128x128xN_CANAIS
    # O número de canais é definido por output_shape[-1] (1 para escala de cinza)
    model.add(layers.Conv2DTranspose(output_shape[-1], (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    assert model.output_shape == (None, output_shape[-3], output_shape[-2], output_shape[-1])  # Verifica a saída da camada final
    # Output esperado: (None, 128, 128, output_shape[-1])



    return model
'''