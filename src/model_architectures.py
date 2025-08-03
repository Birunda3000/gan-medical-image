# -*- coding: utf-8 -*-
"""
Módulo para as arquiteturas da ACGAN (Versão Final Corrigida)

Garante que as dimensões dinâmicas ('original', 'auto') sejam
traduzidas corretamente para o formato que o Keras entende (None).
"""
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple

try:
    from . import config
except ImportError:
    import config

def build_generator(latent_dim: int = None, num_classes: int = None) -> Model:
    # ... (esta função permanece igual, sem alterações)
    if latent_dim is None: latent_dim = config.Model.LATENT_DIM
    if num_classes is None: num_classes = config.NUM_CLASSES
    latent_input = layers.Input(shape=(latent_dim,), name="generator_latent_input")
    label_input = layers.Input(shape=(1,), name="generator_label_input")
    label_embedding = layers.Embedding(num_classes, latent_dim)(label_input)
    label_embedding = layers.Dense(7 * 7)(label_embedding)
    label_embedding = layers.Reshape((7, 7, 1))(label_embedding)
    noise = layers.Dense(7 * 7 * 255, use_bias=False)(latent_input)
    noise = layers.Reshape((7, 7, 255))(noise)
    merged_input = layers.Concatenate()([noise, label_embedding])
    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(merged_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    final_channels = config.Model.CHANNELS if config.Model.COLOR_MODE != 'auto' else 3
    output_image = layers.Conv2DTranspose(final_channels, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)
    model = Model(inputs=[latent_input, label_input], outputs=output_image, name="generator")
    print("✅ Modelo Gerador (ACGAN) construído com sucesso.")
    return model

def build_discriminator(input_shape: Tuple[int, int, int] = None, num_classes: int = None) -> Model:
    """Constrói o Discriminador da ACGAN com duas saídas."""
    if input_shape is None:
        img_height = None if config.Model.IMG_HEIGHT == 'original' else config.Model.IMG_HEIGHT
        img_width = None if config.Model.IMG_WIDTH == 'original' else config.Model.IMG_WIDTH
        channels = config.Model.CHANNELS
        input_shape = (img_height, img_width, channels)
        
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    image_input = layers.Input(shape=input_shape, name="discriminator_image_input")
    
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(image_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    # Substituí a camada Flatten por GlobalAveragePooling2D.
    # Isto cria uma saída de tamanho fixo, independentemente do tamanho da imagem de entrada.
    x = layers.GlobalAveragePooling2D()(x)
    
    # Agora a camada Dense recebe uma entrada de tamanho fixo e bem definido.
    source_output = layers.Dense(1, activation='sigmoid', name='source_output')(x)
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    
    model = Model(inputs=image_input, outputs=[source_output, class_output], name="discriminator")
    print("✅ Modelo Discriminador (ACGAN) construído com sucesso.")
    return model