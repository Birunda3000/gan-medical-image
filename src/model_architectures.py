# -*- coding: utf-8 -*-
"""
Módulo para as arquiteturas da ACGAN (Auxiliary Classifier GAN).

Define um Gerador que recebe o ruído e a classe, e um Discriminador
com duas saídas: real/falso e classificação da classe.
"""
from . import config
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple

try:
    from . import config
except ImportError:
    import config

def build_generator(
    latent_dim: int = config.Model.LATENT_DIM,
    num_classes: int = config.NUM_CLASSES
) -> Model:
    """
    Constrói o Gerador da ACGAN com duas entradas.
    """
    # Entrada 1: Vetor de ruído latente
    latent_input = layers.Input(shape=(latent_dim,), name="generator_latent_input")
    
    # Entrada 2: Rótulo da classe
    label_input = layers.Input(shape=(1,), name="generator_label_input")
    
    # Camada de Embedding para o rótulo
    # Transforma o rótulo num vetor denso e o redimensiona
    label_embedding = layers.Embedding(num_classes, latent_dim)(label_input)
    label_embedding = layers.Dense(7 * 7)(label_embedding)
    label_embedding = layers.Reshape((7, 7, 1))(label_embedding)
    
    # Prepara o vetor de ruído
    noise = layers.Dense(7 * 7 * 255, use_bias=False)(latent_input)
    noise = layers.Reshape((7, 7, 255))(noise)
    
    # Concatena o ruído e o rótulo processado
    merged_input = layers.Concatenate()([noise, label_embedding])
    
    # Arquitetura de upsampling (igual à anterior)
    x = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(merged_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    final_channels = 1 if config.Model.COLOR_MODE == 'grayscale' else 3
    output_image = layers.Conv2DTranspose(final_channels, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')(x)

    # Cria o modelo final com duas entradas e uma saída
    model = Model(inputs=[latent_input, label_input], outputs=output_image, name="generator")
    print("✅ Modelo Gerador (ACGAN) construído com sucesso.")
    return model

def build_discriminator(
    input_shape: Tuple[int, int, int] = (config.Model.IMG_HEIGHT, config.Model.IMG_WIDTH, config.Model.CHANNELS),
    num_classes: int = config.NUM_CLASSES
) -> Model:
    """
    Constrói o Discriminador da ACGAN com duas saídas.
    """
    image_input = layers.Input(shape=input_shape, name="discriminator_image_input")
    
    # Corpo principal do discriminador (igual ao anterior)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(image_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Flatten()(x)
    
    # "Cabeça" de Saída 1: Fonte (Real ou Falso)
    source_output = layers.Dense(1, activation='sigmoid', name='source_output')(x)
    
    # "Cabeça" de Saída 2: Classe da Imagem
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)
    
    # Cria o modelo final com uma entrada e duas saídas
    model = Model(inputs=image_input, outputs=[source_output, class_output], name="discriminator")
    print("✅ Modelo Discriminador (ACGAN) construído com sucesso.")
    return model

if __name__ == '__main__':
    print("--- A testar a construção dos modelos ACGAN ---")
    generator = build_generator()
    generator.summary()
    print("\n" + "="*50 + "\n")
    discriminator = build_discriminator()
    discriminator.summary()