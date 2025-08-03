# -*- coding: utf-8 -*-
"""
Módulo para todas as funções de visualização do projeto de GANs.

Inclui funções para plotar perdas, gerar imagens de amostra e criar GIFs de treino.
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from typing import List
import tensorflow as tf

try:
    from . import config
except ImportError:
    import config


def plot_gan_losses(
    generator_losses: List[float], discriminator_losses: List[float]
) -> plt.Figure:
    """Plota o histórico de perdas do Gerador e do Discriminador ao longo das épocas."""
    fig = plt.figure(figsize=(12, 6))
    plt.plot(generator_losses, label="Perda do Gerador (Generator Loss)")
    plt.plot(discriminator_losses, label="Perda do Discriminador (Discriminator Loss)")
    plt.title("Histórico de Perdas do Treino da GAN")
    plt.xlabel("Épocas")
    plt.ylabel("Perda (Loss)")
    plt.legend()
    plt.grid(True)
    return fig


def generate_and_save_images(model: tf.keras.Model, epoch: int, test_input: list):
    """Gera imagens de amostra e as salva num ficheiro."""
    predictions = model(test_input, training=False)
    predictions = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)

    fig = plt.figure(figsize=(10, 10))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        # Assumindo que o segundo canal é 1 para escala de cinza
        channel = 0 if predictions.shape[-1] == 1 else slice(None)
        plt.imshow(predictions[i, :, :, channel], cmap="gray" if channel == 0 else None)

        class_number = test_input[1].numpy()[i]
        class_name = config.CLASS_NAMES[class_number]

        title_text = f"{class_name.capitalize()}\n({class_number})"
        plt.title(title_text)
        plt.axis("off")

    plt.subplots_adjust(hspace=0.5, wspace=0.1)
    save_path = config.Paths.GENERATED_IMAGES_DIR / f"image_at_epoch_{epoch:04d}.png"
    plt.savefig(save_path)
    plt.close(fig)


def create_training_gif(final_run_dir: Path):
    """Cria um GIF animado a partir das imagens de amostra salvas."""
    print("A criar o GIF de treino...")
    anim_file = final_run_dir / "acgan_training.gif"
    images_source_dir = final_run_dir / "generated_images"

    with imageio.get_writer(anim_file, mode="I", duration=0.5) as writer:
        filenames = sorted(images_source_dir.glob("*.png"))
        if not filenames:
            print("AVISO: Nenhuma imagem de amostra encontrada para criar o GIF.")
            return
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"✅ GIF de treino salvo em: {anim_file}")
