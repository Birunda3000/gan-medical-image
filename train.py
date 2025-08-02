# -*- coding: utf-8 -*-
"""
Script Principal para Treino da ACGAN (Auxiliary Classifier GAN).
"""
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
import imageio.v2 as imageio
from pathlib import Path
from src import config, data_loader, model_architectures, visualization


# --- 1. Inicialização dos Modelos e Otimizadores ---

# Constrói o Gerador e o Discriminador
generator = model_architectures.build_generator()
discriminator = model_architectures.build_discriminator()

# Otimizadores separados para cada rede
generator_optimizer = tf.keras.optimizers.Adam(
    config.Training.GENERATOR_LR, beta_1=config.Training.ADAM_BETA_1
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    config.Training.DISCRIMINATOR_LR, beta_1=config.Training.ADAM_BETA_1
)

# Funções de perda
# Para a saída de real/falso (Fonte)
source_loss_fn = tf.keras.losses.BinaryCrossentropy()
# Para a saída de classificação da classe
class_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()


# --- 2. O Coração do Treino: A Função `train_step` ---


@tf.function
def train_step(real_images, real_labels):
    """Executa um único passo de treino para o Gerador e o Discriminador."""

    # Gera um lote de ruído e rótulos aleatórios para o Gerador
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, config.Model.LATENT_DIM])
    fake_labels = tf.random.uniform(
        [batch_size], minval=0, maxval=config.NUM_CLASSES, dtype=tf.int32
    )

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Gera um lote de imagens falsas
        generated_images = generator([noise, fake_labels], training=True)

        # Obtém as previsões do Discriminador para as imagens reais e falsas
        real_source_output, real_class_output = discriminator(
            real_images, training=True
        )
        fake_source_output, fake_class_output = discriminator(
            generated_images, training=True
        )

        # --- Calcula as Perdas (Losses) ---

        # Perda do Discriminador
        real_source_loss = source_loss_fn(
            tf.ones_like(real_source_output), real_source_output
        )
        fake_source_loss = source_loss_fn(
            tf.zeros_like(fake_source_output), fake_source_output
        )
        real_class_loss = class_loss_fn(real_labels, real_class_output)
        discriminator_loss = real_source_loss + fake_source_loss + real_class_loss

        # Perda do Gerador (ele quer que o Discriminador pense que as suas imagens são reais e da classe correta)
        generator_source_loss = source_loss_fn(
            tf.ones_like(fake_source_output), fake_source_output
        )
        generator_class_loss = class_loss_fn(fake_labels, fake_class_output)
        generator_loss = generator_source_loss + generator_class_loss

    # --- Calcula e Aplica os Gradientes ---

    gradients_of_generator = gen_tape.gradient(
        generator_loss, generator.trainable_variables
    )
    gradients_of_discriminator = disc_tape.gradient(
        discriminator_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    return generator_loss, discriminator_loss


# --- 3. Função de Treino Principal ---


def train():
    """Orquestra o loop de treino completo."""
    print("\n[ETAPA 1/3] Carregando dataset...")
    # (dentro da função train() em train.py)
    dataset, num_images = data_loader.create_acgan_dataset(
        config.CLASS_COUNT_DIR, config.Training.BATCH_SIZE
    )

    # Cria um vetor de ruído fixo para visualização. A mesma semente sempre gera as mesmas imagens.
    seed_noise = tf.random.normal([16, config.Model.LATENT_DIM])
    seed_labels = tf.constant(np.arange(16) % config.NUM_CLASSES)

    print("\n[ETAPA 2/3] Iniciando o loop de treino...")
    for epoch in range(config.Training.EPOCHS):
        start = time.time()

        gen_loss_epoch = []
        disc_loss_epoch = []

        for image_batch, label_batch in dataset:
            g_loss, d_loss = train_step(image_batch, label_batch)
            gen_loss_epoch.append(g_loss)
            disc_loss_epoch.append(d_loss)

        # No final da época
        avg_gen_loss = np.mean(gen_loss_epoch)
        avg_disc_loss = np.mean(disc_loss_epoch)

        print(
            f"Época {epoch + 1}/{config.Training.EPOCHS} | "
            f"Perda Gerador: {avg_gen_loss:.4f} | "
            f"Perda Discriminador: {avg_disc_loss:.4f} | "
            f"Tempo: {time.time()-start:.2f} sec"
        )

        # Salva imagens de amostra e o modelo em intervalos definidos
        if (epoch + 1) % config.Training.SAVE_INTERVAL == 0:
            generate_and_save_images(generator, epoch + 1, [seed_noise, seed_labels])
            generator.save(config.Paths.GENERATOR_MODEL_FILE)
            print(f"✅ Modelo do gerador salvo em: {config.Paths.GENERATOR_MODEL_FILE}")

    # --- ETAPA 3: Finalização ---
    print("\n[ETAPA 3/3] Treino concluído!")
    generate_and_save_images(
        generator, config.Training.EPOCHS, [seed_noise, seed_labels]
    )
    create_training_gif()


# --- 4. Funções de Utilidade ---


def generate_and_save_images(model, epoch, test_input):
    """Gera imagens e as salva num ficheiro."""
    predictions = model(test_input, training=False)
    # Converte a saída de [-1, 1] para [0, 255]
    predictions = (predictions * 127.5 + 127.5).numpy().astype(np.uint8)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap="gray")
        plt.axis("off")

    save_path = config.Paths.GENERATED_IMAGES_DIR / f"image_at_epoch_{epoch:04d}.png"
    plt.savefig(save_path)
    plt.close(fig)


def create_training_gif():
    """Cria um GIF animado a partir das imagens salvas."""
    anim_file = config.Paths.TMP_RUN_DIR / "acgan.gif"
    with imageio.get_writer(anim_file, mode="I") as writer:
        filenames = sorted(config.Paths.GENERATED_IMAGES_DIR.glob("*.png"))
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"✅ GIF de treino salvo em: {anim_file}")


if __name__ == "__main__":
    # Importa matplotlib e numpy aqui para manter a parte de cima limpa
    import matplotlib.pyplot as plt
    import numpy as np

    train()
