# -*- coding: utf-8 -*-
"""
Script Principal para Treino da ACGAN
"""
import tensorflow as tf
import numpy as np
import time
from datetime import datetime

# Importa todos os nossos módulos
from src import config, data_loader, model_architectures, reporting, visualization


def train():
    """Função principal que executa o fluxo de treino."""
    # --- INICIALIZAÇÃO ---
    process_start_time = time.time()
    process_start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    generator = model_architectures.build_generator()
    discriminator = model_architectures.build_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(
        config.Training.GENERATOR_LR, beta_1=config.Training.ADAM_BETA_1
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        config.Training.DISCRIMINATOR_LR, beta_1=config.Training.ADAM_BETA_1
    )

    source_loss_fn = tf.keras.losses.BinaryCrossentropy()
    class_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def train_step(real_images, real_labels):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, config.Model.LATENT_DIM])
        fake_labels = tf.random.uniform(
            [batch_size], minval=0, maxval=config.NUM_CLASSES, dtype=tf.int32
        )
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator([noise, fake_labels], training=True)
            real_source_output, real_class_output = discriminator(
                real_images, training=True
            )
            fake_source_output, fake_class_output = discriminator(
                generated_images, training=True
            )
            real_source_loss = source_loss_fn(
                tf.ones_like(real_source_output), real_source_output
            )
            fake_source_loss = source_loss_fn(
                tf.zeros_like(fake_source_output), fake_source_output
            )
            real_class_loss = class_loss_fn(real_labels, real_class_output)
            discriminator_loss = real_source_loss + fake_source_loss + real_class_loss
            generator_source_loss = source_loss_fn(
                tf.ones_like(fake_source_output), fake_source_output
            )
            generator_class_loss = class_loss_fn(fake_labels, fake_class_output)
            generator_loss = generator_source_loss + generator_class_loss
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

    # --- CARREGAMENTO DE DADOS ---
    print("\n[ETAPA 1/3] Carregando dataset...")
    dataset, num_images = data_loader.create_acgan_dataset(
        config.CLASS_COUNT_DIR, config.Training.BATCH_SIZE
    )
    seed_noise = tf.random.normal([16, config.Model.LATENT_DIM])
    seed_labels = tf.constant(np.arange(16) % config.NUM_CLASSES)

    # --- LOOP DE TREINO ---
    print("\n[ETAPA 2/3] Iniciando o loop de treino...")
    g_loss_history, d_loss_history = [], []
    for epoch in range(config.Training.EPOCHS):
        start_epoch_time = time.time()
        gen_loss_epoch, disc_loss_epoch = [], []
        for image_batch, label_batch in dataset:
            g_loss, d_loss = train_step(image_batch, label_batch)
            gen_loss_epoch.append(g_loss)
            disc_loss_epoch.append(d_loss)

        avg_g_loss = np.mean(gen_loss_epoch)
        avg_d_loss = np.mean(disc_loss_epoch)
        g_loss_history.append(avg_g_loss)
        d_loss_history.append(avg_d_loss)

        print(
            f"Época {epoch + 1}/{config.Training.EPOCHS} | Perda Gerador: {avg_g_loss:.4f} | Perda Discriminador: {avg_d_loss:.4f} | Tempo: {time.time()-start_epoch_time:.2f} sec"
        )

        if (epoch + 1) % config.Training.SAVE_INTERVAL == 0:
            visualization.generate_and_save_images(
                generator, epoch + 1, [seed_noise, seed_labels]
            )
            generator.save(str(config.Paths.GENERATOR_MODEL_FILE))
            print(f"✅ Modelo do gerador salvo em: {config.Paths.GENERATOR_MODEL_FILE}")

    # --- FINALIZAÇÃO ---
    print("\n[ETAPA 3/3] Treino concluído! Finalizando a execução...")
    process_end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_g_loss = g_loss_history[-1]
    final_d_loss = d_loss_history[-1]
    new_run_dir_name = f"{config.PROJECT_NAME}___g_loss={final_g_loss:.2f}_d_loss={final_d_loss:.2f}___{config.TIMESTAMP}"
    final_run_dir = config.Paths.OUTPUT_DIR / new_run_dir_name

    try:
        config.Paths.TMP_RUN_DIR.rename(final_run_dir)
        print(f"✅ Diretório da execução finalizado e salvo como: {final_run_dir.name}")
    except Exception as e:
        print(f"Erro ao finalizar o diretório da execução: {e}")
        final_run_dir = config.Paths.TMP_RUN_DIR

    visualization.create_training_gif(final_run_dir)

    try:
        fig = visualization.plot_gan_losses(g_loss_history, d_loss_history)
        loss_plot_path = final_run_dir / "loss_history.png"
        fig.savefig(loss_plot_path)
        plt.close(fig)
        print(f"Gráfico do histórico de perdas salvo em: {loss_plot_path}")
    except Exception as e:
        print(f"Não foi possível salvar o gráfico do histórico de perdas: {e}")

    timings = {
        "start_dt": process_start_dt,
        "end_dt": process_end_dt,
        "total_duration": time.time() - process_start_time,
    }
    final_losses = {"g_loss": final_g_loss, "d_loss": final_d_loss}
    report_path = final_run_dir / "report.txt"
    reporting.save_gan_report(report_path, timings, num_images, final_losses)


if __name__ == "__main__":
    train()
