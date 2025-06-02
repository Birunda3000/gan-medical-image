import time
import tensorflow as tf
from IPython.display import display, clear_output # Importe aqui se ainda não o fez

# Supondo que as seguintes variáveis/funções já estão definidas:
# generator, discriminator, seed, train_image_out_path, num_of_classes,
# checkpoint, checkpoint_prefix, noise_dim, gen (módulo com generator_loss, etc.),
# dis (módulo com discriminator_loss, etc.), ut (módulo com generate_and_save_images),
# e a função plot_epoch_losses (definida acima ou importada).

def train(dataset, epochs, file_writer):
    start_training = time.time()
    g_loss_history = [] # Para armazenar a perda do gerador a cada época
    d_loss_history = [] # Para armazenar a perda do discriminador a cada época

    for epoch in range(epochs):
        start_epoch_time = time.time()
        
        epoch_gen_loss_accumulator = 0.0
        epoch_disc_loss_accumulator = 0.0
        num_batches = 0
        # Mantém as imagens geradas do último batch da época para o log do TensorBoard
        last_generated_images_for_log = None 

        for image_batch, label_batch in dataset:
            # A função train_step retorna as perdas do batch atual e as imagens geradas
            current_batch_gen_loss, current_batch_disc_loss, generated_images = train_step(image_batch, label_batch)
            
            epoch_gen_loss_accumulator += current_batch_gen_loss
            epoch_disc_loss_accumulator += current_batch_disc_loss
            last_generated_images_for_log = generated_images # Salva para o log do TensorBoard
            num_batches += 1
        
        # Calcula a média das perdas para a época
        avg_epoch_gen_loss = epoch_gen_loss_accumulator / num_batches
        avg_epoch_disc_loss = epoch_disc_loss_accumulator / num_batches

        # Adiciona as perdas médias da época ao histórico
        g_loss_history.append(avg_epoch_gen_loss.numpy()) # .numpy() para obter o valor escalar
        d_loss_history.append(avg_epoch_disc_loss.numpy())

        # Limpa a saída da célula do notebook ANTES de plotar ou imprimir novas informações da época
        display.clear_output(wait=True)

        # Chama a função de plotagem modular
        if epoch > 0 : # Opcional: não plotar na primeira época se a lista tiver apenas 1 ponto
             plot_epoch_losses(epoch + 1, g_loss_history, d_loss_history)

        # Exibir imagens geradas (sua função ut.generate_and_save_images)
        ut.generate_and_save_images(generator, epoch + 1, seed, train_image_out_path, num_classes=num_of_classes)
        
        # Logs para o TensorBoard
        if last_generated_images_for_log is not None: # Garante que houve pelo menos um batch
            with file_writer.as_default():
                tf.summary.scalar('perda_media_gerador_epoca', avg_epoch_gen_loss, step=epoch)
                tf.summary.scalar('perda_media_discriminador_epoca', avg_epoch_disc_loss, step=epoch)
                img_to_log = (last_generated_images_for_log * 0.5) + 0.5 # Normaliza de [-1,1] para [0,1]
                tf.summary.image('imagens_geradas_epoca', img_to_log, max_outputs=4, step=epoch)
        
        # Imprime informações da época
        print(f"Época {epoch + 1}/{epochs} concluída.")
        print(f"  Perda Média Gerador: {avg_epoch_gen_loss.numpy():.4f}")
        print(f"  Perda Média Discriminador: {avg_epoch_disc_loss.numpy():.4f}")
        print(f"  Tempo da Época: {time.time()-start_epoch_time:.2f} seg")
        print("  Rodando em GPU" if tf.config.list_physical_devices('GPU') else "  Rodando em CPU")

        # Salvar modelo (checkpoint)
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    # Após o término de todas as épocas
    # O último plot e geração de imagem já foram feitos no loop,
    # mas você pode querer um display final se desejar.
    # display.clear_output(wait=True) # Se quiser limpar tudo no final
    # plot_epoch_losses(epochs, g_loss_history, d_loss_history)
    # ut.generate_and_save_images(generator, epochs, seed, train_image_out_path, num_classes=num_of_classes)
    print(f'\nTempo Total de Treinamento: {time.time()-start_training:.2f} seg após {epochs} épocas.')

# Sua função train_step permanece inalterada:
# @tf.function(jit_compile=True)
# def train_step(images, labels):
#     real_batch_size = tf.shape(images)[0]
#     noise = tf.random.normal([real_batch_size, noise_dim])
#     labels = tf.cast(labels, tf.float32)
#     noise = tf.concat([noise, labels], axis=1)
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator(noise, training=True)
#         real_output = discriminator([images, labels], training=True)
#         fake_output = discriminator([generated_images, labels], training=True)
#         gen_loss = gen.generator_loss(fake_output)
#         disc_loss = dis.discriminator_loss(real_output, fake_output)
#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#     gen.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     dis.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
#     return gen_loss, disc_loss, generated_images