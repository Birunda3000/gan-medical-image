import tensorflow as tf
import numpy as np
import os

print('')
print('')

# Ensure TensorFlow is using the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU found")

def create_seed(num_examples_to_generate, noise_dim, num_classes):
    # Gera ruído normal de tamanho (num_examples_to_generate, noise_dim)
    noise = tf.random.normal([num_examples_to_generate, noise_dim])

    # Gera rótulos aleatórios entre 0 e (num_classes - 1)
    random_classes = tf.random.uniform([num_examples_to_generate], minval=0, maxval=num_classes, dtype=tf.int32)

    # Converte os rótulos para one-hot encoding
    one_hot_labels = tf.one_hot(random_classes, depth=num_classes)

    # Concatena o ruído com os rótulos one-hot
    seed = tf.concat([noise, one_hot_labels], axis=1)

    return seed, random_classes  # Retorna o tensor gerado e os rótulos para referência


np.set_printoptions(linewidth=os.get_terminal_size().columns-30)
np.set_printoptions(formatter={'float': '{:12.6f}'.format})# 10 casas decimais e 100 caracteres de largura
print('')

# Exemplo de uso
num_examples = 10
noise_dim = 2
num_classes = 4  # Pode ser alterado para mais classes

seed, labels = create_seed(num_examples, noise_dim, num_classes)
print("Seed shape:", seed.shape)  # Esperado: (16, 100 + num_classes)
print("Labels:", labels.numpy())  # Visualiza as classes geradas
print('')
for i in range(len(seed)):
    print(seed[i])
    print('')


