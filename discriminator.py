import tensorflow as tf
from tensorflow.keras import layers


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
discriminator_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, learning_rate=1e-5)

def discriminator_loss(real_output, fake_output):
    
    real_smooth_labels = tf.ones_like(real_output) * 0.9  # Smoothing the labels for real images
    real_loss = cross_entropy(real_smooth_labels, real_output)
    #real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    
    fake_smooth_labels = tf.zeros_like(fake_output) 
    fake_loss = cross_entropy(fake_smooth_labels, fake_output)
    
    total_loss = real_loss + fake_loss
    return total_loss


def make_discriminator_model(num_of_labels=10):
    # Input para a imagem
    image_input = layers.Input(shape=(128, 128, 1), name="image_input")
    # Input para o vetor one-hot com o número de classes (e.g., 2 para COVID/Normal)
    label_input = layers.Input(shape=(num_of_labels,), name="label_input")

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(image_input)
    x = layers.BatchNormalization()(x) # Adicionar BatchNormalization
    x = layers.LeakyReLU(alpha=0.2)(x) # Usar alpha para LeakyReLU

    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x) # Acha um tensor unidimensional para as camadas densas

    x = layers.Dense(256, activation='relu')(x) # Mais camadas para maior capacidade
    x = layers.BatchNormalization()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x) # Dropout para evitar overfitting

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    features = layers.Dense(num_of_labels * 2, activation='tanh')(x) # Ajustar dimensão do vetor de features

    # Separar as features para condicionamento (ACGAN-like)
    # A dimensão das features condicionais deve ser igual à dimensão do embedding do rótulo
    # Exemplo: Se 'features' tem 20 neurônios, dividir em 10 incondicionais e 10 condicionais
    x_incond = layers.Lambda(lambda z: z[:, :num_of_labels])(features) # Primeiros 'num_of_labels' neurônios
    x_cond = layers.Lambda(lambda z: z[:, num_of_labels:])(features)   # Últimos 'num_of_labels' neurônios

    # Cria um embedding para o rótulo, mapeando o vetor one-hot para a mesma dimensão de x_cond
    label_embedding = layers.Dense(num_of_labels, activation='tanh')(label_input) # Dimensão do embedding deve casar com x_cond

    # Calcula o produto escalar entre as features condicionais e o embedding do rótulo
    dot_product = layers.Dot(axes=1)([x_cond, label_embedding]) # O resultado tem dimensão (batch_size, 1)

    # Combina as features incondicionais, condicionais e o valor do produto escalar
    concatenated = layers.Concatenate()([x_incond, dot_product]) # Removi x_cond da concatenação pois já foi usado no dot_product

    # Parte densa final para a decisão Real/Falso
    # Adicionando mais camadas densas para a decisão final
    x = layers.Dense(32, activation='relu')(concatenated) # Aumentar capacidade
    x = layers.Dense(16, activation='relu')(x)
    # A última camada densa do discriminador (antes do sigmoid) deve ter saída 'from_logits=True' na loss
    # ou uma ativação linear, e o sigmoid é aplicado na loss. No entanto, se o sigmoid está aqui, a loss não deve usar from_logits=True
    # Para consistência com DCGAN, muitas vezes a última camada não tem ativação, e o sigmoid é na loss.
    # Mas com o sigmoid aqui, a loss deve ser tf.keras.losses.BinaryCrossentropy(from_logits=False)
    output = layers.Dense(1, activation='sigmoid')(x)

    # Define o modelo com os dois inputs (imagem e label) e uma saída (real/falso)
    model = tf.keras.Model(inputs=[image_input, label_input], outputs=output)
    return model