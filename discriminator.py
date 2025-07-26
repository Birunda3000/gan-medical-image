import tensorflow as tf
from tensorflow.keras import layers

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-4,
    decay_steps=28,
    decay_rate=0.5,
    staircase=True) 

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
discriminator_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, learning_rate=2e-4)

def discriminator_loss(real_output, fake_output):
    real_smooth_labels = tf.ones_like(real_output) * 0.9  # Label smoothing para imagens reais
    real_loss = cross_entropy(real_smooth_labels, real_output)

    fake_smooth_labels = tf.zeros_like(fake_output)
    fake_loss = cross_entropy(fake_smooth_labels, fake_output)
    
    total_loss = real_loss + fake_loss
    return total_loss


def make_discriminator_model(num_of_labels=4, input_shape=(128, 128, 1)): # Defaults ajustados para o contexto
    # Input para a imagem
    image_input = layers.Input(shape=input_shape, name="image_input")
    # Input para o vetor one-hot com o número de classes
    label_input = layers.Input(shape=(num_of_labels,), name="label_input")

    # Sequência de camadas convolucionais para input_shape=(128,128,1):
    # Conv1: (128,128,1) -> (64,64,64)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Conv2: (64,64,64) -> (32,32,64)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)

    # Conv3: (32,32,64) -> (16,16,128)
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)

    # Conv4: (16,16,128) -> (8,8,256)
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x) # Output: 8*8*256 = 16384

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    features = layers.Dense(num_of_labels * 2, activation='tanh')(x)

    x_incond = layers.Lambda(lambda z: z[:, :num_of_labels])(features)
    x_cond = layers.Lambda(lambda z: z[:, num_of_labels:])(features)

    label_embedding = layers.Dense(num_of_labels, activation='tanh')(label_input)
    dot_product = layers.Dot(axes=1)([x_cond, label_embedding])
    concatenated = layers.Concatenate()([x_incond, dot_product])

    x = layers.Dense(32, activation='relu')(concatenated)
    x = layers.Dense(16, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[image_input, label_input], outputs=output)
    return model




'''
def make_discriminator_model(num_of_labels=4, input_shape=(128, 128, 1)): # Defaults ajustados para o contexto
    # Input para a imagem
    image_input = layers.Input(shape=input_shape, name="image_input")
    # Input para o vetor one-hot com o número de classes
    label_input = layers.Input(shape=(num_of_labels,), name="label_input")

    # Sequência de camadas convolucionais para input_shape=(128,128,1):
    # Conv1: (128,128,1) -> (64,64,64)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    # Conv2: (64,64,64) -> (32,32,64)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)

    # Conv3: (32,32,64) -> (16,16,128)
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)

    # Conv4: (16,16,128) -> (8,8,256)
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x) # Output: 8*8*256 = 16384

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    features = layers.Dense(num_of_labels * 2, activation='tanh')(x)

    x_incond = layers.Lambda(lambda z: z[:, :num_of_labels])(features)
    x_cond = layers.Lambda(lambda z: z[:, num_of_labels:])(features)

    label_embedding = layers.Dense(num_of_labels, activation='tanh')(label_input)
    dot_product = layers.Dot(axes=1)([x_cond, label_embedding])
    concatenated = layers.Concatenate()([x_incond, dot_product])

    x = layers.Dense(32, activation='relu')(concatenated)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(8, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[image_input, label_input], outputs=output)
    return model
'''