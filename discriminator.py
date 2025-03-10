import tensorflow as tf
from tensorflow.keras import layers

def make_discriminator_model(num_of_labels=10):
    # Input para a imagem
    image_input = layers.Input(shape=(28, 28, 1), name="image_input")
    # Input para o vetor one-hot com num_of_labels classes
    label_input = layers.Input(shape=(num_of_labels,), name="label_input")

    # Processamento da imagem via convoluções
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(image_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Flatten()(x)

    # Parte densa do discriminador para extração de features
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    features = layers.Dense(16, activation='tanh')(x)  # Vetor de features com dimensão 16    
    features = layers.BatchNormalization()(features)

    # Separar as features em dois grupos:
    # - x_incond: características que não serão moduladas (primeiros 6 neurônios)
    # - x_cond: características que serão combinadas com o rótulo (últimos 10 neurônios)
    x_incond = layers.Lambda(lambda z: z[:, :6], name="x_incond")(features)
    x_cond = layers.Lambda(lambda z: z[:, 6:], name="x_cond")(features)

    # Cria um embedding para o rótulo, mapeando o vetor one-hot para dimensão 10
    label_embedding = layers.Dense(10, activation='tanh', name="label_embedding")(label_input)

    # Normaliza x_cond e label_embedding para que tenham norma 1 (similaridade cosseno)
    normalize = lambda tensor: tf.math.l2_normalize(tensor, axis=1)
    x_cond_normalized = layers.Lambda(normalize, name="x_cond_normalized")(x_cond)
    label_embedding_normalized = layers.Lambda(normalize, name="label_embedding_normalized")(label_embedding)

    # Calcula o produto escalar (similaridade cosseno) entre x_cond_normalized e label_embedding_normalized
    dot_product = layers.Dot(axes=1, name="dot_product")([x_cond_normalized, label_embedding_normalized])
    # Resultado: tensor com shape (batch_size, 1)

    # Combina as features incondicionais, condicionais e o valor do produto escalar
    concatenated = layers.Concatenate(name="concatenated_features")([x_incond, x_cond, dot_product])

    # Parte densa final para a decisão
    x = layers.BatchNormalization(name="batch_norm")(concatenated)
    x = layers.Dense(4, activation='tanh', name="dense_final")(x)  # Valores entre -1 e 1
    output = layers.Dense(1, activation='sigmoid', name="output")(x)  # Saída: probabilidade de real/falso

    # Define o modelo com os dois inputs e uma saída
    model = tf.keras.Model(inputs=[image_input, label_input], outputs=output)
    return model










'''def make_discriminator_model(num_of_labels=10):
    # Input para a imagem
    image_input = layers.Input(shape=(28, 28, 1), name="image_input")
    # Input para o vetor one-hot com 2 classes
    label_input = layers.Input(shape=(num_of_labels,), name="label_input")

    # Processamento da imagem via convoluções
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(image_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Flatten()(x)

    # Parte densa do discriminador para extração de features
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)


    features = layers.Dense(16, activation='tanh')(x)  # Vetor de features com dimensão 16    
    features = layers.BatchNormalization()(features)

    # Separar as features em dois grupos:
    # - x_incond: características que não serão moduladas (primeiros 6 neurônios)
    # - x_cond: características que serão combinadas com o rótulo (últimos 10 neurônios)
    x_incond = layers.Lambda(lambda z: z[:, :6])(features)
    x_cond = layers.Lambda(lambda z: z[:, 6:])(features)

    # Cria um embedding para o rótulo, mapeando o vetor one-hot para dimensão 10
    label_embedding = layers.Dense(10, activation='tanh')(label_input)


    # Calcula o produto escalar entre as features condicionais e o embedding do rótulo
    dot_product = layers.Dot(axes=1)([x_cond, label_embedding])
    # O resultado tem dimensão (batch_size, 1)


    # Combina as features incondicionais, condicionais e o valor do produto escalar
    concatenated = layers.Concatenate()([x_incond, x_cond, dot_product])

    # Parte densa final para a decisão
    x = layers.Dense(4, activation='tanh')(concatenated)  # Valores entre -1 e 1
    output = layers.Dense(1, activation='sigmoid')(x)  # Saída: probabilidade de real/falso

    # Define o modelo com os dois inputs e uma saída
    model = tf.keras.Model(inputs=[image_input, label_input], outputs=output)
    return model'''





#-------------------------------------------------------------------------------------------------------------

'''def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(16))
    model.add(layers.Dense(4))
    model.add(layers.Dense(1))#isso vai pra dois---------------------------------

    return model'''


'''def make_discriminator_model():
    # Input para a imagem
    image_input = layers.Input(shape=(28, 28, 1), name="image_input")
    # Input para o vetor one-hot com 2 classes
    label_input = layers.Input(shape=(2,), name="label_input")

    # Processamento da imagem via convoluções
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(image_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Flatten()(x)

    # Concatena as features extraídas com o vetor one-hot

    # Parte densa do discriminador

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(16, activation='relu')(x)



    # adicionado para produto escalar
    # Cria um embedding linear para o rótulo, mapeando o vetor one-hot para dimensão 16
    label_embedding = layers.Dense(16, activation='linear')(label_input)
    # Calcula o produto escalar (dot product) entre as features e o embedding do rótulo
    dot_product = layers.Dot(axes=1)([x, label_embedding])
    # O dot_product tem forma (batch_size, 1)

    # Concatena as features com o valor do produto escalar para incorporar a interação
    x = layers.Concatenate()([x, dot_product])
    # adicionado para produto escalar


    #x = layers.Concatenate()([x, label_input])
    x = layers.Dense(8, activation='relu')(x)
    x = layers.Dense(4, activation='tanh')(x)  # Mudança na ativação para valores entre -1 e 1
    output = layers.Dense(1, activation='sigmoid')(x)  # Saída com ativação sigmoid para classificação binária

    # Define o modelo com dois inputs e uma saída
    model = tf.keras.Model(inputs=[image_input, label_input], outputs=output)
    return model'''