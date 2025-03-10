import random
import numpy as np
import cv2
import os

import tensorflow as tf
import matplotlib.pyplot as plt


try:
    terminal_width = os.get_terminal_size().columns - 30
except OSError:
    terminal_width = 100  # Default value if os.get_terminal_size() fails

np.set_printoptions(linewidth=terminal_width)


def get_classes(matriz) -> list:
    """
    Extracts unique classes from the given matrix.

    Parameters:
    matriz (list): A list of lists where the third element of each sublist is a class label.

    Returns:
    list: A list of unique class labels.
    """
    res = []
    for v in [x[2] for x in matriz]:
        if v not in res:
            res.append(v)
    return res


def prep_data(data, CATEGORIES, IMG_SIZE, num_of_channels, shuffle=True) -> tuple:
    """
    Prepares data for training by shuffling, extracting features and labels, and reshaping.

    Parameters:
    data (list): A list of tuples where each tuple contains features, label, and name.
    CATEGORIES (list): A list of category names.
    IMG_SIZE (int): The size to which each image will be resized.
    num_of_channels (int): The number of channels in the images.
    shuffle (bool): Whether to shuffle the data before processing. Default is True.
    for features, label in data:
    Returns:
    tuple: A tuple containing the processed features (X) and one-hot encoded labels (y).
    """

    if shuffle:
        random.shuffle(data)
    
    X = []
    y = []
    for features, label, name in data:
        X.append(features)
        y.append(label)

    y = np.eye(  len(CATEGORIES)  )[y]
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, num_of_channels)

    assert X.shape[0] == y.shape[0], "Number of images and labels do not match."

    return X, y


def prepare(filepath, IMG_SIZE) -> tuple:
    """
    Reads an image from the specified file path, converts it to grayscale, 
    resizes it to a predefined size, and returns both the resized image 
    and its reshaped version suitable for model input.
    Args:
        filepath (str): The path to the image file.
    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The resized grayscale image.
            - numpy.ndarray: The reshaped image array with dimensions 
              (-1, IMG_SIZE, IMG_SIZE, 1) suitable for model input.   
    """
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array, new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def plot_image(prediction_array, true_label, img, CATEGORIES):
    """
    Plots an image with its predicted and true labels.
    Args:
        prediction_array (numpy.ndarray): Array of prediction probabilities for each class.
        true_label (int): The true label index of the image.
        img (numpy.ndarray): The image to be plotted.
    Returns:
        None
    """
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap='gray')
    predicted_label = np.argmax(prediction_array)        
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("Classe - {} | {:2.0f}% (true class {})".format(CATEGORIES[predicted_label], 100*np.max(prediction_array), CATEGORIES[true_label]), color=color)


def plot_value_array(prediction_array, true_label, CATEGORIES):
    """
    Plots a bar chart of the prediction array with the true label highlighted.
    Parameters:
    prediction_array (list or numpy array): Array of prediction probabilities for each category.
    true_label (int): Index of the true label in the CATEGORIES list.
    The function will plot a bar for each category with the height corresponding to the prediction probability.
    The bar corresponding to the predicted label will be colored red, and the bar corresponding to the true label
    will be colored green.
    """
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(CATEGORIES)), prediction_array, color= "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_array)

    thisplot[predicted_label].set_color('red')
    print(CATEGORIES[true_label])
    thisplot[true_label].set_color('green')



def plot_images(data_train):
    """
    Plots a grid of images with their corresponding labels and additional information.

    Parameters:
    data_train (list): A list of tuples where each tuple contains an image, a label, and additional information.
                       The image should be in a format compatible with plt.imshow() and the label and additional 
                       information should be strings.

    The function displays a grid of up to 70 images (or fewer if data_train contains less than 70 images).
    Each image is displayed without axis ticks and grid, and the label and additional information are shown 
    below each image in white color.
    """
    plt.figure(figsize=(10,10))
    m = 70
    if( len(data_train) < 70 ):
        m = len(data_train)
    for i in range(m):
        plt.subplot(7,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data_train[i][0], cmap='gray')
        plt.xlabel("{}\n({})".format( data_train[i][1] , data_train[i][2] ), color='black')    
    plt.show()


def create_seed(num_examples_to_generate, noise_dim, num_classes):
    """
    Generates a seed tensor by combining random noise and one-hot encoded class labels.
    Args:
        num_examples_to_generate (int): The number of examples to generate.
        noise_dim (int): The dimensionality of the noise vector.
        num_classes (int): The number of classes for one-hot encoding.
    Returns:
        tuple: A tuple containing:
            - seed (tf.Tensor): A tensor of shape (num_examples_to_generate, noise_dim + num_classes) 
              containing the concatenated noise and one-hot encoded labels.
            - random_classes (tf.Tensor): A tensor of shape (num_examples_to_generate,) containing 
              the randomly generated class labels.
    """

    # Gera ruído normal de tamanho (num_examples_to_generate, noise_dim)
    noise = tf.random.normal([num_examples_to_generate, noise_dim])

    # Gera rótulos aleatórios entre 0 e (num_classes - 1)
    random_classes = tf.random.uniform([num_examples_to_generate], minval=0, maxval=num_classes, dtype=tf.int32)

    # Converte os rótulos para one-hot encoding
    one_hot_labels = tf.one_hot(random_classes, depth=num_classes)

    # Concatena o ruído com os rótulos one-hot
    seed = tf.concat([noise, one_hot_labels], axis=1)

    return seed, random_classes  # Retorna o tensor gerado e os rótulos para referência


def generate_and_save_images(model, epoch, test_input, outpath):
    """
    Generates and saves images produced by the model.
    Args:
        model (tf.keras.Model): The trained model used to generate images.
        epoch (int): The current epoch number, used for naming the output file.
        test_input (tf.Tensor): The input tensor for the model to generate images from.
        outpath (str): The directory path where the generated images will be saved.
    Returns:
        None
    """

    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        #plt.title(f'Classe {test_input[i][100]} {test_input[i][101]}', fontsize=6)
        plt.title(f'Classe {np.argmax(test_input[i][-10:])}', fontsize=6)

    # Ajusta o espaçamento entre os subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    plt.savefig(outpath+'image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()