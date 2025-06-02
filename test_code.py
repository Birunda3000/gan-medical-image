import tensorflow as tf

# Verify TensorFlow installation and GPU availability
def main():
    print("Hello, TensorFlow!")
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices('GPU'))

if __name__ == "__main__":
    main()