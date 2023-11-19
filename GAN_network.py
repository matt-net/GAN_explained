import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import os
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
import numpy as np


class GenAI_network:
    def __init__(self, data_path="archive/fashion-mnist_train.csv", latent_dim=100):
        self.data_path = data_path
        self.latent_dim = latent_dim
        self.images, self.labels, self.class_names = self.load_data()
        self.batch = 128

    def load_data(self):
        # Construct the full path to the dataset file
        dataset_path = os.path.join(os.getcwd(), self.data_path)

        # Read the CSV file containing the Fashion-MNIST data
        df = pd.read_csv(dataset_path)

        # Extract labels and pixel values from the DataFrame
        self.labels = df.iloc[:, 0]  # First column contains the class labels
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                            'Ankle boot']

        # Reshape pixel values to 28x28 images
        self.images = df.iloc[:, 1:].values.reshape(-1, 28, 28)

        return self.images, self.labels, self.class_names

    def viz_data(self):
        current_directory = os.getcwd()

        plots_directory = os.path.join(current_directory, 'plots')
        os.makedirs(plots_directory, exist_ok=True)

        visualizations_directory = os.path.join(plots_directory, 'visualisations')
        os.makedirs(visualizations_directory, exist_ok=True)

        num_rows = 2
        num_cols = 3

        # Visualize six images in a 2x3 grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 7))

        for i in range(num_rows):
            for j in range(num_cols):
                index = i * num_cols + j
                axes[i, j].imshow(self.images[index], cmap='gray')
                axes[i, j].set_title(f'Label: {self.labels[index]}\nClass: {self.class_names[self.labels[index]]}')
                axes[i, j].axis('off')  # Turn off axis labels for better visualization

        # Save the plot in the 'visualisations' directory
        plot_filename = os.path.join(visualizations_directory, 'fashion_mnist_visualization.png')
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.show()

        print(f"Plot saved at: {plot_filename}")

    def scale_image(self):
        return self.images / 255

    def preprocess(self):
        # Assuming images is a NumPy array of shape (num_images, height, width)
        self.images = self.scale_image()
        num_images = self.images.shape[0]

        # Reshape the images to (num_images, height, width, channels) for compatibility
        images = self.images.reshape((num_images, 28, 28, 1))

        # Create a dataset from the images
        dataset = tf.data.Dataset.from_tensor_slices(images)

        # Shuffle the dataset
        self.dataset = dataset.shuffle(buffer_size=num_images)
        self.batched = dataset.batch(self.batch)

        return dataset, self.batched

    def build_generator(self):
        model = Sequential()
        # Takes in random values and reshapes it to 7x7x128
        model.add(Dense(7 * 7 * 128, input_dim=128))
        model.add(LeakyReLU(0.2))
        model.add(Reshape((7, 7, 128)))

        # Up sampling block 1
        model.add(UpSampling2D())  # Double it
        model.add(Conv2D(128, 5, padding='same'))
        model.add(LeakyReLU(0.2))

        # Up sampling block 2
        model.add(UpSampling2D())  # Double it
        model.add(Conv2D(128, 5, padding='same'))
        model.add(LeakyReLU(0.2))

        # Down sampling block 1
        model.add(Conv2D(128, 4, padding='same'))
        model.add(LeakyReLU(0.2))

        # Down sampling block 2
        model.add(Conv2D(128, 4, padding='same'))
        model.add(LeakyReLU(0.2))

        # Conv layer to get one channel
        model.add(Conv2D(1, 4, padding='same', activation='sigmoid'))

        # model.summary()
        return model

    def build_discriminator(self):
        model = Sequential()

        # First Conv Block
        model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        # Second Conv Block
        model.add(Conv2D(64, 5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        # Third Conv Block
        model.add(Conv2D(128, 5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        # Third Conv Block
        model.add(Conv2D(256, 5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))

        # Flatten and then pass to dense layer
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))  # : False, 0 : True1

        model.summary()

        return model

    def plot_generated_image(self, gen_image):
        current_directory = os.getcwd()

        plots_directory = os.path.join(current_directory, 'plots')
        os.makedirs(plots_directory, exist_ok=True)

        visualizations_directory = os.path.join(plots_directory, 'visualisations')
        os.makedirs(visualizations_directory, exist_ok=True)
        num_rows = 2
        num_cols = 3

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 7))

        for i in range(num_rows):
            for j in range(num_cols):
                index = i * num_cols + j
                axes[i, j].imshow(gen_image[index], cmap='gray')
                axes[i, j].axis('off')  # Turn off axis labels for better visualization

        # Save the plot in the 'visualisations' directory
        plot_filename = os.path.join(visualizations_directory, 'fashion_mnist_generated.png')
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.show()
