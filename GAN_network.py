import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import os

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