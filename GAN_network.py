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


    def load_data(self):
        # Construct the full path to the dataset file
        dataset_path = os.path.join(os.getcwd(), self.data_path)

        # Read the CSV file containing the Fashion-MNIST data
        df = pd.read_csv(dataset_path)

        # Extract labels and pixel values from the DataFrame
        labels = df.iloc[:, 0]  # First column contains the class labels
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Reshape pixel values to 28x28 images
        images = df.iloc[:, 1:].values.reshape(-1, 28, 28)

        return images, labels, class_names

    # def viz_data(self):

