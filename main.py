import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import os
from GAN_network import GenAI_network


if __name__ == '__main__':

    network = GenAI_network()
    images,labels, class_names = network.load_data()

    # Get the current working directory
    current_directory = os.getcwd()
    #
    # # Create a directory named 'plots' if it doesn't exist
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
            axes[i, j].imshow(images[index], cmap='gray')
            axes[i, j].set_title(f'Label: {labels[index]}\nClass: {class_names[labels[index]]}')
            axes[i, j].axis('off')  # Turn off axis labels for better visualization

    # Save the plot in the 'visualisations' directory
    plot_filename = os.path.join(visualizations_directory, 'fashion_mnist_visualization.png')
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.show()

    print(f"Plot saved at: {plot_filename}")