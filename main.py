import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
from GAN_network import GenAI_network

if __name__ == '__main__':

    network = GenAI_network()
    images, labels, class_names = network.load_data()
    # network.viz_data()

    generator = network.build_generator()
    im = generator.predict(np.random.randn(6, 128))
    network.plot_generated_image(im)

    discriminator = network.build_discriminator()
    discriminator.predict(im)
    print()


