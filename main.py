import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import os
from GAN_network import GenAI_network

if __name__ == '__main__':

    network = GenAI_network()
    images, labels, class_names = network.load_data()
    network.viz_data()
    a = network.shuffle()
    print()


