'''
Artificial Intelligence Model
'''
from __future__ import print_function
from __future__ import absolute_import
import os
import numpy as np

import keras 
import tensorflow as tf
import keras.backend as K

from keras.models import Model 
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, Add
from keras.optimizers import Adam
from keras import regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only error will be shown

class NeuralNetwork:
    def __init__(self, 
        input_shape, output_dim, network_structure, 
        learning_rate=1e-3, 
        l2_const=1e-4, 
        verbose=False
    ):
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.network_structure = network_structure

        self.learning_rate = learning_rate
        self.l2_const = l2_const
        self.verbose = verbose

        self.model = self.build_model()

    def build_model(self):
        state_tensor = Input(shape=self.input_shape)


    def __conv_block(self, x, filters, kernel_size=3):

    def __res_block(self, x, filters, kernel_size=3):

    def __action_value_block(self, x):

    def predict(self, state):

    def fit(self, dataset, epochs, batch_size):

    def update(self, dataset):

    def load_model(self, filename):

    def save_model(self, filename):

    def plot_model(self, filename):


class AI:
    def __init__(self, state_shape, action_dim, verbose=False):

    def get_state_shape(self):

    def train(self, dataset, epochs, batch_size):

    def update(self, dataset):

    def play(self):

    def load_nnet(self, filename):

    def save_nnet(self, filename):

    def plot_nnet(self, filename):