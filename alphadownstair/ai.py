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

        x = self.__conv_block(state_tensor, self.network_structure[0]['filters'], self.network_structure[0]['kernel_size'])

        if len(self.network_structure) > 1:
            for h in self.network_structure[1:]:
                x = self.__res_block(x, h['filters'], h['kernel_size'])

        action_values_tensor = self.__action_value_block(x)

        model = Model(inputs=state_tensor, outputs=action_values_tensor)
        model.compile(
            loss='mse',
            optimizer=Adam(self.learning_rate)
        )

        return model

    def __conv_block(self, x, filters, kernel_size=3):
        '''
        Convolutional layer block
        '''
        out = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)
        return out

    def __res_block(self, x, filters, kernel_size=3):
        '''
        Residual layer block
        '''
        out = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = Add()([out, x])
        out = LeakyReLU()(out)
        return out

    def __action_value_block(self, x):
        '''
        Action Values layer block
        '''
        out = Conv2D(
            filters = 64,
            kernel_size = (3,3),
            padding = 'same',
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
        )(x)
        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)

        out = Flatten()(out)
        out = Dense(
            64,
            use_bias=False,
            activation='linear',
            kernel_regularizer= regularizers.l2(self.l2_const)
		)(out)
        out = LeakyReLU()(out)

        action_values = Dense(
			self.output_dim, 
            use_bias=False,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_const),
            name = 'action'
			)(out)

        return action_values

    def predict(self, state):
        state = np.array(state)
        states = state.reshape(1, self.state_shape)
        action_values = self.model.predict(states)
        return action_values[0]

    def fit(self, states, action_values, epochs, batch_size):
        history = self.model.fit(states, action_values, epochs=epochs, batch_size=batch_size, verbose=self.verbose)
        return history

    def update(self, states, action_values):
        '''
        Update the coefficients of network one updation
        '''
        loss = self.model.train_on_batch(states, action_values)
        return loss

    def load_model(self, filename):
        self.model.load_weights(filename)

    def save_model(self, filename):
        self.model.save_weights(filename)

    def plot_model(self, filename):
        from keras.utils import plot_model
        plot_model(self.model, show_shapes=True, to_file=filename)

class AI:
    def __init__(self, state_shape, action_dim, verbose=False):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.verbose = verbose

        network_structure = list()
        network_structure.append({'filters':64, 'kernel_size':3})
        network_structure.append({'filters':64, 'kernel_size':3})
        network_structure.append({'filters':64, 'kernel_size':3})
        network_structure.append({'filters':64, 'kernel_size':3})

        self.nnet =  NeuralNetwork(
            input_shape=self.state_shape, 
            output_dim=self.action_dim, 
            network_structure=network_structure,
            verbose=self.verbose
        )

    def get_state_shape(self):
        return np.copy(self.state_shape)

    def train(self, dataset, epochs, batch_size):
        states, action_values = dataset
        history = self.nnet.fit(states, action_values, epochs=epochs, batch_size=batch_size)
        return history

    def update(self, dataset):
        states, action_values = dataset
        loss = self.nnet.update(states, action_values)
        return loss

    def evaluate_function(self, state):
        action_value = self.nnet.predict(state)
        action = np.argmax(action_value)
        return action, action_value

    def play(self, state):
        action_value = self.nnet.predict(state)
        action = np.argmax(action_value)
        return action

    def load_nnet(self, filename):
        self.nnet.load_model(filename)

    def save_nnet(self, filename):
        self.nnet.save_model(filename)

    def plot_nnet(self, filename):
        self.nnet.plot_model(filename)