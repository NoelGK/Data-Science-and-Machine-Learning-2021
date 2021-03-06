import os
import pickle
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


def _create_new_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


class Autoencoder:
    """
    Deep Convolutional Autoencoder with mirrored encoder and decoder.
    """

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape = input_shape  # [width, height, channels]
        self.conv_filters = conv_filters  # [number of filters of each convolutional layer]
        self.conv_kernels = conv_kernels  # [kernel size at each layer]
        self.conv_strides = conv_strides  # [stride for each layer]
        self.latent_space_dim = latent_space_dim  # number of axis of the latent space

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_kernels)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()  # This method will build encoder and decoder

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(
            x_train,
            x_train,
            batch_size=batch_size,
            epochs=num_epochs,
            shuffle=True
        )

    def denoise_train(self, x_train, y_train, batch_size, num_epochs):
        self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            shuffle=True
        )

    def save(self, save_folder='.'):
        _create_new_folder(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        latent_representation = self.encoder.predict(images)
        reconstruction = self.decoder.predict(latent_representation)
        return reconstruction, latent_representation

    def denoise(self, images):
        clean_images = self.model.predict(images)
        return clean_images

    @classmethod
    def load(cls, save_folder='.'):
        parameters_path = os.path.join(save_folder, 'parameters.pkl')
        weights_path = os.path.join(save_folder, 'weights.h5')

        with open(parameters_path, 'rb') as f:
            parameters = pickle.load(f)

        autoencoder = Autoencoder(*parameters)
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name='autoencoder')

    """BUILD ENCODER"""
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name='encoder')

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name='encoder_input')

    def _add_conv_layers(self, encoder_input):
        """Creates all convolutional blocks in encoder"""
        x = encoder_input
        for layer_indx in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_indx, x)

        return x

    def _add_conv_layer(self, layer_indx, x):
        """
        Adds a convolutional block to a graph of layers.
        block = 2D-convolution + ReLU + batch normalization
        """
        lay_num = layer_indx + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_indx],
            kernel_size=self.conv_kernels[layer_indx],
            strides=self.conv_strides[layer_indx],
            padding='same',
            name=f'encoder_conv_layer_{lay_num}'
        )

        x = conv_layer(x)
        x = ReLU(name=f'encoder_relu_{lay_num}')(x)
        x = BatchNormalization(name=f'encoder_bn_{lay_num}')(x)

        return x

    def _add_bottleneck(self, x):
        """Flatten data and add a bottleneck (dense layer).
        K.int_shape() returns [batch_size, width, height, channels] so we slice it."""
        self._shape_before_bottleneck = K.int_shape(x)[1:]  # [width, height, channels]
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name='encoder_output')(x)
        return x

    """BUILD DECODER"""
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name='decoder')

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name='decoder_input')

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name='decoder_dense')(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Adds blocks of convolutional transposed layers. It loops through all the convolutional
        layers in reverse and stops at the first layer."""
        for layer_indx in reversed(range(self._num_conv_layers)[1:]):
            x = self._add_conv_transpose_layer(layer_indx, x)

        return x

    def _add_conv_transpose_layer(self, layer_indx, x):
        num_layer = self._num_conv_layers - layer_indx
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_indx],
            kernel_size=self.conv_kernels[layer_indx],
            strides=self.conv_strides[layer_indx],
            padding='same',
            name=f'decoder_conv_transpose_layer_{num_layer}'
        )

        x = conv_transpose_layer(x)
        x = ReLU(name=f'decoder_relu_{num_layer}')(x)
        x = BatchNormalization(name=f'batch_normalization_{num_layer}')(x)

        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,  # [height, width, channel]
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding='same',
            name=f'decoder_conv_transpose_layer_{self._num_conv_layers}'
        )

        x = conv_transpose_layer(x)
        output_layer = Activation('sigmoid', name='decoder_output')(x)

        return output_layer

    """SAVE MODEL"""
    def _save_parameters(self, folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(folder, 'parameters.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(parameters, f)

    def _save_weights(self, folder):
        save_path = os.path.join(folder, 'weights.h5')
        self.model.save_weights(save_path)
