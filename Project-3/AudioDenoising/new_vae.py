import os
import pickle
from abc import ABC
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, ReLU, BatchNormalization, \
    Flatten, Conv2DTranspose, Activation, Reshape
import keras.metrics
from tensorflow.keras.losses import mean_squared_error


class NormalSample(layers.Layer):
    """Generates multivariate gaussian distribution sample from latent space"""

    def call(self, inputs, **kwargs):
        z_mean, z_log_variance = inputs
        epsilon = K.random_normal(shape=tf.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_variance) * epsilon


class Encoder:
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_dim):
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim

        self._num_conv_layers = len(conv_strides)

        self.shape_before_bottleneck = None

        self.mu = None
        self.log_variance = None

        self.encoder = None
        self.params = [
            input_shape, conv_filters, conv_kernels, conv_strides, latent_dim
        ]

        self.encoder_input = None

        self._build_encoder()

    def summary(self):
        self.encoder.summary()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)

        self.encoder_input = encoder_input
        self.encoder = Model(encoder_input, [bottleneck, self.mu, self.log_variance], name='Encoder')

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
        """Flatten data and add a bottleneck with gaussian multivariate sampling.
        K.int_shape() returns [batch_size, width, height, channels] so we slice it."""
        self.shape_before_bottleneck = K.int_shape(x)[1:]  # [width, height, channels]
        x = Flatten()(x)

        # self.mu, self.log_variance represent the mean and variance vectors of the multivariate distribution
        self.mu = Dense(self.latent_dim, name='mu')(x)
        self.log_variance = Dense(self.latent_dim, name='log_variance')(x)

        z = NormalSample(name='encoder_output')(
            [self.mu, self.log_variance]
        )

        return z


class Decoder:
    def __init__(self, encoder):
        self.input_shape = encoder.input_shape
        self.conv_filters = encoder.conv_filters
        self.conv_kernels = encoder.conv_kernels
        self.conv_strides = encoder.conv_strides
        self.latent_dim = encoder.latent_dim

        self._num_conv_layers = len(self.conv_strides)
        self.shape_before_bottleneck = encoder.shape_before_bottleneck

        self.decoder = None

        self._build_decoder()

    def summary(self):
        self.decoder.summary()

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)

        self.decoder = Model(decoder_input, decoder_output, name='Decoder')

    def _add_decoder_input(self):
        return Input(shape=self.latent_dim)

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self.shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name='decoder_dense')
        return dense_layer(decoder_input)

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self.shape_before_bottleneck)(dense_layer)

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


class VAE(keras.Model, ABC):
    def __init__(self, encoder, decoder, reconstruction_weight=50000, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.params = encoder.params
        self.encoder = encoder.encoder
        self.decoder = decoder.decoder

        self.model_input = encoder.encoder_input
        self.vae = None

        self._build_vae()

        self.reconstruction_weight = reconstruction_weight
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def train_step(self, data):
        """
        data = (x_data, y_data)
        """
        with tf.GradientTape() as tape:
            z_mean, z_log_variance, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = self.reconstruction_weight * tf.reduce_mean(
                tf.reduce_sum(
                    mean_squared_error(data[1], reconstruction), axis=(1, 2)
                )
            )

            kl_loss = -0.5 * (1 + z_log_variance - tf.square(z_mean) - tf.exp(z_log_variance))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def save_model_params(self, save_dir):
        params_path = os.path.join(save_dir, 'parameters.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump(self.params, f)

        weights_path = os.path.join(save_dir, 'weights.h5')
        self.vae.save_weights(weights_path)

    @classmethod
    def load(cls, model_dir):
        params_path = os.path.join(model_dir, 'parameters.pkl')
        weights_path = os.path.join(model_dir, 'weights.h5')

        with open(params_path, 'rb') as f:
            params = pickle.load(f)

        encoder = Encoder(*params)
        decoder = Decoder(encoder)
        vae = VAE(encoder, decoder)
        vae.vae.load_weights(weights_path)
        return vae

    def _build_vae(self):
        model_output = self.decoder(self.encoder(self.model_input)[2])
        self.vae = Model(self.model_input, model_output)
