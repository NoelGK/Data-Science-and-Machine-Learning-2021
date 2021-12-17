from autoencoder import Autoencoder
from tensorflow.keras.datasets import mnist
from extra_funcs import plot_latent_representation, select_images


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')/255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    y_train = y_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape + (1,))
    y_test = y_test.astype('float32') / 255
    y_test = y_test.reshape(y_test.shape + (1,))

    return x_train, y_train, x_test, y_test


LR = 0.0005
BS = 16
EPS = 10


def train(x_train, rate, b_size, epochs):
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(16, 32, 32, 16),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
    autoencoder.compile(rate)
    autoencoder.train(x_train, b_size, epochs)
    return autoencoder


if __name__ == '__main__':
    x_tr, y_tr, x_te, y_te = load_mnist()
    ae = train(x_tr[:2000], LR, BS, EPS)

    selected_images, selected_labels = select_images(x_te, y_te, num_images=600)
    _, latent_representations = ae.reconstruct(selected_images)
    plot_latent_representation(latent_representations, selected_labels)
    print(y_tr.shape)
