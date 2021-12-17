from vae import VAE
from train_test_split import x_train, y_train, x_test, y_test
from tensorflow.keras.losses import MeanSquaredError


# Parameters for training
LR = 0.0005
BS = 32
EPS = 20

# Save folder
vae_save_dir = 'C:/Users/noelg/models/denoising_vae/'

if __name__ == '__main__':
    vae = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(16, 8),
        conv_kernels=(3, 3),
        conv_strides=(2, 2),
        latent_space_dim=16
    )

    vae.summary()
    vae.compile(learning_rate=0.0005)
    vae.denoise_train(x_train, y_train, batch_size=BS, num_epochs=EPS)

    vae.save(vae_save_dir)
