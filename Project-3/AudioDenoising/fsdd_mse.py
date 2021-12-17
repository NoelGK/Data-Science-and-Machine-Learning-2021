# from autoencoder import Autoencoder
# from tensorflow.keras.losses import MeanSquaredError
# from train_test_split import x_test, y_test
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


# model_dir = 'C:/Users/noelg/models/denoising_ae/3_64/0.00036_41_91/'
# mse = MeanSquaredError()

# ae_3_64 = Autoencoder.load(model_dir)
# prediction = ae_3_64.denoise(x_test)
# mse_3_64 = mse(y_test, prediction).numpy()
if __name__ == '__main__':
    orig = np.load('C:/Users/noelg/Datasets/FSDD/clean_spectrograms/4_nicolas_0.wav.npy')
    corr = np.load('C:/Users/noelg/Datasets/FSDD/noisy_spectrograms/4_nicolas_0.wav_corrupted.wav.npy')

    librosa.display.specshow(orig, hop_length=256, x_axis='time', y_axis='log')
    plt.show()
    librosa.display.specshow(corr, hop_length=256, x_axis='time', y_axis='log')
    plt.show()
