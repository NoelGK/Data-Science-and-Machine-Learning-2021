import os
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def select_images(images, labels, num_images=10):
    indxs = np.arange(len(images))
    image_indx = np.random.choice(indxs, num_images, replace=False)
    selected_images = images[image_indx]
    selected_labels = labels[image_indx]

    return selected_images, selected_labels


def plot_restored_images(images, restored_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, restored_image) in enumerate(zip(images, restored_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")

        restored_image = restored_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(restored_image, cmap="gray_r")
    plt.show()


def plot_latent_representation(latent_representation, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(
        latent_representation[:, 0],
        latent_representation[:, 1],
        cmap='rainbow',
        c=sample_labels,
        alpha=0.5,
        s=2
    )
    plt.colorbar()
    plt.show()


def load_fsdd(noisy_path, clean_path):
    noisy_spectros = []
    clean_spectros = []

    for root, _, file_names in os.walk(noisy_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            noisy_spectros.append(spectrogram)

    for root, _, file_names in os.walk(clean_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            clean_spectros.append(spectrogram)

    noisy_spectros = np.array(noisy_spectros)[..., np.newaxis]
    clean_spectros = np.array(clean_spectros)[..., np.newaxis]

    return noisy_spectros, clean_spectros


def train_test_split(x_data, y_data, x_min_max_values, y_min_max_values, train_size=0.8):
    """
    Randomizes data and splits it into train and test.
    """
    indxs = np.arange(0, len(x_data))
    split_indx = int(train_size * len(x_data))
    np.random.shuffle(indxs)
    x_train, x_test = x_data[indxs][:split_indx], x_data[indxs][split_indx:]
    y_train, y_test = y_data[indxs][:split_indx], y_data[indxs][split_indx:]
    x_test_min_max_values = x_min_max_values[split_indx:]
    y_test_min_max_values = y_min_max_values[split_indx:]

    return x_train, y_train, x_test, y_test, x_test_min_max_values, y_test_min_max_values


if __name__ == '__main__':
    clean_dir = 'C:/Users/noelg/Datasets/FSDD/clean_spectrograms/0_george_0.wav.npy'
    corr_dir = 'C:/Users/noelg/Datasets/FSDD/noisy_spectrograms/0_george_0.wav_corrupted.wav.npy'

    clean = np.load(clean_dir)
    corrupted = np.load(corr_dir)

    librosa.display.specshow(corrupted, hop_length=256, x_axis='time', y_axis='log')
    plt.show()
