import pickle
import numpy as np
import librosa
from preprocess import MinMaxNormalizer


class SoundGenerator:
    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normalizer = MinMaxNormalizer(0, 1)

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representation = self.vae.reconstruct(spectrograms)
        signals = self.convert_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representation

    def noisy_spec_to_audio(self, corrupted_spectrograms, min_max_values):
        """Takes a corrupted spectrogram and converts it to audio"""
        clean_spectrograms = self.vae.denoise(corrupted_spectrograms)
        signals = self.convert_to_audio(clean_spectrograms, min_max_values)
        return signals

    def convert_to_audio(self, spectrograms, min_max_values):
        """Converts spectrograms to signal through ISTFT."""
        signals = []
        for spectrogram, min_max in zip(spectrograms, min_max_values):
            log_spectrogram = spectrogram[:, :, 0]
            denorm_spectrogram = self._min_max_normalizer.denormalize(
                log_spectrogram, min_max[0], min_max[1]
            )
            amplitude_spectrogram = librosa.db_to_amplitude(denorm_spectrogram)
            signal = librosa.istft(amplitude_spectrogram, hop_length=self.hop_length)
            signals.append(signal)

        return signals


if __name__ == '__main__':
    HL = 256
    audio_save_dir = 'C:/Users/noelg/Datasets/FSDD/reconstructed_audio/'
    min_max_values_path = 'C:/Users/noelg/Datasets/FSDD/noisy_min_max/min_max_values.pkl'

    with open(min_max_values_path, 'rb') as f:
        min_max_vals = pickle.load(f)

    min_maxes = []
    for key, val in min_max_vals.items():
        min_maxes.append(np.array([val['min'], val['max']]))

    min_maxes = np.array(min_maxes)
    print(min_maxes.shape)
