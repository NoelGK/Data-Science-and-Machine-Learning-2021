"""
1.- Load file
2.- Pad the signal if necessary
3.- extract log spectrogram from signal
4.- Normalize spectrogram
5.- Save normalized espectrogram
"""
import os
import librosa
import pickle
import numpy as np


class Loader:
    """
    Responsible for loading audio file.
    """
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        """librosa returns a tuple of 2 items: the actual signal and the sample rate."""
        signal = librosa.load(
            file_path,
            sr=self.sample_rate,
            duration=self.duration,
            mono=self.mono
        )[0]
        return signal


class Padder:
    """
    Padder is responsible to apply padding to an array.
    """
    def __init__(self, mode='constant'):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)
        return padded_array


class LogSpectrogramExtractor:
    """
    Gets the log spectrogram in dB from a time series signal.
    """
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract_spectrogram(self, signal):
        # librosa stft returns an array of dimension (1+frame_size/2, _), but we want the dim to be an even number
        stft = librosa.stft(
            signal,
            n_fft=self.frame_size,
            hop_length=self.hop_length
        )[:-1]
        spectrogram = np.absolute(stft)
        log_spectrogram = librosa.power_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormalizer:
    """
    Applies MinMax Normalization to an array.
    """
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalize(self, array):
        norm_array = (array - array.min())/(array.max() - array.min())  # Maps into [0, 1]
        norm_array = norm_array * (self.max - self.min) + self.min  # Maps into [self.min, self.max]
        return norm_array

    def denormalize(self, norm_array, original_min, original_max):
        """Inverts the normalize method."""
        array = (norm_array - self.min)/(self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class Saver:
    """
    Responsible of saving features and the min/max values
    """
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, 'min_max_values.pkl')
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name+'.npy')
        return save_path


class PreprocessingPipeline:
    """
    Preprocess audio files in a directory applying the following steps to each file:
        1.- Load file
        2.- Pad the signal if necessary
        3.- extract log spectrogram from signal
        4.- Normalize spectrogram
        5.- Save normalized espectrogram
        6.- Stores the min and max original values for each spectrogram.
    """
    def __init__(self):
        self._loader = None
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.min_max_values = {}
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir, display=False):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                if display:
                    print(f'Processed file {file_path}')
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)

        feature = self.extractor.extract_spectrogram(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_values(save_path, feature.min(), feature.max())

    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def _store_min_max_values(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            'min': min_val,
            'max': max_val
        }


if __name__ == '__main__':
    # Parameters for preprocessing
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.55
    SAMPLE_RATE = 22050
    MONO = True

    """ORIGINAL RECORDINGS"""
    # Directories for loading/saving data
    CLEAN_SPECTROGRAM_SAVE_DIR = 'C:/Users/noelg/Datasets/FSDD/clean_spectrograms/'
    CLEAN_MIN_MAX_VALUES_DIR = 'C:/Users/noelg/Datasets/FSDD/clean_min_max/'
    CLEAN_REC_DIR = 'C:/Users/noelg/Datasets/FSDD/clean_recordings/'

    # Instantiation of all objects
    loader_ = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectro_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_norm = MinMaxNormalizer(0.0, 1.0)
    saver = Saver(CLEAN_SPECTROGRAM_SAVE_DIR, CLEAN_MIN_MAX_VALUES_DIR)

    preprocess_pipeline = PreprocessingPipeline()
    preprocess_pipeline.loader = loader_
    preprocess_pipeline.padder = padder
    preprocess_pipeline.extractor = log_spectro_extractor
    preprocess_pipeline.normalizer = min_max_norm
    preprocess_pipeline.saver = saver

    preprocess_pipeline.process(CLEAN_REC_DIR)

    """CORRUPTED RECORDINGS"""
    CORR_SPECTROGRAMS_DIR = 'C:/Users/noelg/Datasets/FSDD/noisy_spectrograms/'
    CORR_MIN_MAX_DIR = 'C:/Users/noelg/Datasets/FSDD/noisy_min_max'
    CORR_REC_DIR = 'C:/Users/noelg/Datasets/FSDD/noisy_recordings'

    saver2 = Saver(CORR_SPECTROGRAMS_DIR, CORR_MIN_MAX_DIR)
    preprocess_pipeline.saver = saver2
    preprocess_pipeline.process(CORR_REC_DIR)
