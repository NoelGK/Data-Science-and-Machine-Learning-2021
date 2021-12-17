import os
import numpy as np
import soundfile


"""
1.- Loads audio file
2.- Adds white noise to it
3.- Saves corrupted audio to chosen directory
"""


class AudioSaver:
    def __init__(self, save_dir, sample_rate=22050):
        self.save_dir = save_dir
        self.sample_rate = sample_rate

    def save(self, file_path, audio):
        save_path = self._generate_save_path(file_path)
        soundfile.write(save_path, audio, self.sample_rate)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]+'_corrupted.wav'
        save_path = os.path.join(self.save_dir, file_name)
        return save_path


class Noise:
    def __init__(self):
        self.loader = None
        self.audio_saver = None

    def add_noise(self, audio_files_dir, display=False):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._noisify(file_path)
                if display:
                    print(f'Added noise to file {file}')

    @staticmethod
    def _white_noise(amplitude, samples, sample_rate):
        time = np.linspace(0, samples / sample_rate, samples)
        noise = np.zeros(samples)
        freqs = np.linspace(0, 20000, 1000)

        for f in freqs:
            noise += 1 / len(freqs) * amplitude * np.random.randn() * np.sin(2.0 * np.pi * f * time)

        return noise

    def _noisify(self, file_path):
        audio = self.loader.load(file_path)
        n_samples = len(audio)
        amplitude = 0.1 * np.random.randn()
        corrupted_audio = audio + self._white_noise(amplitude, n_samples, self.loader.sample_rate)
        self.audio_saver.save(file_path, corrupted_audio)
