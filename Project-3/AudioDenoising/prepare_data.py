from preprocess import PreprocessingPipeline, Loader, Saver, \
    Padder, MinMaxNormalizer, LogSpectrogramExtractor


"""
In this file: 
    - Add white noise to audio files in directory
    - Convert original audio to spectrogram
    - Convert corrupted audio to spectrogram
"""


if __name__ == '__main__':
    # Parameters for adding noise
    clean_load_dir = 'C:/Users/noelg/Datasets/FSDD/clean_recordings/'
    noise_save_dir = 'C:/Users/noelg/Datasets/FSDD/noisy_recordings/'
    SR = 22050
    DUR = 0.74

    # noise = Noise()
    # noise.loader = Loader(SR, DUR, mono=True)
    # noise.audio_saver = AudioSaver(noise_save_dir)
    #
    # noise.add_noise(clean_load_dir, display=True)

    # Parameters for storing spectrograms. Clean spectrograms done in preprocess.py
    noise_load_dir = noise_save_dir
    noisy_spectro_save_dir = 'C:/Users/noelg/Datasets/FSDD/noisy_spectrograms/'
    noisy_min_max_save_dir = 'C:/Users/noelg/Datasets/FSDD/noisy_min_max/'
    FS = 512
    HP = 256

    pipeline = PreprocessingPipeline()
    pipeline.loader = Loader(SR, DUR, mono=True)
    pipeline.padder = Padder()
    pipeline.normalizer = MinMaxNormalizer(0.0, 1.0)
    pipeline.extractor = LogSpectrogramExtractor(FS, HP)
    pipeline.saver = Saver(noisy_spectro_save_dir, noisy_min_max_save_dir)

    pipeline.process(noise_load_dir, display=True)
