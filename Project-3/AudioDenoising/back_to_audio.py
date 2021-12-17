import os
from train_test_split import x_test, x_test_min_max_vals
import soundfile as sf
from sound_generator import SoundGenerator
from new_vae import VAE


model_dir = 'C:/Users/noelg/models/denoising_vae/lr0003_b32_e100/'
recovered_dir = 'C:/Users/noelg/Datasets/FSDD/reconstructed_audio/'


def save_signals(signals, saving_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        file_name = os.path.join(saving_dir, f'denoised_{i}.wav')
        sf.write(file_name, signal, sample_rate)


if __name__ == '__main__':
    model = VAE.load(model_dir)
    prediction = model.vae.predict(x_test)

    generator = SoundGenerator(model, hop_length=256)
    audios = generator.convert_to_audio(prediction[:4], x_test_min_max_vals[:4])
    save_signals(audios, recovered_dir)
