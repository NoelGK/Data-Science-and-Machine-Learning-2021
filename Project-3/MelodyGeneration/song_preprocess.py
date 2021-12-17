import json
import numpy as np
from tensorflow.keras.utils import to_categorical


kern_dir = 'C:/Users/noelg/Datasets/schubert/'
string_dir = 'C:/Users/noelg/Datasets/sch_string'


def load(file_path):
    with open(file_path, 'r') as fp:
        song = fp.read()
    return song


def create_mapping(songs, mapping_path=None, save=False):
    """Maps all the symbols of the string song to numbers relevant to the neural network"""
    # Get vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))  # getting all the elements without repetition

    # Create a mapping dictionary
    map_dict = {}
    for i, symbol in enumerate(vocabulary):
        map_dict[symbol] = i

    if save:
        with open(mapping_path, 'w') as f:
            json.dump(map_dict, f, indent=4)

    return map_dict


def cast_to_int(str_songs, mapping):
    int_songs = []
    str_songs = str_songs.split()
    for symbol in str_songs:
        int_songs.append(mapping[symbol])

    return int_songs


def generate_train_sequences(int_songs, sequence_length):
    """
    Generates training sequences of notes and converts them to one-hot encoding.

    :param int_songs: list of symbols representing the songs
    :param sequence_length: length of the training sequences
    :return: one-hot encoded inputs and targets
    """
    inputs = []
    targets = []

    num_training_seqs = len(int_songs) - sequence_length
    for i in range(num_training_seqs):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    categories = len(set(int_songs))
    inputs = to_categorical(inputs, num_classes=categories)
    targets = np.array(targets)

    return inputs, targets


if __name__ == '__main__':
    from song_loader import SongLoader
    from song_encoder import SongEncoder
    from transposer import Transposer

    loader = SongLoader(input_format='krn')
    encoder = SongEncoder(time_step=0.25)
    transposer = Transposer()

    foster_dir = 'C:/Users/noelg/Datasets/foster/'
    encoded_songs_dir = 'C:/Users/noelg/Datasets/encoded_foster/'
    foster, names = loader.load(foster_dir)

    songs = []
    for song in foster:
        songs.append(transposer.transpose(song))

    for song, name in zip(songs, names):
        encoded_song = encoder.encode(song)
        encoder.save_song(encoded_song, song_name=name, save_dir=encoded_songs_dir)
