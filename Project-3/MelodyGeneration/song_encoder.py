import os.path
import music21 as m21


def _create_dir(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


class SongEncoder:
    """Encodes each note of a song by its pitch and duration, in sixteenth representation. Example:
    p = 60, d = 1.0 --> [60, "_", "_", "_"]
    """

    def __init__(self, time_step):
        self.time_step = time_step

    def encode(self, song):
        encoded_song = []

        for event in song.flat.notesAndRests:
            # Notes
            if isinstance(event, m21.note.Note):
                symbol = event.pitch.midi

            # Rests
            elif isinstance(event, m21.note.Rest):
                symbol = "r"

            else:
                raise Exception(f'{event} is not a valid character')

            steps = int(event.duration.quarterLength / self.time_step)

            for step in range(steps):
                if step == 0:
                    encoded_song.append(symbol)
                else:
                    encoded_song.append('_')

        # Cast encoded song into string
        encoded_song = " ".join(map(str, encoded_song))

        return encoded_song

    @staticmethod
    def create_string(encoded_songs, sequence_length, save_dir=None, save=False):
        delimiter = '/ ' * sequence_length
        songs = ''

        for song in encoded_songs:
            songs = songs + song + delimiter

        songs = songs[:-1]

        if save:
            with open(save_dir, 'w') as f:
                f.write(songs)

        return songs

    @staticmethod
    def save_song(encoded_song, song_name, save_dir):
        _create_dir(save_dir)
        encoded_song_path = os.path.join(save_dir, song_name+'_enocded')
        with open(encoded_song_path, 'w') as f:
            f.write(encoded_song)
