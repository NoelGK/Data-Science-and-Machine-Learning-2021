import os
import music21 as m21


class SongLoader:
    def __init__(self, input_format):
        self.format = input_format
        self.songs = []
        self.names = []
        self.acceptable_durations = [
            0.25, 0.5, m21.duration.Duration(1/3).quarterLength, 0.75, 1, 1.5, 2, 3, 4
        ]

    def load(self, dataset_path):
        for path, subdir, files in os.walk(dataset_path):
            for file in files:
                if self._format_filter(file):
                    file_path = os.path.join(path, file)
                    song = m21.converter.parse(file_path)
                    self.songs.append(song)
                    self.names.append(file)
        self._filter_by_durations()
        return self.songs, self.names

    def _filter_by_durations(self):
        for song, name in zip(self.songs, self.names):
            if not self._is_acceptable(song):
                self.songs.remove(song)
                self.names.remove(name)

    def _format_filter(self, file):
        format_length = len(self.format)
        if file[-format_length:] == self.format:
            return True

        return False

    def _is_acceptable(self, song):
        for note in song.flat.notesAndRests:
            if note.duration.quarterLength not in self.acceptable_durations:
                return False

        return True


if __name__ == '__main__':
    loader = SongLoader(input_format='krn')
    sgs, nms = loader.load('songs')
    print(sgs[0])
