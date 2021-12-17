import numpy as np
from tensorflow.keras.utils import to_categorical
import music21 as m21


class MelodyGenerator:
    def __init__(self, model, mapping, seq_length):
        self.model = model
        self.mapping = mapping
        self._start_symbols = ['/'] * seq_length

    def generate(self, seed, steps, max_seq_length, temperature):
        """
        Generates a melody from a seed sequence.
        :param seed: string to build melody from
        :param steps: number of steps to produce
        :param max_seq_length: maximum length to be taken when prediction new step
        :param temperature: stands for how explorative the generation is. High temperature means more randomly selected
        values from the probability distribution given by the model. Low temperature means strict select from the prob.
        distribution.
        :return: melody sequence of length seed + steps
        """
        seed = seed.split()  # conver string to list
        melody = seed
        seed = self._start_symbols + seed

        seed = [self.mapping[symbol] for symbol in seed]  # map seed to int

        for _ in range(steps):
            previous = to_categorical(
                seed[-max_seq_length:], num_classes=len(self.mapping)
            )  # selects the max length taken

            previous = previous[np.newaxis, ...]

            probabilities = self.model.predict(previous)[0]
            output_int = self.sample_with_temperature(probabilities, temperature)
            seed.append(output_int)

            output_symbol = [k for k, v in self.mapping.items() if v == output_int][0]

            if output_symbol == '/':
                break

            melody.append(output_symbol)

        return melody

    @staticmethod
    def save_melody(melody, step_duration=0.25, save_format='midi', file_name='melody.mid'):
        stream = m21.stream.Stream()

        # Parse the symbols in the melody into m21 notes
        # 58 _ _ _ 54 _ 55 _ 58 _ _ _ r _ _ _
        start_symbol = None  # Either midi pitch or a rest
        step_counter = 1  # counts the number off sixteenth notes it lasts.

        for i, symbol in enumerate(melody):
            if symbol != '_' or i+1 == len(melody):
                if start_symbol is not None:
                    quarter_length = step_duration * step_counter

                    # Rests
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength=quarter_length)

                    # Notes
                    else:
                        m21_event = m21.note.Note(pitch=int(start_symbol), quarterLength=quarter_length)

                    stream.append(m21_event)
                    step_counter = 1

                start_symbol = symbol

            else:
                step_counter += 1

        stream.write(save_format, file_name)

    @staticmethod
    def sample_with_temperature(probabilities, temperature):
        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        indxs = range(len(probabilities))
        chosen_indx = np.random.choice(indxs, p=probabilities)

        return chosen_indx
