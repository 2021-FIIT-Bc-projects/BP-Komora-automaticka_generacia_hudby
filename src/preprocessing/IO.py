import os
import numpy as np
from music21 import *


def load_midi(midi_dir, withLengths=False, withRests=True, nameFilter='', instrumentFilter='') -> np.array:
    """Function, which returns np.array of sequences of notes"""

    note_sequences = []

    for file in os.listdir(midi_dir):

        if file.endswith(".mid") and nameFilter in file:
            try:
                midi = converter.parse(midi_dir + "\\" + file)
            except Exception:
                print("File " + str(file) + " failed to load")

            partitioned_midi = instrument.partitionByInstrument(midi)
            note_seq = []

            # Looping over all the instruments
            for part in partitioned_midi.parts:

                if instrumentFilter not in str(part):
                    continue

                notes_to_parse = part.recurse()

                # Looping over elements of song
                for element in notes_to_parse:

                    # note
                    if isinstance(element, note.Note):
                        # print("element: " + str(element.pitch) + " => " + str(element.pitch.ps))
                        if withLengths:
                            note_seq.append(str(element.pitch) + " " + str(element.quarterLength))
                        else:
                            note_seq.append(str(element.pitch))

                    # chord
                    elif isinstance(element, chord.Chord):
                        if withLengths:
                            note_seq.append('.'.join(str(n) for n in element.normalOrder) + " " + str(element.quarterLength))
                        else:
                            note_seq.append('.'.join(str(n) for n in element.normalOrder))

                    # rest
                    elif isinstance(element, note.Rest) and withRests:
                        if withLengths:
                            note_seq.append('Rest' + " " + str(element.quarterLength))
                        else:
                            note_seq.append('Rest')

            note_sequences.append(note_seq)

    return np.array(note_sequences, dtype=object)