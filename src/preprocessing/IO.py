import os
import json
import shutil
import datetime
import numpy as np
import pandas as pd
from music21 import *


def load_midi_input(midi_dir, withLengths=False, withRests=True, nameFilter='', instrumentFilter='') -> np.array:
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
            if partitioned_midi is None:
                continue

            for part in partitioned_midi.parts:

                print(part)

                if instrumentFilter.lower() not in str(part).lower():
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
                            note_seq.append(
                                '.'.join(str(n) for n in element.normalOrder) + " " + str(element.quarterLength))
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


def generate_output(X, y, int_to_note, no_of_timesteps, midi_dir):
    parent_dir = os.path.abspath(".")
    dirname = "output"
    path = os.path.join(parent_dir, dirname)
    os.makedirs(path)

    file_paths = ["output/X.csv", "output/y.csv", "output/notes.json"]

    X = pd.DataFrame(X)
    X.to_csv(file_paths[0])
    y = pd.DataFrame(y)
    y.to_csv(file_paths[1])

    with open(file_paths[2], 'w') as convert_file:
        convert_file.write(json.dumps(int_to_note))

    filename = "../../output/preprocessed/preprocessed(%d)_%s" % (no_of_timesteps, midi_dir)
    file_format = "zip"
    root_dir = "./output"
    shutil.make_archive(filename, file_format, root_dir)
    shutil.rmtree(path)

    print("Data were preprocessed and exported to " + os.path.abspath(filename))
