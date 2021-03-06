from collections import Counter
from music21 import *
import numpy as np
import os

"""V tomto subore sa venujem praci s datami"""


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


def remove_rare(data, threshold=50):

    notes = [element for notes in data for element in notes]
    freq = dict(Counter(notes))
    frequent_notes = [note_ for note_, count in freq.items() if count >= threshold]

    new_data = []

    for song in data:
        temp = []
        for note in song:
            if note in frequent_notes:
                temp.append(note)
        new_data.append(temp)

    return np.array(new_data, dtype=object)


def prepare_data(data, input_sequence_length) -> (np.array, np.array):
    """function, which generates input_sequences and output_notes"""

    input_list = []
    output_list = []

    for sequence in data:

        for i in range(0, len(sequence) - input_sequence_length):
            # preparing input and output sequences

            input_sequence = sequence[i:i + input_sequence_length]
            output_note = sequence[i + input_sequence_length]

            input_list.append(input_sequence)
            output_list.append(output_note)

    input_list = np.array(input_list)
    output_list = np.array(output_list)

    return input_list, output_list


def normalize_X(input_list, note_to_int) -> np.array:
    """convert input np.array from note sequences to int sequences"""

    input_seq = []
    for i in input_list:
        temp = []
        for j in i:
            # assigning unique integer to every note
            temp.append(note_to_int[j])

        input_seq.append(temp)

    input_seq = np.array(input_seq)
    return input_seq


def normalize_y(output_list, note_to_int):
    """convert output np.array from notes to ints"""

    output_seq = []

    for i in output_list:
        output_seq.append(note_to_int[i])

    output_seq = np.array(output_seq)
    return output_seq


def reshape_X(x):
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    return x


def generate_midi(prediction_output, filename='music.mid'):
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:

        if ' ' in pattern:
            element, quarterLength = pattern.split(' ')
        else:
            element = pattern
            quarterLength = '1.0'

        if '/' in quarterLength:
            x, y = quarterLength.split('/')
            quarterLength = float(x)/float(y)
        else:
            quarterLength = float(quarterLength)

        # element is a rest
        if element == 'Rest':
            new_note = note.Rest(quarterLength=quarterLength)
            output_notes.append(new_note)

        # element is a chord
        elif ('.' in element) or element.isdigit():
            notes_in_chord = element.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note), quarterLength=quarterLength)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        # element is a note
        else:
            new_note = note.Note(element, quarterLength=quarterLength)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 1

    midi_stream = stream.Stream(output_notes)
    folder = 'output_midi\\'
    path = folder + filename
    midi_stream.write('midi', fp=path)
