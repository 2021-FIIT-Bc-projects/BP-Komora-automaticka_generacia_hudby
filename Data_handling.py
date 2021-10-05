from music21 import *
import numpy as np
import os


### NOTES + QUARTER LENGTHS
def load_midi(midi_dir) -> np.array:
    """Function, which returns np.array of sequences of notes"""

    note_sequences = []

    for file in os.listdir(midi_dir):

        try:
            if file.endswith(".mid"):

                midi = converter.parse(midi_dir + "\\" + file)
                partitioned_midi = instrument.partitionByInstrument(midi)
                note_seq = []

                # Looping over all the instruments
                for part in partitioned_midi.parts:

                    notes_to_parse = part.recurse()

                    # Looping over elements of song
                    for element in notes_to_parse:

                        if str(element.quarterLength) == '1/3':
                            print(element.fullName)

                        # note
                        if isinstance(element, note.Note):
                            note_seq.append(str(element.pitch) + " " + str(element.quarterLength))

                        # chord
                        elif isinstance(element, chord.Chord):
                            note_seq.append(
                                '.'.join(str(n) for n in element.normalOrder) + " " + str(element.quarterLength))

                        # rest
                        elif isinstance(element, note.Rest):
                            note_seq.append('Rest' + " " + str(element.quarterLength))

                note_sequences.append(note_seq)
        except Exception:
            print("Error - Failed to load data from " + str(file))

    return np.array(note_sequences, dtype=object)


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


def normalize_input(input_list) -> np.array:
    """convert input np.array from note sequences to int sequences"""

    raveled_input_list = list(set(input_list.ravel()))
    note_to_int = dict((note_, number) for number, note_ in enumerate(raveled_input_list))

    input_seq = []
    for i in input_list:
        temp = []
        for j in i:
            # assigning unique integer to every note
            temp.append(note_to_int[j])

        input_seq.append(temp)

    input_seq = np.array(input_seq)
    return input_seq


def normalize_output(output_list):
    """convert output np.array from notes to ints"""

    unique_notes = list(set(output_list))
    note_to_int = dict((note_, number) for number, note_ in enumerate(unique_notes))
    output_seq = []

    for i in output_list:
        output_seq.append(note_to_int[i])

    output_seq = np.array(output_seq)
    return output_seq


def reshapeX(x):
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    return x


def generate_midi(prediction_output):
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:

        element, quarterLength = pattern.split(' ')

        if '/' in quarterLength:
            x, y = '1/3'.split('/')
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
                cn = int(current_note)

                new_note = note.Note(cn, quarterLength=quarterLength)
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
    filename = 'output_midi\\music.mid'
    midi_stream.write('midi', fp=filename)
