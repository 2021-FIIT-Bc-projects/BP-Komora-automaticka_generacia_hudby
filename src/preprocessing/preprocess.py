import sys
import getopt
from collections import Counter
import numpy as np


from IO import load_midi_input, generate_output


# Remove less frequent notes(less frequent, @threshold) from raw data
def remove_rare(data: np.array, threshold: int) -> np.array:

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


# Get input sequences and output notes from list of songs
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


# Normalize input sequences - encode notes to numbers with @note_to_int dictionary
def Xnormalize(input_list, note_to_int) -> np.array:
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


# Normalize output notes - encode notes to numbers with @note_to_int dictionary
def ynormalize(output_list, note_to_int) -> np.array:
    """convert output np.array from notes to ints"""

    output_seq = []

    for i in output_list:
        output_seq.append(note_to_int[i])

    output_seq = np.array(output_seq)
    return output_seq


if __name__ == '__main__':

    try:
        dirname = sys.argv[1]
    except IndexError:
        print("You have to specify input folder! ")
        dirname = '../../input_midi'

    try:
        no_of_timesteps = int(sys.argv[2])
    except IndexError:
        print("You have to specify number of timesteps! ")
        no_of_timesteps = 16

    try:
        freq_threshold = sys.argv[3]
    except IndexError:
        freq_threshold = 50

    # 1. load raw data from midi files
    data = load_midi_input(dirname, withLengths=False, withRests=True, instrumentFilter='Piano')
    clean_data = remove_rare(data, threshold=freq_threshold)

    # 2. prepare data - split input/output
    raw_input, raw_output = prepare_data(clean_data, no_of_timesteps)

    # define variables
    unique_notes = list(set([element for notes in clean_data for element in notes]))
    note_to_int = dict((note_, number) for number, note_ in enumerate(unique_notes))
    int_to_note = dict((number, note_) for number, note_ in enumerate(unique_notes))

    # 3. normalize data
    X = Xnormalize(raw_input, note_to_int) / len(unique_notes)
    y = ynormalize(raw_output, note_to_int)

    # 4. export data
    generate_output(X, y, int_to_note, no_of_timesteps)




