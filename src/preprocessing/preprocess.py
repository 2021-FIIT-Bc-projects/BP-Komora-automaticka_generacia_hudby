import sys
from collections import Counter
import json
import numpy as np
import pandas as pd

from IO import load_midi


def remove_rare(data, threshold):

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


if __name__ == '__main__':

    try:
        dirname = sys.argv[1]
    except IndexError:
        print("You have to specify input folder")
        dirname = '../../input_midi'

    try:
        no_of_timesteps = int(sys.argv[2])
    except IndexError:
        print("You have to specify number of timesteps")
        no_of_timesteps = 16

    try:
        freq_threshold = sys.argv[3]
    except IndexError:
        freq_threshold = 50

    # load data from midi files
    data = load_midi(dirname, withLengths=False, withRests=True, instrumentFilter='Piano')
    data = remove_rare(data, threshold=freq_threshold)

    print("Data are loaded")

    # prepare and normalize data
    notes = [element for notes in data for element in notes]
    unique_notes = list(set(notes))
    n_vocab = len(unique_notes)

    note_to_int = dict((note_, number) for number, note_ in enumerate(unique_notes))
    int_to_note = dict((number, note_) for number, note_ in enumerate(unique_notes))
    raw_input_list, raw_output_list = prepare_data(data, no_of_timesteps)
    print("Data are prepared")

    X = normalize_X(raw_input_list, note_to_int) / n_vocab
    y = normalize_y(raw_output_list, note_to_int)
    print("---------------------------------------------------------------")
    X = pd.DataFrame(X)
    X.to_csv(r"output/X.csv")
    print("---------------------------------------------------------------")
    y = pd.DataFrame(y)
    y.to_csv(r"output/y.csv")

    with open('output/notes_dict.json', 'w') as convert_file:
        convert_file.write(json.dumps(int_to_note))



