import argparse
import os
import sys
import getopt
from collections import Counter
import numpy as np


from IO import load_midi_input, generate_output


# Remove less frequent notes(less frequent, @threshold) from raw data
def remove_rare(data: np.array, threshold: int) -> np.array:

    notes = [note for song in data for note in song]
    freq = dict(Counter(notes))
    frequent_notes = [note_ for note_, count in freq.items() if count >= threshold]

    filtered_dataset = []

    for song in data:
        filtered_song = []
        for note in song:
            if note in frequent_notes:
                filtered_song.append(note)
        filtered_dataset.append(filtered_song)

    return np.array(filtered_dataset, dtype=object)


# Get input sequences and output notes from list of songs
def prepare_data(data, input_sequence_length) -> (np.array, np.array):
    """function, which generates input_sequences and output_notes"""

    input_list = []
    output_list = []

    for song in data:
        for i in range(0, len(song) - input_sequence_length):
            # preparing input and output sequences

            input_sequence = song[i:i + input_sequence_length]
            output_note = song[i + input_sequence_length]

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


def parse_input_args():
    parser = argparse.ArgumentParser(description="An addition program")

    # add argument
    parser.add_argument("-dir", type=str, nargs=1, metavar="path to midi dir")
    parser.add_argument("-nots", type=int, nargs=1, default=8, metavar="number of timesteps")
    parser.add_argument("-ft", type=int, nargs=1, default=[50], metavar="minimum note frequency")
    parser.add_argument("-i", type=str, nargs=1, default="piano", metavar="instrument")

    # parse the arguments from standard input
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # parse the arguments from standard input
    args = parse_input_args()

    dirname = args.dir[0]
    no_of_timesteps = args.nots[0]
    freq_threshold = args.ft[0]
    instrument = args.i[0]

    # 1. load raw data from midi files
    print("Starting data load...")
    data = load_midi_input(dirname, withLengths=False, withRests=False, instrumentFilter=instrument)

    # 2. remove low frequency notes
    print("Removing low frequency notes...")
    clean_data = remove_rare(data, threshold=freq_threshold)

    # 3. prepare data - split input/output
    print("Preparing data...")
    raw_input, raw_output = prepare_data(clean_data, no_of_timesteps)

    # define dicts for encoding of notes
    unique_notes = list(set([element for notes in clean_data for element in notes]))
    note_to_int = dict((note_, number) for number, note_ in enumerate(unique_notes))
    int_to_note = dict((number, note_) for number, note_ in enumerate(unique_notes))

    # 4. normalize data
    print("Normalizing data...")
    X = Xnormalize(raw_input, note_to_int) / len(unique_notes)
    y = ynormalize(raw_output, note_to_int)

    # 5. export data
    dirname = os.path.basename(os.path.normpath(dirname))
    generate_output(X, y, int_to_note, no_of_timesteps, dirname)




