def generate_data1(no_of_timesteps=16, no_of_seqs=10):
    notes = [x for y in range(no_of_seqs) for x in range(no_of_timesteps)]
    return notes


def generate_data2(size=10):
    notes = [(x % 2) for x in range(size)]
    return notes


def generate_data3():
    seq1 = [0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0]
    seq2 = [0, 4, 2, 4, 0, 4, 2, 4, 0, 4, 2, 4, 0, 4, 2, 4]
    seq3 = [2, 4, 2, 2, 4, 4, 0, 4, 2, 4, 2, 2, 4, 0, 4, 0]
    seq4 = [0, 1, 2, 4, 2, 3, 2, 2, 0, 4, 4, 0, 0, 1, 2, 4]

    notes = seq1 + seq2 + seq3 + seq4
    return notes


