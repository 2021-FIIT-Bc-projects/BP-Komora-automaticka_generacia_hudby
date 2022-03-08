import numpy as np
from keras.models import load_model

from data_handling import load_midi, remove_rare, prepare_data, normalize_X, generate_midi
from predict import prediction_only
from visualize import generate_notes

best_model = load_model('models/exp2_NoT=8.h5')
no_of_timesteps = 8

data = load_midi("..\\input_midi", withLengths=False, withRests=True, instrumentFilter='Piano')

# get n_vocab and note enumerating dictionary
notes = [element for notes in data for element in notes]
unique_notes = list(set(notes))
n_vocab = len(unique_notes)


note_to_int = dict((note_, number) for number, note_ in enumerate(unique_notes))
int_to_note = dict((number, note_) for number, note_ in enumerate(unique_notes))

raw_X, raw_y = prepare_data(data, no_of_timesteps)
X = normalize_X(raw_X, note_to_int) / n_vocab

for i in range(5):

    ind = np.random.randint(0, len(X) - 1)
    random_music = X[ind]

    print(int_to_note[i] for i in random_music)

    prediction = prediction_only(random_music, best_model, no_of_timesteps, n_vocab, range_of_prediction=4*no_of_timesteps)
    predicted_notes = [int_to_note[i] for i in prediction]
    generate_midi(predicted_notes, filename='prediction_only' + str(i) + '.mid')
    print(predicted_notes)
    generate_notes(predicted_notes)
