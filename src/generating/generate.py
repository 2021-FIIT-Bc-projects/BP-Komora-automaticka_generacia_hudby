import sys

import numpy as np

from IO import load_data
from IO import bundle_output
from IO import export_music
from IO import loadParams


def generate_music(random_music, model, no_of_timesteps, n_vocab, range_of_prediction=10):
    predictions = []

    for i in range(range_of_prediction):
        random_music = random_music.reshape(1, no_of_timesteps)
        random_music = np.reshape(random_music, (random_music.shape[0], 1, random_music.shape[1]))
        prob = model.predict(random_music)[0]

        y_pred = np.argmax(prob, axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0][0], len(random_music[0][0]), y_pred / n_vocab)
        random_music = random_music[1:]

    return np.array(predictions)


if __name__ == "__main__":

    params = loadParams()

    # Load data
    try:
        model, int_to_note = load_data('./trained-16_1656.zip')
    except IndexError:
        print("No path parameter")
        sys.exit(2)

    try:
        predictions = sys.argv[2]
        prediction_length = sys.argv[3]
    except IndexError:
        predictions = 5
        prediction_length = 50

    n_vocab = len(int_to_note)
    x_val = load_sample(params['no_of_timesteps'])
    """ >>> 1. z trenovacich dat"""
    """ >>> 2. vygenerovat si nejaky nahodny tensor"""
    """ >>> 3. ina hudba"""

    for i in range(predictions):
        prediction = generate_music(x_val, model, params['no_of_timesteps'], n_vocab, range_of_prediction=prediction_length)
        predicted_notes = [int_to_note[i] for i in prediction]
        export_music(predicted_notes, filename=('music(%d).mid' % i))

    bundle_output()
