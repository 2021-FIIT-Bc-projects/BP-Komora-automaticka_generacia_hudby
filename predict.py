import numpy as np

"""V tomto subore sa venujem generovani predpovedi, pomocou natrenovaneho modelu"""

from data_handling import reshape_X


def prediction_combined(random_music, model, no_of_timesteps, note_to_int, n_vocab, range_of_prediction=10):
    predictions = list(random_music[0] * n_vocab)

    for i in range(5):
        predictions.append(note_to_int['Rest'])

    for i in range(range_of_prediction):
        random_music = random_music.reshape(1, no_of_timesteps)
        random_music = reshape_X(random_music)
        prob = model.predict(random_music)[0]

        y_pred = np.argmax(prob, axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0][0], len(random_music[0][0]), y_pred / n_vocab)
        random_music = random_music[1:]

    return np.array(predictions)


def prediction_only(random_music, model, no_of_timesteps, n_vocab, range_of_prediction=10):
    predictions = []

    for i in range(range_of_prediction):
        random_music = random_music.reshape(1, no_of_timesteps)
        random_music = reshape_X(random_music)
        prob = model.predict(random_music)[0]

        y_pred = np.argmax(prob, axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0][0], len(random_music[0][0]), y_pred / n_vocab)
        random_music = random_music[1:]

    return np.array(predictions)


