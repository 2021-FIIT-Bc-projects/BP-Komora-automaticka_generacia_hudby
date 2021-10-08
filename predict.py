import numpy as np

from data_handling import reshapeX


def generate_predictions(x_val, model, no_of_timesteps, range_of_prediction=10):
    ind = np.random.randint(0, len(x_val) - 1)
    random_music = x_val[ind]
    predictions = list(random_music[0])

    for i in range(10):
        predictions.append(0)

    for i in range(range_of_prediction):
        random_music = random_music.reshape(1, no_of_timesteps)
        random_music = reshapeX(random_music)
        prob = model.predict(random_music)[0]

        y_pred = np.argmax(prob, axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0][0], len(random_music[0][0]), y_pred)
        random_music = random_music[1:]

    return predictions