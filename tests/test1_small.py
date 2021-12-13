from sklearn.model_selection import train_test_split
from keras.callbacks import *
from keras.models import load_model
import numpy as np

from data_handling import prepare_data, reshape_X
from LSTM import lstm_model
from data_generation import generate_data3
from visualize import show_evaluation, show_acc, show_loss

# knobs to tweak
LSTM_size = 64
Dense_size = 16
recurrent_dropout = 0.2
no_of_epochs = 250
batch_size = 32

# load data
data = [generate_data3() for i in range(10)]

# prepare and normalize data
notes = [element for notes in data for element in notes]
unique_notes = list(set(notes))
n_vocab = len(unique_notes)


# train model
for no_of_timesteps in range(2, 32):

    # split training and testing values
    X, y = prepare_data(data, no_of_timesteps)
    x_tr, x_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=n_vocab)
    x_tr = reshape_X(x_tr)
    x_val = reshape_X(x_val)

    # create model
    model = lstm_model(n_vocab, no_of_timesteps, LSTM_size=LSTM_size, Dense_size=Dense_size,
                       recurrent_dropout=recurrent_dropout)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    # train model
    history = model.fit(np.array(x_tr), np.array(y_tr), batch_size=batch_size, epochs=no_of_epochs,
                        validation_data=(np.array(x_val), np.array(y_val)), verbose=0, callbacks=[mc])

    # predict music
    best_model = load_model('best_model.h5')

    # evaluate and visualise
    score = best_model.evaluate(x_val, y_val, verbose=0)

    # write results to file
    f = open("experiments/exp1_scores.txt", "a")
    f.write("No_of_timesteps = {}\n".format(no_of_timesteps))
    f.write("test score: {}".format(score[0]))
    f.write(", test accuracy: {}\n".format(score[1]))
    f.close()

    # generate plots
    show_evaluation(history, no_of_epochs, 'experiments/exp1_NOT={}'.format(no_of_timesteps), show=False)
    show_acc(history, no_of_epochs, 'experiments/exp1_acc_NOT={}'.format(no_of_timesteps), show=False)
    show_loss(history, no_of_epochs, 'experiments/exp1_loss_NOT={}'.format(no_of_timesteps), show=False)

