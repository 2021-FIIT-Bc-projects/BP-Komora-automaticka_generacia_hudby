# Importing libraries
from sklearn.model_selection import train_test_split
from keras.callbacks import *
from keras.models import load_model
import numpy as np
import datetime

# Importing my modules
from data_handling import load_midi, prepare_data, normalize_X, normalize_y, reshape_X, remove_rare
from LSTM import lstm_model
from visualize import show_evaluation, show_acc, show_loss


# Write LSTM model parameters to file
def initExperiment(LSTM_size, Dense_size, recurrent_dropout, no_of_epochs, batch_size, freq_threshold):
    f = open("experiments/exp2_scores.txt", "a")
    f.write("\n----------------------------------------\n")
    f.write("test - {}\n".format(datetime.datetime.now()))
    f.write("LSTM_size = {}\n".format(LSTM_size))
    f.write("Dense_size = {}\n".format(Dense_size))
    f.write("recurrent_dropout = {}\n".format(recurrent_dropout))
    f.write("no_of_epochs = {}\n".format(no_of_epochs))
    f.write("batch_size = {}\n".format(batch_size))
    f.write("freq_threshold = {}\n".format(freq_threshold))
    f.close()


# Write results of validation to file
def writeResults(no_of_timesteps, score):
    f = open("experiments/exp2_scores.txt", "a")
    f.write("No_of_timesteps = {}\n".format(no_of_timesteps))
    f.write("test score: {}".format(score[0]))
    f.write("test accuracy: {}\n\n".format(score[1]))
    f.close()


def test_timesteps(prepare=False, normalize=False, train=False, evaluate=False,
                   LSTM_size=256, Dense_size=128, recurrent_dropout=0.2, no_of_epochs=150, batch_size=16,
                   freq_threshold=20, start_NOT=2):

    # write experiment parameters to file
    initExperiment(LSTM_size, Dense_size, recurrent_dropout, no_of_epochs, batch_size, freq_threshold)

    # load data from midi files
    data = load_midi("..\\input_midi", withLengths=False, withRests=True, instrumentFilter='Piano')
    data = remove_rare(data, threshold=freq_threshold)

    # get n_vocab and note enumerating dictionary
    notes = [element for notes in data for element in notes]
    unique_notes = list(set(notes))
    n_vocab = len(unique_notes)
    note_to_int = dict((note_, number) for number, note_ in enumerate(unique_notes))

    # looping through different numbers of timesteps
    for no_of_timesteps in range(start_NOT, 21, 2):

        # Prepare data
        if prepare:
            raw_X, raw_y = prepare_data(data, no_of_timesteps)

        # Normalize data
        if normalize:
            X = normalize_X(raw_X, note_to_int) / n_vocab
            y = normalize_y(raw_y, note_to_int)
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=n_vocab)
            X_tr = reshape_X(X_tr)
            X_val = reshape_X(X_val)

        # Train model
        if train:
            model = lstm_model(n_vocab, no_of_timesteps, LSTM_size=LSTM_size, Dense_size=Dense_size,
                               recurrent_dropout=recurrent_dropout)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
            history = model.fit(np.array(X_tr), np.array(y_tr), batch_size=batch_size, epochs=no_of_epochs,
                                validation_data=(np.array(X_val), np.array(y_val)), verbose=0, callbacks=[mc])

        # Evaluate best model
        if evaluate:
            best_model = load_model('best_model.h5')
            score = best_model.evaluate(X_val, y_val, verbose=0)

            writeResults(no_of_timesteps, score)

            show_evaluation(history, no_of_epochs, 'experiments/exp2_NOT={}'.format(no_of_timesteps), show=False)
            show_acc(history, no_of_epochs, 'experiments/exp2_acc_NOT={}'.format(no_of_timesteps), show=False)
            show_loss(history, no_of_epochs, 'experiments/exp2_loss_NOT={}'.format(no_of_timesteps), show=False)

    print("Test completed")


test_timesteps(prepare=True, normalize=True, train=True, evaluate=True, start_NOT=18)
