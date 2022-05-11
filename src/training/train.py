import os
import argparse

from keras.callbacks import *
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


from IO import load_data
from IO import loadParams
from IO import init_output
from IO import bundle_output
from models import generate_model, basic_RNN

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def visualize_loss(history, no_of_epochs, filename=""):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.xlim([0, no_of_epochs - 1])
    plt.ylim([0, 5])
    plt.legend(loc='upper left')
    plt.savefig(filename)
    plt.figure().clear()


def visualize_acc(history, no_of_epochs, filename=""):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.xlim([0, no_of_epochs - 1])
    plt.ylim([0, 1])
    plt.legend(loc='upper left')
    plt.savefig(filename)
    plt.figure().clear()


if __name__ == "__main__":

    # Extract parameters from config.json
    params = loadParams()
    print(params)

    # create a parser object
    parser = argparse.ArgumentParser(description="An addition program")

    # add argument
    parser.add_argument("-path", type=str, nargs=1, metavar="path to preprocessed data")
    parser.add_argument("-model", type=str, nargs=1, metavar="type of rnn")
    parser.add_argument("-mc", type=str, nargs=1, default=["1"], metavar="complexity of rnn")
    parser.add_argument("-nots", type=int, nargs=1, default=[8], metavar="number of timesteps")
    parser.add_argument("-activation", type=str, nargs=1, default=['relu'], metavar="activation function")

    # parse the arguments from standard input
    args = parser.parse_args()

    # Load data
    input_path = args.path[0]
    model_type = args.model[0]
    model_complexity = args.mc[0]
    no_of_timesteps = args.nots[0]
    activation = args.activation[0]
    params["model"] = model_type
    params["model_complexity"] = model_complexity
    params["no_of_timesteps"] = no_of_timesteps
    params["activation"] = activation
    X, y, int_to_note = load_data(input_path)

    # number of distinct notes
    n_vocab = len(int_to_note)

    # split training and testing values
    x_tr, x_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    init_output(x_tr, x_val, params, int_to_note)

    x_tr = np.reshape(x_tr, (x_tr.shape[0], 1, x_tr.shape[1]))
    x_val = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))

    one_hot_ytr = np.zeros((y_tr.size, n_vocab))
    one_hot_ytr[np.arange(y_tr.size), y_tr] = 1

    one_hot_yval = np.zeros((y_val.size, n_vocab))
    one_hot_yval[np.arange(y_val.size), y_val] = 1

    # train model
    model = generate_model(model_type=model_type,
                           model_complexity=model_complexity,
                           n_vocab=n_vocab,
                           dense_size=params["dense_size"],
                           recurrent_size=params["recurrent_size"],
                           no_of_timesteps=params["no_of_timesteps"],
                           recurrent_dropout=params["recurrent_dropout"],
                           metrics=params["metrics"],
                           activation=params["activation"]
    )

    # model = basic_RNN(no_of_timesteps=params["no_of_timesteps"], n_vocab=n_vocab)

    model.summary()

    mc = ModelCheckpoint('output/model.h5', monitor='loss', mode='min', save_best_only=True, verbose=1)
    earlyStopping = EarlyStopping(monitor='loss', patience=3)

    history = model.fit(x=x_tr,
                        y=one_hot_ytr,
                        batch_size=params['batch_size'],
                        epochs=params['no_of_epochs'],
                        validation_data=(x_val, one_hot_yval),
                        verbose=1,
                        callbacks=[mc])

    best_model = load_model('output/model.h5')
    score = best_model.evaluate(x_val, one_hot_yval, verbose=0)
    print("Test Score: ", score[0])
    print("Test accuracy: ", score[1])

    visualize_acc(history, params['no_of_epochs'], filename="output/acc")
    visualize_loss(history, params['no_of_epochs'], filename="output/loss")

    bundle_output(params['no_of_timesteps'], str(str(model_type) + str(activation)))
    