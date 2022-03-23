from keras.callbacks import *
from keras.layers import *
from keras.models import *
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


from IO import load_data
from IO import loadParams
from IO import init_output
from IO import bundle_output


def reshape_X(x):
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    return x


def lstm_model(n_vocab, no_of_timesteps, LSTM_size=512, Dense_size=256, recurrent_dropout=0.3):
    model = Sequential()
    model.add(LSTM(LSTM_size, input_shape=(1, no_of_timesteps), return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(LSTM(LSTM_size, return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(LSTM(LSTM_size))
    model.add(BatchNormalization())
    model.add(Dropout(recurrent_dropout))
    model.add(Dense(Dense_size))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(recurrent_dropout))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def show_loss(history, no_of_epochs, filename="", show=True):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.xlim([0, no_of_epochs - 1])
    plt.ylim([0, 5])
    plt.legend(loc='upper left')
    if show: plt.show()
    plt.savefig(filename)
    plt.figure().clear()


def show_acc(history, no_of_epochs, filename="", show=True):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.xlim([0, no_of_epochs - 1])
    plt.ylim([0, 1])
    plt.legend(loc='upper left')
    if show: plt.show()
    plt.savefig(filename)
    plt.figure().clear()


if __name__ == "__main__":

    # Extract parameters from config.json
    params = loadParams()

    print(params)

    # Load data
    try:
        X, y, int_to_note = load_data('../preprocessing/preprocessed-16_1740.zip')
    except IndexError:
        print("No path parameter")
        sys.exit(2)

    # split training and testing values
    x_tr, x_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    x_tr = reshape_X(x_tr)
    x_val = reshape_X(x_val)

    # number of distinct notes
    n_vocab = len(int_to_note)

    # train model
    model = lstm_model(n_vocab=n_vocab,
                       no_of_timesteps=params['no_of_timesteps'],
                       LSTM_size=params['LSTM_size'],
                       Dense_size=params['Dense_size'],
                       recurrent_dropout=params['recurrent_dropout'])

    model.summary()

    init_output()
    mc = ModelCheckpoint('output/model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    history = model.fit(x=np.array(x_tr),
                        y=np.array(y_tr),
                        batch_size=params['batch_size'],
                        epochs=params['no_of_epochs'],
                        validation_data=(np.array(x_val), np.array(y_val)),
                        verbose=1,
                        callbacks=[mc])

    best_model = load_model('output/model.h5')
    score = best_model.evaluate(x_val, y_val, verbose=0)
    print("Test Score: ", score[0])
    print("Test accuracy: ", score[1])

    show_acc(history, params['no_of_epochs'], filename="output/acc", show=False)
    show_loss(history, params['no_of_epochs'], filename="output/loss", show=False)

    bundle_output(int_to_note, reshape_X(X), params['no_of_timesteps'])
    