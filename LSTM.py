from keras.layers import BatchNormalization as BatchNorm
from keras.layers import *
from keras.models import *


def lstm_model(n_vocab, no_of_timesteps):
    model = Sequential()
    model.add(LSTM(512, input_shape=(1, no_of_timesteps), return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model
