from keras.layers import BatchNormalization as BatchNorm
from keras.layers import *
from keras.models import *


def lstm_model(n_vocab, no_of_timesteps, LSTM_size=512, Dense_size=256, recurrent_dropout=0.3):
    model = Sequential()
    model.add(LSTM(LSTM_size, input_shape=(1, no_of_timesteps), return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(LSTM(LSTM_size, return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(LSTM(LSTM_size))
    model.add(BatchNorm())
    model.add(Dropout(recurrent_dropout))
    model.add(Dense(Dense_size))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(recurrent_dropout))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
