from keras import Input, losses
import tensorflow as tf
from keras.layers import LSTM, SimpleRNN, CategoryEncoding, ConvLSTM2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import GRU
from keras.models import Sequential, Model
from tensorflow import Tensor


def lstm_model1(no_of_timesteps, lstm_size, recurrent_dropout):
    model = Sequential()

    model.add(LSTM(lstm_size, input_shape=(1, no_of_timesteps), return_sequences=True, recurrent_dropout=recurrent_dropout))

    return model


def lstm_model2(no_of_timesteps, lstm_size, recurrent_dropout):
    model = Sequential()

    model.add(Input(shape=(1, no_of_timesteps)))
    model.add(LSTM(lstm_size, return_sequences=True))
    model.add(LSTM(lstm_size))

    return model


def lstm_model3(no_of_timesteps, lstm_size, recurrent_dropout):
    model = Sequential()

    ######## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Na tomto som to trenoval pred tym
    # model.add(Input(shape=(1, no_of_timesteps)))
    # model.add(LSTM(lstm_size, return_sequences=True))
    # model.add(LSTM(lstm_size))

    model.add(LSTM(lstm_size, input_shape=(1, no_of_timesteps), return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(LSTM(lstm_size, return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(LSTM(lstm_size))

    return model


def gru_model(no_of_timesteps, gru_size, recurrent_dropout):
    model = Sequential()

    model.add(GRU(gru_size, input_shape=(1, no_of_timesteps), return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(GRU(gru_size, return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(GRU(gru_size))

    return model


def simple_rnn_model(no_of_timesteps, simple_rnn_size, recurrent_dropout):
    model = Sequential()

    model.add(SimpleRNN(simple_rnn_size, input_shape=(1, no_of_timesteps), return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(SimpleRNN(simple_rnn_size, return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(SimpleRNN(simple_rnn_size))

    return model


def generate_model(model_type, model_complexity, no_of_timesteps, recurrent_size, recurrent_dropout, dense_size, n_vocab, metrics, activation):
    print(model_type)

    if model_type == "lstm":
        model = lstm_model3(no_of_timesteps, recurrent_size, recurrent_dropout)
    elif model_type == "gru":
        model = gru_model(no_of_timesteps, recurrent_size, recurrent_dropout)
    elif model_type == "simple_rnn":
        model = simple_rnn_model(no_of_timesteps, recurrent_size, recurrent_dropout)

    # model.add(BatchNormalization())
    # model.add(Dropout(recurrent_dropout))
    # model.add(Dense(dense_size))
    # model.add(Activation('relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(recurrent_dropout))
    # model.add(Dense(n_vocab))
    # model.add(CategoryEncoding(num_tokens=1, output_mode='one_hot'))
    # model.add(Activation('softmax'))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if model_complexity == "1":
        model.add(Dropout(recurrent_dropout))
        model.add(Dense(dense_size, activation=activation))
        model.add(Dense(n_vocab, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
    elif model_complexity == "2":
        model.add(BatchNormalization())
        model.add(Dropout(recurrent_dropout))
        model.add(Dense(dense_size, activation=activation))
        model.add(BatchNormalization())
        model.add(Dropout(recurrent_dropout))
        model.add(Dense(n_vocab, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    """
        model.add(BatchNormalization())
        model.add(Dropout(recurrent_dropout))
        model.add(Dense(dense_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(n_vocab, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    """

    return model


def basic_RNN(no_of_timesteps, n_vocab):
    model = Sequential()
    model.add(Input(shape=(None, no_of_timesteps)))
    model.add(SimpleRNN(512, return_sequences=True, activation='relu'))
    model.add(SimpleRNN(512, activation='relu'))
    model.add(Dense(n_vocab))

    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    return model
