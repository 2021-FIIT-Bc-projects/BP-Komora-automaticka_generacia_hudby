from sklearn.model_selection import train_test_split
from keras.callbacks import *
from keras.models import load_model
import numpy as np
import pandas as pd

from data_handling import load_midi, prepare_data, normalize_input, normalize_output, reshapeX, generate_midi, \
    remove_rare
from LSTM import lstm_model
from predict import generate_predictions
from visualize import show_loss, show_acc

# knobs to tweak
freq_threshold = 10
no_of_timesteps = 16
LSTM_size = 512
Dense_size = 256
recurrent_dropout = 0.2
no_of_epochs = 100
batch_size = 100


# loading data
data = load_midi(".\\input_midi", withLengths=False)
data = remove_rare(data, threshold=freq_threshold)

notes = [element for notes in data for element in notes]
unique_notes = list(set(notes))
n_vocab = len(unique_notes)
note_to_int = dict((note_, number) for number, note_ in enumerate(unique_notes))
int_to_note = dict((number, note_) for number, note_ in enumerate(unique_notes))

# prepare and normalize data
input_list, output_list = prepare_data(data, no_of_timesteps)
normalized_input_list = normalize_input(input_list, note_to_int)
normalized_output_list = normalize_output(output_list, note_to_int)

# split training and testing values
x_tr, x_val, y_tr, y_val = train_test_split(normalized_input_list, normalized_output_list, test_size=0.2, random_state=n_vocab)
x_tr = reshapeX(x_tr)
x_val = reshapeX(x_val)

# train model
model = lstm_model(n_vocab, no_of_timesteps, LSTM_size=LSTM_size, Dense_size=Dense_size, recurrent_dropout=recurrent_dropout)
model.summary()
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
history = model.fit(np.array(x_tr), np.array(y_tr), batch_size=batch_size, epochs=no_of_epochs, validation_data=(np.array(x_val), np.array(y_val)), verbose=1, callbacks=[mc])

# predict music
best_model = load_model('best_model.h5')
for i in range(5):
    predictions = generate_predictions(x_val, best_model, no_of_timesteps, range_of_prediction=no_of_timesteps)
    predicted_notes = [int_to_note[i] for i in predictions]
    print(predicted_notes)
    generate_midi(predicted_notes)

print(int_to_note[0])
# evaluate and visualize
score = model.evaluate(x_val, y_val, verbose=0)
print("Test Score: ", score[0])
print("Test accuracy: ", score[1])

show_acc(history, no_of_epochs)
show_loss(history, no_of_epochs)

