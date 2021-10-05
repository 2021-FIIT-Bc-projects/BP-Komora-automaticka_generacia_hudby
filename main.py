from sklearn.model_selection import train_test_split
from keras.callbacks import *
from keras.models import load_model
import numpy as np

from Data_handling import load_midi, prepare_data, normalize_input, normalize_output, reshapeX, generate_midi
from LSTM import lstm_model
from Predict import generate_predictions

data = load_midi(".\\input_midi")
notes = [element for notes in data for element in notes]
n_vocab = len(list(set(notes)))

no_of_timesteps = 16
input_list, output_list = prepare_data(data, no_of_timesteps)
normalized_input_list = normalize_input(input_list)
normalized_output_list = normalize_output(output_list)

x_tr, x_val, y_tr, y_val = train_test_split(normalized_input_list, normalized_output_list, test_size=0.2, random_state=0)

x_tr = reshapeX(x_tr)
x_val = reshapeX(x_val)

model = lstm_model(n_vocab, no_of_timesteps)
model.summary()
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
history = model.fit(np.array(x_tr), np.array(y_tr), batch_size=100, epochs=30, validation_data=(np.array(x_val), np.array(y_val)), verbose=1, callbacks=[mc])

best_model = load_model('best_model.h5')
predictions = generate_predictions(x_val, best_model, no_of_timesteps, range_of_prediction=no_of_timesteps)

unique_x = list(set(input_list.ravel()))
x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x))

print(x_int_to_note)
predicted_notes = [x_int_to_note[i] for i in predictions]
print(predicted_notes)


generate_midi(predicted_notes)
