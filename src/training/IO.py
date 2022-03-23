import os
import json
import shutil
import datetime
from os.path import exists

import pandas as pd


def loadParams():

    with open('config.json') as configFile:
        param_dict = json.load(configFile)

    mandatory_params = ["no_of_timesteps", "LSTM_size", "Dense_size", "recurrent_dropout", "batch_size", "no_of_epochs"]
    for param in mandatory_params:
        if param not in param_dict:
            raise ValueError("%s have to be included in config.json" % param)

    return param_dict


def load_data(path):

    # Target directory
    extract_dir = "./input"

    # Format of archive file
    archive_format = "zip"

    # Unpack the archive file
    try:
        shutil.unpack_archive(path, extract_dir, archive_format)
        print("Archive file unpacked successfully.")
    except shutil.ReadError:
        raise ValueError("%s is not recognized as a zip file" % path)

    X = pd.read_csv('input/X.csv', index_col=0).to_numpy()

    y = pd.read_csv('input/y.csv', index_col=0).to_numpy()

    with open('input/notes.json') as configFile:
        int_to_note = json.load(configFile)

    parent_dir = os.path.abspath(".")
    path = os.path.join(parent_dir, "input")
    shutil.rmtree(path)

    return X, y, int_to_note


def init_output():
    parent_dir = os.path.abspath(".")
    dirname = "output"
    path = os.path.join(parent_dir, dirname)

    if not exists(path):
        os.makedirs(path)


def bundle_output(int_to_note, X, no_of_timesteps):

    parent_dir = os.path.abspath(".")
    dirname = "output"
    path = os.path.join(parent_dir, dirname)

    with open('output/notes.json', 'w') as convert_file:
        convert_file.write(json.dumps(int_to_note))

    dt = datetime.datetime.now()
    hour = dt.strftime("%H")
    minute = dt.strftime("%M")

    filename = "trained-%d_%s%s" % (no_of_timesteps, hour, minute)
    file_format = "zip"
    directory = path
    shutil.make_archive(filename, file_format, directory)
    shutil.rmtree(path)

    print('All files zipped successfully!')