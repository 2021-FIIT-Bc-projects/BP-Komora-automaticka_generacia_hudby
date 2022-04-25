import os
import json
import shutil
import datetime
from os.path import exists

import pandas as pd


def loadParams():

    with open('./config.json') as configFile:
        param_dict = json.load(configFile)

    mandatory_params = ["no_of_timesteps", "recurrent_size", "dense_size", "recurrent_dropout", "batch_size", "no_of_epochs", "model"]
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
    y = pd.read_csv('input/y.csv', index_col=0).to_numpy().flatten()

    with open('input/notes.json') as configFile:
        int_to_note = json.load(configFile)

    parent_dir = os.path.abspath(".")
    path = os.path.join(parent_dir, "input")
    shutil.rmtree(path)

    return X, y, int_to_note


def init_output(x_tr, x_val, params, int_to_note):
    parent_dir = os.path.abspath(".")
    dirname = "output"
    path = os.path.join(parent_dir, dirname)

    if not exists(path):
        os.makedirs(path)

    pd.DataFrame(x_tr).to_csv("output/x_tr.csv")
    pd.DataFrame(x_val).to_csv("output/x_val.csv")
    with open('output/config.json', 'w') as convert_file:
        convert_file.write(json.dumps(params))

    with open('output/notes.json', 'w') as convert_file:
        convert_file.write(json.dumps(int_to_note))


def bundle_output(no_of_timesteps, model_type):

    parent_dir = os.path.abspath(".")
    dirname = "output"
    path = os.path.join(parent_dir, dirname)

    dt = datetime.datetime.now()
    day = dt.strftime("%d")
    month = dt.strftime("%m")

    filename = "../../output/trained/%s-%s %s_trained(%d)" % (day, month, model_type, no_of_timesteps)
    file_format = "zip"
    root_dir = "./output"
    shutil.make_archive(filename, file_format, root_dir)
    shutil.rmtree(path)

    print('All files zipped successfully!')
