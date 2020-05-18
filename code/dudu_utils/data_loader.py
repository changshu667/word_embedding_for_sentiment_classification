# Built-in
import os
import pickle

# Libraries
import numpy as np
import pandas as pd

# Custom

# Settings


def load_file(file_name, **kwargs):
    """
    Read the given file name and call the specific function determined by the extension
    :param file_name: path to the file
    :param kwargs: any parameters
    :return: read data
    """
    ext = os.path.splitext(file_name)[-1]
    if ext == '.csv':
        # use pd to read csv
        data = pd.read_csv(file_name, **kwargs)
    elif ext == '.npy':
        # np to load numpy file
        data = np.load(file_name, **kwargs)
    elif ext == '.txt':
        with open(file_name, 'r') as f:
            data = f.readlines()
    elif ext == '.pkl':
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
    else:
        raise NotImplementedError('File type {} not supported yet'.format(ext))
    return data


def save_file(file_name, data):
    ext = os.path.splitext(file_name)[-1]
    if ext == '.txt':
        with open(file_name, 'w+') as f:
            f.writelines(data)
    elif ext == '.pkl':
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
    elif ext == '.csv':
        data.to_csv(file_name)


if __name__ == '__main__':
    file_name = r'../data/training.1600000.processed.noemoticon.csv'
    df = load_file(file_name, encoding="ISO-8859-1")
    print(df.head())
