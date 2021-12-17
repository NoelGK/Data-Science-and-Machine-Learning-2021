import os
import numpy as np
import pickle
from extra_funcs import load_fsdd, train_test_split


corr_spectro_path = 'C:/Users/noelg/Datasets/FSDD/noisy_spectrograms/'
clean_spectro_path = 'C:/Users/noelg/Datasets/FSDD/clean_spectrograms/'
corr_min_max_path = 'C:/Users/noelg/Datasets/FSDD/noisy_min_max/min_max_values.pkl'
clean_min_max_path = 'C:/Users/noelg/Datasets/FSDD/clean_min_max/min_max_values.pkl'

x_data, y_data = load_fsdd(corr_spectro_path, clean_spectro_path)

with open(corr_min_max_path, 'rb') as f:
    corr_min_max_dict = pickle.load(f)  # Dictionary

with open(clean_min_max_path, 'rb') as f:
    clean_min_max_dict = pickle.load(f)

corr_min_max_vals = []
for key, val in corr_min_max_dict.items():
    corr_min_max_vals.append([val['min'], val['max']])  # array

clean_min_max_vals = []
for key, val in clean_min_max_dict.items():
    clean_min_max_vals.append([val['min'], val['max']])  # array

corr_min_max_vals = np.array(corr_min_max_vals)
clean_min_max_vals = np.array(clean_min_max_vals)

x_train, y_train, x_test, y_test, x_test_min_max_vals, y_test_min_max_vals = \
    train_test_split(x_data, y_data, corr_min_max_vals, clean_min_max_vals)


if __name__ == '__main__':
    train_test_dir = 'C:/Users/noelg/Datasets/FSDD/train_test_data/'
    names = ['x_train.npy', 'x_test.npy', 'y_train.npy', 'y_test.npy', 'x_test_min_max.npy', 'y_test_min_max.npy']

    files = [x_train, x_test, y_train, y_test, x_test_min_max_vals, y_test_min_max_vals]

    for name, file in zip(names, files):
        file_path = os.path.join(train_test_dir, name)
        np.save(file_path, file)
