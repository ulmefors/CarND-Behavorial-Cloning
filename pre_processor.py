import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

__nb_bins__ = 100

# Normalize pixel values from 0-255 to approx [-0.5, 0.5]
def normalize(X):
    # Subtract to center around 0
    X_tmp = np.add(X, - 127.5)

    # Divide by 255
    return np.divide(X_tmp, 255.0)


def reduce_max_count(X, y, min, max, count_ratio):
    count = 0
    step = count_ratio

    assert (X.shape[0] == y.shape[0])

    X_filtered = []
    y_filtered = []
    nb = X.shape[0]

    for i in range(nb):
        # http://stackoverflow.com/questions/29305131/problems-with-pandas
        # Use iloc to find numerical index of DataFrame instead of label after shuffling data
        y_tmp = y.iloc[i]

        within_range = min < y_tmp < max

        if not within_range or count % step == 0:
            X_filtered.append(X[i])
            y_filtered.append(y_tmp)

        if within_range:
            count += 1

    return X_filtered, y_filtered


def process(X, y, norm_count=False):

    # Shuffle data
    X, y = shuffle(X, y)

    # Split into train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.125)

    X_train = normalize(X_train)
    X_val = normalize(X_val)

    if norm_count:

        n, bins, patches = plt.hist(y_train, __nb_bins__)

        # Find two largest counts with fast algorithm
        val_runner_up, val_largest = np.partition(n, (len(n) - 1, len(n) - 2))[-2:]
        count_ratio = val_largest // val_runner_up

        # Index at which the largest count of steering angles are found
        max_idx = np.argmax(n)

        # Find the bounds of the most frequent bin
        low_bin_bound, high_bin_bound = bins[max_idx:max_idx + 2]

        X_train, y_train = reduce_max_count(X_train, y_train, low_bin_bound, high_bin_bound, count_ratio)

    return X_train, X_val, y_train, y_val


