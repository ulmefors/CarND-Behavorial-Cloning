# Create and train model
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pickle
import pre_processor


def read_image(image_file):
    bgr_image = cv2.imread(image_file)
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def load_data():
    dir = 'data/sample/'
    driving_log = pd.read_csv(dir + 'driving_log.csv')

    nb_rows = driving_log.shape[0] // 3
    nb_cols = driving_log.shape[1]
    col_ctr_image = 0
    col_steering = 3

    center_images = driving_log.ix[:nb_rows - 1, col_ctr_image]
    y_train = driving_log.ix[:nb_rows - 1, col_steering]

    file_name = dir + center_images[0]
    sample_image = read_image(file_name)

    # Initialize X_train array
    X_train = np.zeros((nb_rows,) + sample_image.shape)

    for i in range(nb_rows):
        image = read_image(dir + center_images[i])
        X_train[i] = image
    assert (X_train.shape[0] == y_train.shape[0])

    return X_train, y_train


def plot_count(y_train, y_val):
    nb_bins = pre_processor.__nb_bins__
    plt.subplot(1, 2, 1)
    plt.hist(y_val, nb_bins)
    plt.xlabel('Steering angle')
    plt.ylabel('Count validation')

    plt.subplot(1, 2, 2)
    plt.hist(y_train, nb_bins)
    plt.xlabel('Steering angle')
    plt.ylabel('Count training (normalized)')

    plt.show()


def save_data(X_train, y_train, X_val, y_val):
    pickle.dump({'features': np.array(X_train), 'labels': np.array(y_train)}, open('train.p', 'wb'))
    pickle.dump({'features': np.array(X_val), 'labels': np.array(y_val)}, open('val.p', 'wb'))
    print('Saved data')


def main():
    start_time = time.time()

    X_train, y_train = load_data()
    X_train, X_val, y_train, y_val = pre_processor.process(X_train, y_train, norm_count=True)

    end_time = time.time()
    print('Elapsed time {:.1f} s'.format(end_time - start_time))

    save_data(X_train, y_train, X_val, y_val)
    # plot_count(y_train, y_val)


if __name__ == "__main__":
   main()










