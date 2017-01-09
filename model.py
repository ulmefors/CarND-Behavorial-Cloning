# Create and train model
import numpy as np
import time
import pickle
from keras.layers import Flatten, Dense, Dropout, ELU, Convolution2D, Lambda
from keras.models import Sequential
import os
import json
import data_creator
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils.visualize_util import plot

start_time = time.time()

def load_data():
    data_train = pickle.load(open('train.p', 'rb'))
    data_val = pickle.load(open('val.p', 'rb'))
    return data_train['features'], data_train['labels'], data_val['features'], data_val['labels']


def get_nvidia_model(input_shape):
    # https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    # Based on Nvidia model with some personal tweaks and inspiration from Vivek @ Slack

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0,
                     input_shape=input_shape,
                     output_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, subsample=(4, 4), border_mode='same', input_shape=input_shape))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='same', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='same', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', init='he_normal'))
    model.add(Flatten())
    model.add(ELU())
    model.add(Dense(1164, init='he_normal'))
    model.add(ELU())
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    model.add(Dense(1, init='he_normal'))

    model.compile(optimizer="adam", loss="mse")

    return model


def get_comma_model(input_shape):
    # https: // github.com / commaai / research / blob / master / train_steering_model.py

    # Parameter subsample is same as stride. Reduces size.
    # Parameter border_mode is same as padding.
    # Keras docs | border_mode: 'valid', 'same' or 'full'. ('full' requires the Theano backend.)

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0,
                     input_shape=input_shape,
                     output_shape=input_shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='same', input_shape=input_shape))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='same'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def save_model_viz(model):
    # https://keras.io/visualization/
    # brew install graphviz
    # pip install graphviz pydot pydot_ng

    directory = 'outputs/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plot(model, to_file=directory+'model.png', show_shapes=True)


def save_model(model):
    model.save_weights("model.h5", True)
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)


def gen(X, y):

    batch_size = 64

    start = 0
    end = start + batch_size
    data_size = X.shape[0]

    while True:

        batch_images = X[start:end]
        batch_angles = y[start:end]

        X_batch = []
        y_batch = []

        X_batch = batch_images
        y_batch = batch_angles

        start += batch_size
        end += batch_size

        if end >= data_size:
            start = 0
            end = batch_size

        yield (X_batch, y_batch)


def main():
    # Model/Training parameters
    nb_epoch = 10
    h, w, ch = 160, 320, 3
    input_shape = (h, w, ch)

    model = get_nvidia_model(input_shape)
    model.summary()

    X_train, y_train, X_val, y_val = load_data()

    assert input_shape == np.array(X_train).shape[1:]
    assert input_shape == np.array(X_val).shape[1:]

    if (False):
        save_model_viz(model)
        data_creator.plot_count(y_train, y_val)

    h = model.fit_generator(gen(X_train, y_train),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=nb_epoch, validation_data = gen(X_val, y_val), nb_val_samples=500)

    save_model(model)


if __name__ == "__main__":
    main()
