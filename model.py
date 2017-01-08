# Create and train model
import numpy as np
import time
import pickle
from keras.layers import Flatten, Dense, Dropout, ELU, Convolution2D
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


def get_model(input_shape):
    #https: // github.com / commaai / research / blob / master / train_steering_model.py

    # Parameter subsample is same as stride. Reduces size.
    # Parameter border_mode is same as padding.
    # Keras docs | border_mode: 'valid', 'same' or 'full'. ('full' requires the Theano backend.)

    model = Sequential()
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


def main():
    # Model/Training parameters
    nb_epoch = 10
    batch_size = 16
    h, w, ch = 160, 320, 3
    input_shape = (h, w, ch)

    model = get_model(input_shape)
    model.summary()

    X_train, y_train, X_val, y_val = load_data()

    assert input_shape == np.array(X_train).shape[1:]
    assert input_shape == np.array(X_val).shape[1:]

    if (False):
        save_model_viz(model)
        data_creator.plot_count(y_train, y_val)

    save_model(model)


if __name__ == "__main__":
    main()
