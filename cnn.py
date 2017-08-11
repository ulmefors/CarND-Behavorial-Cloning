from keras.layers import Lambda, Convolution2D, Dense, ELU, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam


def get_nvidia_model():
    # https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    # Based on Nvidia model with some personal tweaks and inspiration from Vivek @ Slack

    learning_rate = 0.001

    row, col, ch = 66, 200, 3  # camera format
    shape = (row, col, ch)

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0,
                     input_shape=shape,
                     output_shape=shape))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(ELU())
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))

    # TODO: init method?
    # TODO: other activation?
    # TODO: learning rate?

    model.compile(optimizer="adam", loss="mse")

    return model


def get_comma_model():
    # https: // github.com / commaai / research / blob / master / train_steering_model.py

    # Parameter subsample is same as stride. Reduces size.
    # Parameter border_mode is same as padding.
    # Keras docs | border_mode: 'valid', 'same' or 'full'. ('full' requires the Theano backend.)

    ch, row, col = 3, 160, 320  # camera format
    shape = (row, col, ch)

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=shape,
                     output_shape=shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='valid', input_shape=shape))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def get_subodh_model():
    ch, row, col = 3, 64, 64  # camera format
    shape = (row, col, ch)

    model = Sequential()

    # layer 1 output shape is 32x32x32
    model.add(Lambda(lambda x: x / 127.5 - 1.0,
                     input_shape=shape,
                     output_shape=shape))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    # layer 2 output shape is 15x15x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # layer 3 output shape is 12x12x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))

    # Flatten the output
    model.add(Flatten())

    # layer 4
    model.add(Dense(1024))
    model.add(Dropout(.3))
    model.add(ELU())

    # layer 5
    model.add(Dense(512))
    model.add(ELU())

    # Finally a single output, since this is a regression problem
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model


def get_model():
    ch, row, col = 3, 96, 320  # camera format
    shape = (row, col, ch)

    model = Sequential()

    # Normalization performed in the model using Lambda layer
    # Credit to comma.ai https://github.com/commaai/research/blob/master/train_steering_model.py
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=shape,
                     output_shape=shape))

    # 1 x 1 convolution to learn best color space (credit to Vivek blog post)
    model.add(Convolution2D(3, 1, 1, border_mode="same", activation='relu'))

    model.add(Convolution2D(8, 8, 8, subsample=(4, 4), border_mode='valid', activation='relu'))

    model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))

    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(64))
    model.add(ELU())

    model.add(Dense(16))
    model.add(ELU())

    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.0001), loss="mse")

    return model

def get_small_model():
    ch, row, col = 3, 20, 64  # camera format
    shape = (row, col, ch)

    model = Sequential()

    # Normalization performed in the model using Lambda layer
    # Credit to comma.ai https://github.com/commaai/research/blob/master/train_steering_model.py
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=shape,
                     output_shape=shape))

    # 1 x 1 convolution to learn best color space (credit to Vivek blog post)
    model.add(Convolution2D(3, 1, 1, border_mode="same", activation='relu'))

    model.add(Convolution2D(8, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu'))

    model.add(Convolution2D(16, 2, 2, subsample=(2, 2), border_mode='valid', activation='relu'))

    model.add(Convolution2D(32, 2, 2, subsample=(1, 1), border_mode='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(64))
    model.add(ELU())

    model.add(Dense(16))
    model.add(ELU())

    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.0001), loss="mse")

    return model


if __name__ == "__main__":
    pass
