from keras.layers import Lambda, Convolution2D, Dense, ELU, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam


'''
    Used for initial testing but eventually abandoned.
'''
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

'''
    Final model used for submission
'''


def get_model():
    ch, row, col = 3, 20, 64  # camera format
    shape = (row, col, ch)

    model = Sequential()

    # Normalization performed in the model using Lambda layer
    # Credit to comma.ai https://github.com/commaai/research/blob/master/train_steering_model.py
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=shape,
                     output_shape=shape))

    # 1 x 1 convolution to learn best color space (credit to Vivek blog post)
    model.add(Convolution2D(3, 1, 1, border_mode="same"))

    # First convolution
    model.add(Convolution2D(16, 3, 3, subsample=(2, 2), border_mode='valid', activation='relu'))

    # Second convolution
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))

    # Third convolution
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode='valid', activation='relu'))

    # Fully connected layer
    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(512))
    model.add(Dropout(0.3))

    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.004), loss="mse")

    return model


if __name__ == "__main__":
    pass
