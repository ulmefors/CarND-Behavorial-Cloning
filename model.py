# Create and train model
import pandas as pd
import numpy as np
import cv2
import random
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.mlab as mlab
import time
import pickle
from keras.layers import Flatten, Dense, Dropout, ELU, LeakyReLU, Activation, Conv2D, MaxPooling2D
from keras.models import Sequential


start_time = time.time()

with open('train.p', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['features'], data['labels']


input_shape = np.array(X_train).shape[1:]

print('Loaded data')
print(np.array(X_train).shape)
print(np.array(y_train).shape)

plt.xlabel('Steering angle')
plt.ylabel('Count')

n, bins, patches = plt.hist(np.array(y_train), 100)

end_time = time.time()
print(end_time - start_time)

#plt.show()





def get_model(input_shape, time_len=1):
    #https: // github.com / commaai / research / blob / master / train_steering_model.py

    model = Sequential()
    model.add(Conv2D(16, 8, 8, input_shape=input_shape))
    model.add(ELU())
    model.add(Conv2D(32, 5, 5))
    model.add(ELU())
    model.add(Conv2D(64, 5, 5))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

model = get_model(input_shape)

model.summary()