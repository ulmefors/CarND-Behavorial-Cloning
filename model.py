import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
import cv2
import cnn
import json
import os
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot as kerasplot

rows, cols, ch = 20, 64, 3

TARGET_SIZE = (cols, rows)
IMAGE_SHAPE = (rows, cols, ch)


def augment_brightness_camera_images(image):
    # Credit to Vivek Yadav
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.1nbgoagsm

    # Convert to HSV color space where V channel is brightness
    output = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Randomly generate the brightness modification factor
    # Add a constant so that it prevents the image from being completely dark. Range 0.25-1.25.
    random_bright = .25 + np.random.uniform()

    # Apply the brightness modification to the V channel
    output[:, :, 2] = output[:, :, 2] * random_bright

    # Convert back to RBG
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)
    return output


def preprocess_image(image):
    # Crop image - remove sky and car hood
    image = image[30:130, :, :]

    # Reduce image size using CV2 INTER_AREA
    # http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize
    image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)


def get_augmented_row(row, validation=False):

    # Credit to Vivek Yadav and Subodh Malgonde for the foundation of augmentation techniques

    # Validation data will not be augmented
    steering_adjust = 0.25

    steering = row['steering']

    # Choose random camera image for training data
    camera = 'center' if validation else np.random.choice(['center', 'left', 'right'])

    # Adjust steering angle for left and right cameras
    if camera == 'left':
        steering += steering_adjust
    elif camera == 'right':
        steering -= steering_adjust

    image = load_img('data/sample/' + row[camera].strip())
    image = img_to_array(image)

    # Randomly perform horizontal flip of training images to eliminate left/right bias
    if not validation:
        flip_prob = np.random.random() # Random number 0.0 - 1.0
        if flip_prob > 0.5:
            # Flip image and reverse the steering angle
            steering *= -1
            image = cv2.flip(image, 1)

    # Apply brightness augmentation
    if not validation:
        image = augment_brightness_camera_images(image)

    # Crop, resize and normalize the image
    image = preprocess_image(image)

    return image, steering


def get_generator(data_frame, batch_size=32, validation=False):
    nb_data = data_frame.shape[0]

    i = 0
    while True:
        start = i * batch_size
        end = start + batch_size

        X_batch = np.zeros((batch_size,) + IMAGE_SHAPE, dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0
        data_batch = data_frame.loc[start:end - 1]
        for idx, row in data_batch.iterrows():
            X_batch[j], y_batch[j] = get_augmented_row(row, validation)
            j += 1

        i += 1
        # if next batch exceeds upper bound, reset counter
        if end + batch_size > nb_data:
            i = 0

        yield X_batch, y_batch


def plot_loss(history):
    plt.plot(history.history['loss'], '-b')
    plt.plot(history.history['val_loss'], '-r')
    plt.legend(['training', 'validation'])
    plt.yscale('log')
    plt.show()


def main():
    BATCH_SIZE = 32

    data_frame = pd.read_csv('data/sample/driving_log.csv', usecols=[0, 1, 2, 3])

    # shuffle the data
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    # 80-20 training validation split
    training_split = 0.8
    num_rows_training = int(data_frame.shape[0] * training_split)
    training_data = data_frame.loc[0:num_rows_training]
    validation_data = data_frame.loc[num_rows_training:]

    # Remove all zero angle data from training set to avoid straight driving bias
    # This is quite drastic but proved effective
    training_data = training_data[training_data.steering != 0]

    training_generator = get_generator(training_data, batch_size=BATCH_SIZE)
    validation_data_generator = get_generator(validation_data, batch_size=BATCH_SIZE, validation=True)

    model = cnn.get_model()

    model.summary()

    # Load weights from previous training if more training epochs are required
    #model.load_weights('model.h5')

    samples_per_epoch = (training_data.shape[0] * 8 // BATCH_SIZE) * BATCH_SIZE

    history = model.fit_generator(training_generator, validation_data=validation_data_generator,
        samples_per_epoch=samples_per_epoch, nb_epoch=8, nb_val_samples=validation_data.shape[0])

    print("Saving model weights and configuration file.")

    model.save_weights('model.h5', overwrite=True)
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    plot_loss(history)

    # Save visualization of model
    output_dir = './outputs/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    kerasplot(model, to_file=output_dir + 'model.png', show_shapes=True)


if __name__ == "__main__":
    main()
