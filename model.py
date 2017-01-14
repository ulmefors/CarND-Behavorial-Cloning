import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
import cv2
import cnn

rows, cols, ch = 16, 32, 3

TARGET_SIZE = (cols, rows)
IMAGE_SHAPE = (rows, cols, ch)


def augment_brightness_camera_images(image):
    '''
    :param image: Input image
    :return: output image with reduced brightness
    '''

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def resize_to_target_size(image):
    return cv2.resize(image, TARGET_SIZE)


def crop_and_resize(image):
    '''
    :param image: The input image of dimensions 160x320x3
    :return: Output image of size 64x64x3
    '''
    #cropped_image = image[55:135, :, :]
    #cropped_image = image[30:129, 10:310, :]
    processed_image = resize_to_target_size(image)

    return processed_image


def preprocess_image(image):
    image = crop_and_resize(image)
    image = image.astype(np.float32)

    return image


def get_augmented_row(row):

    steering = row['steering']

    # randomly choose the camera to take the image from
    camera = np.random.choice(['center', 'left', 'right'])

    # adjust the steering angle for left anf right cameras
    if camera == 'left':
        steering += 0.25
    elif camera == 'right':
        steering -= 0.25

    image = load_img('data/sample/' + row[camera].strip())
    image = img_to_array(image)

    # decide whether to horizontally flip the image:
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # flip the image and reverse the steering angle
        steering = -1*steering
        image = cv2.flip(image, 1)

    # Apply brightness augmentation
    image = augment_brightness_camera_images(image)

    # Crop, resize and normalize the image
    image = preprocess_image(image)

    return image, steering


def get_data_generator(data_frame, batch_size=32):
    N = data_frame.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size,) + IMAGE_SHAPE, dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        for index, row in data_frame.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = get_augmented_row(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch



if __name__ == "__main__":
    BATCH_SIZE = 32

    data_frame = pd.read_csv('data/sample/driving_log.csv', usecols=[0, 1, 2, 3])

    data_frame_zero = data_frame[data_frame.steering == 0]
    data_frame_non_zero = data_frame[data_frame.steering != 0]

    # keep some of the straight driving
    #keep_straight_drive = 0.1
    #data_frame_zero = data_frame_zero.sample(frac=keep_straight_drive).reset_index(drop=True)
    #data_frame = pd.concat([data_frame_non_zero, data_frame_zero])

    # shuffle the data
    data_frame = data_frame_non_zero.sample(frac=1).reset_index(drop=True)

    # 80-20 training validation split
    training_split = 0.8
    num_rows_training = int(data_frame.shape[0]*training_split)

    training_data = data_frame.loc[0:num_rows_training]
    validation_data = data_frame.loc[num_rows_training:]

    # release the main data_frame from memory
    data_frame = None

    training_generator = get_data_generator(training_data, batch_size=BATCH_SIZE)
    validation_data_generator = get_data_generator(validation_data, batch_size=BATCH_SIZE)

    model = cnn.get_marcus_model()

    model.summary()

    #model.load_weights('model.h5')

    samples_per_epoch = (num_rows_training*2//BATCH_SIZE)*BATCH_SIZE

    history = model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=40, nb_val_samples=3000)

    print("Saving model weights and configuration file.")

    model.save_weights('model.h5', overwrite=True)
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())