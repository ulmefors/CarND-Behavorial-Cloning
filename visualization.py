import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import model
from keras.preprocessing.image import img_to_array, load_img


def plot_count(data_frame):
    nb_bins = 50

    data_original = data_frame['steering']
    data_non_zero = data_original[data_original != 0]

    plt.subplot(1, 2, 1)
    plt.hist(data_original, nb_bins)
    plt.title('Original distribution')
    plt.xlabel('Steering angle')
    plt.ylabel('Sample count')
    plt.xlim([-1, 1])

    plt.subplot(1, 2, 2)
    plt.hist(data_non_zero, nb_bins)
    plt.title('Modified distribution')
    plt.xlabel('Steering angle')
    plt.ylabel('Sample count')
    plt.xlim([-1, 1])

    plt.show()


def plot_augmented_count(data_frame):
    nb_bins = 50

    data_non_zero = data_frame[data_frame['steering'] != 0]
    original_steering = data_non_zero['steering']

    augmented_steering = []
    for idx, row in data_non_zero.iterrows():
        image, steering = model.get_augmented_row(row)
        augmented_steering.append(steering)

    plt.subplot(1, 2, 1)
    plt.hist(original_steering, nb_bins)
    plt.title('Sampled steering')
    plt.xlabel('Steering angle')
    plt.ylabel('Sample count')
    plt.xlim([-1, 1])

    plt.subplot(1, 2, 2)
    plt.hist(augmented_steering, nb_bins)
    plt.title('Augmented steering')
    plt.xlabel('Steering angle')
    plt.ylabel('Sample count')
    plt.xlim([-1, 1])

    plt.show()


def plot_three_cameras(data_point):
    steering = data_point['steering']

    cameras = ['left', 'center', 'right']

    i = 0
    offset = 0.25
    for camera in cameras:
        num = i + 1
        title = "{0:.3f}".format(steering - (num - 2) * offset)
        image = get_image(data_point, camera=camera)
        plt.subplot(1, 3, num)
        plt.axis("off")
        plt.title(title)
        plt.imshow(image)
        i += 1
    plt.show()


def plot_horozontal_flip(data_point):
    steering = data_point['steering']

    image = get_image(data_point)

    image_flip = cv2.flip(image, 1)

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("{0:.2f}".format(steering))
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("{0:.2f}".format(-steering))
    plt.imshow(image_flip)

    plt.show()


def plot_brightnes_modification(data_point):

    image = get_image(data_point)

    cols = 5
    rows = 3
    num = cols * rows

    for i in range(num):
        plt.subplot(rows, cols, i + 1)
        plt.axis("off")
        image_new = model.augment_brightness_camera_images(image)
        plt.imshow(image_new)

    plt.show()


def get_image(data_point, camera='center'):
    image = load_img('data/sample/' + data_point[camera].strip())
    image = np.array(img_to_array(image), dtype=np.uint8)
    return image


def plot_image_res(data_point):
    image = get_image(data_point)
    image_small = model.preprocess_image(image).astype(np.uint8)

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("original resolution")
    plt.subplot(1, 2, 2)
    plt.imshow(image_small)
    plt.axis("off")
    plt.title("reduced resolution")
    plt.show()


def main():

    data_frame = pd.read_csv('data/sample/driving_log.csv', usecols=[0, 1, 2, 3])

    sample = 3918 # 3968
    data_point = data_frame.loc[sample]

    # Uncomment to run visualizations

    # plot_three_cameras(data_point)
    # plot_horozontal_flip(data_point)
    # plot_brightnes_modification(data_point)
    # plot_count(data_frame)
    # plot_augmented_count(data_frame)
    # plot_image_res(data_point)


if __name__ == "__main__":
   main()


