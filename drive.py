import argparse
import base64
import json

import numpy as np
import socketio
import eventlet.wsgi
import keras
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from model import preprocess_image

from keras.models import model_from_json

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    # Add the preprocessing step to match the process from training
    image_array = preprocess_image(image_array)

    transformed_image_array = image_array[None, :, :, :]

    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    # Try to maintain speed close to goal speed
    goal_speed = 20.0
    diff = goal_speed - float(speed)
    # Proportional control
    c_p = 0.6
    throttle = diff * c_p

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        if keras.__version__ >= "1.2.0":
            model = model_from_json(json.loads(jfile.read()))
        else:
            model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)