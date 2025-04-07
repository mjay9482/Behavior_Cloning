import argparse
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import eventlet
import socketio
import tensorflow as tf
from flask import Flask
from flask import Flask, render_template

sio = socketio.Server()
app = Flask(__name__)

model = tf.keras.models.load_model('model.h5')

speed_limit = 10

def preprocess(image):
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3,3),0)
    image = cv2.resize(image, (200, 66))
    image = image / 255.0
    return image

@sio.on('connect')
def connect(sid, environ):
    print("Simulator connected.")
    send_control(0.0, 0.0)  

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        img_str = data["image"]
        img_bytes = base64.b64decode(img_str)
        image = Image.open(BytesIO(img_bytes))
        image_array = np.asarray(image)

        image_processed = preprocess(image_array)
        image_input = np.expand_dims(image_processed, axis=0)  

        steering_angle = float(model.predict(image_input, verbose=0))

        speed = float(data["speed"])
        throttle = 1.0 - speed / speed_limit

        print(f"Steering Angle: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}")

        send_control(steering_angle, throttle)
    else:
        send_control(0.0, 0.0)

    sio.emit("telemetry_data", {
        "steering_angle": str(steering_angle),
        "throttle": str(throttle),
        "speed": str(speed)
    })

def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

