"""Code adapted from https://github.com/SullyChen/Autopilot-TensorFlow"""
import argparse
import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
from keras.models import load_model
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    img = cv2.imread('steering_wheel_image.jpg',0)
    rows,cols = img.shape

    smoothed_angle = 0

    cap = cv2.VideoCapture(0)
    while(cv2.waitKey(10) != ord('q')):
        ret, frame = cap.read()
        image = scipy.misc.imresize(frame, [66, 200]) / 255.0
        degrees = float(model.predict(image, batch_size=1))
        print("Predicted steering angle: " + str(degrees) + " degrees")
        cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
        #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
        #and the predicted angle
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        cv2.imshow("steering wheel", dst)

    cap.release()
    cv2.destroyAllWindows()
