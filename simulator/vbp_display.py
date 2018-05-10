"""Code adapted from https://github.com/experiencor/deep-viz-keras"""
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
import tensorflow as tf
import PIL.Image
import keras.backend as K
from matplotlib import pylab as plt
from visual_backprop import VisualBackprop
import cv2, os
import argparse

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
HEIGHT_OFFSET_TOP, HEIGHT_OFFSET_BOTTOM = 60, 25
ORIGINAL_HEIGHT, ORIGINAL_WIDTH = 160, 320
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def show_image(image, grayscale = True, ax=None, title=''):
    if ax is None:
        plt.figure()
    plt.axis('off')
    
    if len(image.shape) == 2 or grayscale == True:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)
            
        vmax = np.percentile(image, 99)
        vmin = np.min(image)

        plt.imshow(image, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.title(title)
        print('if')
    else:
        image = image + 127.5
        image = image.astype('uint8')
        
        plt.imshow(image)
        plt.title(title)
        print ('else')
    cv2.imwrite('mask.png', image)
    
def load_image(file_path):
    return cv2.imread(file_path)

def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = image[HEIGHT_OFFSET_TOP : -1 * HEIGHT_OFFSET_BOTTOM, :, :]
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image



def main():
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'img_path',
        type=str,
        help='Path to image file. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # Load and compile the model
    # model = load_model('model_final.h5')
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    model = load_model(model_path)

    # Load an image and make the prediction
    # img_path = '/home/accts/vwh5/senior-project/beta_simulator_linux/recordings_track1/IMG/center_2018_04_25_17_33_36_293.jpg'
    img_path = (args.img_path)
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), img_path)
    img = load_image(img_path)

    img = preprocess(img)
    x = np.expand_dims(img, axis=0)

    visual_bprop = VisualBackprop(model)
    mask = visual_bprop.get_mask(x[0])
    mask = cv2.resize(mask, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT - HEIGHT_OFFSET_TOP - HEIGHT_OFFSET_BOTTOM), cv2.INTER_AREA)

    # mask = np.full((75, 320, 3), 255.0) - np.matmul(mask[:, :, np.newaxis], np.array([[[1, 0, 1]]])) # White background
    mask = np.matmul(mask[:, :, np.newaxis], np.array([[[0, 1, 0]]])) # Black background
    cv2.imwrite('mask.png', mask)
    mask = load_image(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mask.png'))
    bg = load_image(img_path)
    overlay = load_image(img_path)
    x_offset = 0
    y_offset = HEIGHT_OFFSET_TOP

    overlay[y_offset:y_offset+mask.shape[0], x_offset:x_offset+mask.shape[1]] = mask
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, bg, 1 - alpha, 0, bg)
    cv2.imwrite('overlay.png', bg)


if __name__ == '__main__':
    main()
