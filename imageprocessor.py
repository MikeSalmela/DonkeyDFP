import numpy as np
import cv2
import functions as f

d_size = (128,48)

def normalize(img):
    return f.normalize(img)

def reshape(image):
    image = f.splitImage(image)
    image = cv2.resize(image, d_size, interpolation=cv2.INTER_CUBIC)
    return f.normalize(image)

def bw(image):
    image = splitImage(image)
    image = cv2.resize(image, dsize=d_size, interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = normalize(image)
    image = np.reshape(image, (48,128,1))
    return image

