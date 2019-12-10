import cv2
import numpy as np
import os
import random

im = cv2.imread('dog.jpg')

def RotateImage(im, angle, size):
    (h, w) = im.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, size)
    rotated_image = cv2.warpAffine(im, M, (w, h))
    return rotated_image

def GaussianBlur(im, blur):
    if blur%2 ==0:
        blur = blur+1
    GaussianBlur_image = cv2.GaussianBlur(im, (blur,blur), 0)
    return GaussianBlur_image

def MovedImage(im, shiftX, shiftY):
    num_rows, num_cols = im.shape[:2]
    translation_matrix = np.float32([ [1,0,shiftY], [0,1,shiftX] ])
    shift_image = cv2.warpAffine(im, translation_matrix, (num_cols,num_rows))
    return shift_image

    
def MakeRandomParams(rotate=0, blur=0, size = 1.0, shift=0):
    value_rotate =  np.random.randint(-rotate, rotate)
    value_blur =  np.random.randint(1, blur)
    value_size =  np.random.random()
    value_shiftX =  np.random.randint(-shift, shift)
    value_shiftY =  np.random.randint(-shift, shift)
    return {'rotate':value_rotate, 'blur':value_blur, 'size':value_size, 'shiftX':value_shiftX, 'shiftY':value_shiftY}

def MakeTransform(image, value):
    image = RotateImage(image, value['rotate'], value['size'])
    image = GaussianBlur(image, value['blur'])
    image = MovedImage(image, value['shiftX'], value['shiftY'])
    
    return image
    


dict_value = MakeRandomParams(rotate=30, blur=33, shift = 100)
transform_im = MakeTransform(im, dict_value)

cv2.imshow('image',transform_im)

while(1):
    k = cv2.waitKey(33)
    if k==43:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print (k) # else print its value
