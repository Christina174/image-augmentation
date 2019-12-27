import cv2
import numpy as np
import random

im = cv2.imread('dog.jpg')

def RotateImage(im, angle):
    (h, w) = im.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
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
    
def MakeRandomParams(rotate=None, blur=None, size = None, shift=None, salt=None, color=None, width=None, heigh=None):
    dic_value = {}
    if rotate != None:
        value_rotate =  np.random.randint(-rotate, rotate)
        dic_value["rotate"] = value_rotate
    else:
        dic_value["rotate"] = 0
    if blur != None:
        value_blur =  np.random.randint(1, blur)
        dic_value["blur"] = value_blur
    if size != None:
        value_size =  np.random.random()
        dic_value["size"] = value_size
    else:
        dic_value["size"] = 1
    if shift != None:
        value_shiftX =  np.random.randint(-shift, shift)
        value_shiftY =  np.random.randint(-shift, shift)
        dic_value["shiftX"] = value_shiftX
        dic_value["shiftY"] = value_shiftY
    if salt != None:
        dic_value["salt"] = salt
    if color != None:
        value_colorR = np.random.random()
        value_colorG = np.random.random()  
        value_colorB = np.random.random()       
        dic_value["colorB"] = value_colorB 
        dic_value["colorG"] = value_colorG
        dic_value["colorR"] = value_colorR
    if width != None:
        dic_value["width"] = width
        dic_value["heigh"] = heigh
    return dic_value
#{'rotate':value_rotate, 'blur':value_blur, 'size':value_size, 'shiftX':value_shiftX, 'shiftY':value_shiftY, 'salt':value_salt}

def SaltAndPaper (im, probability):
    # probability of the noise 
    out = np.random.randint(0,probability, im.shape)
    out = out + im
    out[out>255]=255
    out = out.astype(np.uint8)
    return out

def ColorBalance(im, colorB, colorG, colorR):
    out = im.copy()
    out[:,:,0] = im[:,:, 0] * colorB
    out[:,:,1] = im[:,:, 1] * colorG
    out[:,:,2] = im[:,:, 2] * colorR
    out[out>255] = 255
    return out

def SizeImage(im, heigh, width):
    mask = np.zeros((heigh,width,3), dtype = np.uint8)
    if im.shape[0]> im.shape[1]:
        k = float(heigh/im.shape[0])
        resize_img = cv2.resize(im,(int(im.shape[1]*k), heigh), interpolation = cv2.INTER_AREA)
        a = int((width - resize_img.shape[1])/2)
        b = resize_img.shape[1]+a
        mask[:,a:b,:]= resize_img
    else:
        k = float(heigh/im.shape[1])
        resize_img = cv2.resize(im,(width, int(im.shape[0]*k)), interpolation = cv2.INTER_AREA)
        c = int((heigh - resize_img.shape[0])/2)
        d = int(resize_img.shape[0]+c)
        mask [c:d,:, :]= resize_img
    return mask

def MakeTransform(image, value):
    if 'colorB' in value:
        image = ColorBalance(im, value['colorB'], value['colorG'], value['colorR'])
    if 'blur' in value:
        image = GaussianBlur(image, value['blur'])
    image = SaltAndPaper (image, value['salt'])
    if 'width' in value:
        image = SizeImage(image, value['width'], value['heigh'])
    if 'shiftX' in value:
        image = MovedImage(image, value['shiftX'], value['shiftY'])
    image = RotateImage(image, value['rotate'])
    return image
    
#---- main-------------

dict_value = MakeRandomParams(rotate=25, blur=11, shift = 20, salt = 40, color = 30, width=600, heigh= 600)
transform_im = MakeTransform(im, dict_value)

cv2.imshow('image',transform_im)

k = cv2.waitKey(0)
