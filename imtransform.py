import cv2
import numpy as np
import random
import time

def RotateImage(im, M):
    (h, w) = im.shape[:2]
    rotated_image = cv2.warpAffine(im, M, (w, h))
    return rotated_image

def GaussianBlur(im, blur):
    if blur%2 ==0:
        blur = blur+1
    GaussianBlur_image = cv2.GaussianBlur(im, (blur,blur), 0)
    return GaussianBlur_image

def BoxFilter(im, ksize):
    BoxFilter_image = cv2.boxFilter(im, 0, (ksize,ksize))
    #print(ksize)
    return BoxFilter_image

def MovedImage(im, shiftX, shiftY):
    num_rows, num_cols = im.shape[:2]
    translation_matrix = np.float32([ [1,0,shiftY], [0,1,shiftX] ])
    shift_image = cv2.warpAffine(im, translation_matrix, (num_cols,num_rows))
    return shift_image
    
def MakeRandomParams(rotate=None, blur=None, boxFilter = None, shift=None, salt=None, color=None, width=None, heigh=None, flip = False):
    dic_value = {}
    value_shiftX=0
    value_shiftY=0
    if rotate != None:
        value_rotate = np.random.randint(-rotate, rotate)
    else:
        value_rotate = 0
        
    if blur != None:
        value_blur =  np.random.randint(0, blur)
        dic_value["blur"] = value_blur
    if boxFilter != None:
        value_boxFilter =  np.random.randint(1, boxFilter)
        dic_value["boxFilter"] = value_boxFilter
    if shift != None:
        value_shiftX =  np.random.randint(-shift, shift)
        value_shiftY =  np.random.randint(-shift, shift)
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
    if flip != False:
        value_flip =  np.random.randint(0, 2)
        dic_value["flip"] = value_flip
    if salt != None:
        dic_value["salt"] = np.random.randint(0,salt, im.shape)
    if rotate != None or shift != None:
        center = (width / 2, heigh / 2)
        M = cv2.getRotationMatrix2D(center, value_rotate, 1)
        M[0][2] += value_shiftY
        M[1][2] += value_shiftX
        dic_value["rotateAndMoved"] = M
    return dic_value
#{'rotate':value_rotate, 'blur':value_blur, 'size':value_size, 'shiftX':value_shiftX, 'shiftY':value_shiftY, 'salt':value_salt}

def SaltAndPaper (im, probability):
    # probability of the noise 
    #out = np.random.randint(0,probability, im.shape)
    out = probability + im
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
    mask_shape = (heigh,width,3)
    if len(im.shape) == 2:
        mask_shape = (heigh,width)
    mask = np.zeros(mask_shape, dtype = np.uint8)
    if im.shape[0]> im.shape[1]:
        k = float(heigh/im.shape[0])
        resize_img = cv2.resize(im,(int(im.shape[1]*k), heigh), interpolation = cv2.INTER_AREA)
        a = int((width - resize_img.shape[1])/2)
        b = resize_img.shape[1]+a
        mask[:,a:b]= resize_img
    else:
        k = float(heigh/im.shape[1])
        resize_img = cv2.resize(im,(width, int(im.shape[0]*k)), interpolation = cv2.INTER_AREA)
        c = int((heigh - resize_img.shape[0])/2)
        d = int(resize_img.shape[0]+c)
        mask [c:d,:]= resize_img
    return mask

def MakeTransform(image, value):
    start_time = time.process_time()
    if 'flip' in value:
        if value['flip'] == 1:
            image = cv2.flip(image, 1)
            #print ((time.process_time() - start_time), '=flip')
    start_time = time.process_time()
    if 'colorB' in value:
        image = ColorBalance(image, value['colorB'], value['colorG'], value['colorR'])
        #print ((time.process_time() - start_time), "=color")
    start_time = time.process_time()
    if 'blur' in value:
        image = GaussianBlur(image, value['blur'])
        #print ((time.process_time() - start_time), "=blur")
    
    start_time = time.process_time()
    if 'boxFilter' in value:
        image = BoxFilter(image, value['boxFilter'])
        #print ((time.process_time() - start_time), "=boxFilter")
    
    start_time = time.process_time()
    if 'salt' in value:
        image = SaltAndPaper (image, value['salt'])
        #print ((time.process_time() - start_time), "=salt")
    
    #start_time = time.process_time()
    if 'width' in value:
        image = SizeImage(image, value['width'], value['heigh'])
        #print ((time.process_time() - start_time), "=width")    
    
    #if 'shiftX' in value:
        #image = MovedImage(image, value['shiftX'], value['shiftY'])
        ##print ((time.process_time() - start_time), "=Moved")
    start_time = time.process_time()
    if 'rotateAndMoved' in value:
        image = RotateImage(image, value['rotateAndMoved'])    
        
    return image
    
#---- main-------------
count = 0
im = cv2.imread('dog.jpg')
start_time = time.process_time()
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
while True:
    #dict_value = MakeRandomParams(rotate=None, blur=None, shift=40, salt=150, color=None, width=224, heigh=224, flip=True)
    
    
    dict_value = MakeRandomParams(rotate=45, blur=None, boxFilter = None, shift = 20, salt=150, color=None, width=224, heigh=224, flip=True)
    
    image = MakeTransform(im, dict_value)
    count +=1


    cv2.imshow('image',image)

    k = cv2.waitKey(30)
    if k == 27:
        break
print ((time.process_time() - start_time)/count, "seconds")
    #print(count, 'count')
