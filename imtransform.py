import cv2
import numpy as np
import random

def RotateImage(im, M):
    (h, w) = im.shape[:2]
    #поворачиваем и сдвигаем исходное изображение к матрице преобразования
    rotated_image = cv2.warpAffine(im, M, (w, h))
    return rotated_image

def BoxFilter(im, ksize):
    BoxFilter_image = cv2.boxFilter(im, 0, (ksize,ksize))
    return BoxFilter_image

def SaltAndPaper (im, probability):
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
    # если изображение черно-белое, то:
    if len(im.shape) == 2:
        mask_shape = (heigh,width)
    # прописываем dtype тип элементов массива изображения
    mask = np.zeros(mask_shape, dtype = np.uint8)
    #mask[:] = 128
    if im.shape[1] / im.shape[0] < width/heigh  :
        k = float(heigh/im.shape[0])
        resize_img = cv2.resize(im,(int(im.shape[1]*k), heigh), interpolation = cv2.INTER_AREA)
        a = int((width - resize_img.shape[1])/2)
        b = resize_img.shape[1]+a
        mask[:,a:b]= resize_img
    else:
        k = float(width/im.shape[1])
        resize_img = cv2.resize(im,(width, int(im.shape[0]*k)), interpolation = cv2.INTER_AREA)
        c = int((heigh - resize_img.shape[0])/2)
        d = int(resize_img.shape[0]+c)
        mask [c:d,:]= resize_img
    return mask

# записываем в словарь случайные значения трансформации изображения, либо None (по умолчанию)
def MakeRandomParams(image, rotate=None, blur=None, shift=None, salt=None, color=None, size=None, flip = False):
    dic_value = {}

    if len(image.shape) == 2 and color != None:
        raise ValueError('Уберите "color", т.к. для черно-белого изображения нельзя изменить цвет')
        
    if rotate != None:
        value_rotate = np.random.randint(-rotate, rotate)
    else:
        value_rotate = 0
    if blur != None:
        dic_value["blur"] =  np.random.randint(1, blur)

    if shift != None:
        value_shiftX =  np.random.randint(-shift, shift)
        value_shiftY =  np.random.randint(-shift, shift)
    else:
        value_shiftX=0
        value_shiftY=0
    if color != None:
        dic_value["colorB"] = np.random.random()
        dic_value["colorG"] = np.random.random()  
        dic_value["colorR"] = np.random.random()       

    if size != None:
        dic_value["size"] = size
    if flip != False:
        dic_value["flip"] =  np.random.randint(0, 2)
    if salt != None:
        dic_value["salt"] = np.random.randint(0,salt, image.shape)
    if rotate != None or shift != None:
        center = (size[0] / 2, size[1] / 2)
        # матрица с определением центра и угла поворота (масштаб не меняем)
        M = cv2.getRotationMatrix2D(center, value_rotate, 1)
        #  |1 0 y|
        #  |0 1 x|
        M[0][2] += value_shiftY
        M[1][2] += value_shiftX
        dic_value["rotateAndMoved"] = M
    return dic_value

# делаем преобразования изображения
def MakeTransform(image, value):
    if 'flip' in value:
        if value['flip'] == 1:
            image = cv2.flip(image, 1)
    if 'colorB' in value:
        image = ColorBalance(image, value['colorB'], value['colorG'], value['colorR'])
    if 'blur' in value:
        image = BoxFilter(image, value['blur'])
    if 'salt' in value:
        image = SaltAndPaper (image, value['salt'])
    if 'size' in value:
        image = SizeImage(image, value['size'][1], value['size'][0])
        print(value['size'][1], value['size'][0])
    if 'rotateAndMoved' in value:
        image = RotateImage(image, value['rotateAndMoved'])    
        
    return image
