import cv2
import time
import imtransform as imt

#---- main-------------
count = 0
im = cv2.imread('dog.jpg')
#print(im.dtype)
start_time = time.process_time()
#im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
try:
    while True:
        dict_value = imt.MakeRandomParams(im, rotate=30, blur = 10, shift = 20, salt=50, color=5, size = (500,500), flip=True)
        
        image = imt.MakeTransform(im, dict_value)
        count +=1

        cv2.imshow('image',image)

        k = cv2.waitKey(0)
        if k == 27:
            break
    print ('Среднее время выполнения операций = %.3f сек.'% ((time.process_time() - start_time)/count))
except ValueError as e:
    print(e)
