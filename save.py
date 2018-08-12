import numpy as np
import cv2
counter = 0
def saveImage(img):
    global counter
    counter = counter+1
    imgs = str(counter) + '.jpg'
    cv2.imwrite('images/segmented/'+imgs, img)


