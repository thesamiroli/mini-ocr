import numpy as np
import cv2
from PIL import Image, ImageEnhance
import save

def grayConversion(inputImage):
    grayValue = 0.07 * inputImage[:, :, 2] + 0.72 * inputImage[:, :, 1] + 0.21 * inputImage[:, :, 0]
    gray = grayValue.astype(np.uint8)
    return gray


def medFilter(image):
    img_out = image.copy()
    height, width= image.shape
    for i in np.arange(1, height - 1):
        for j in np.arange(1, width - 1):
            neighbors = list()
            for k in np.arange(-1, 2):
                for l in np.arange(-1, 2):
                    #a = image[i+k][j+l]
                    a = np.mean(image[i+k][j+l])
                    neighbors.append(a)
            #print(neighbors)
            neighbors.sort()
            median = neighbors[4]
            b = median
            img_out[i][j] = b
    print("Med fil")
    return img_out

def binarization(image):
    ret, imgf = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgf
