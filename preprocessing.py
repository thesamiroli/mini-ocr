import numpy as np
import cv2

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
    cv2.imshow("Median Filter",img_out)
    return img_out

def binarization(image):
    imger = image.copy()
    height, width = image.shape
    value = (np.sum(imger)/np.size(image))
    for i in range(0, height):
        for j in range(0, width):
            if imger[i][j]<= 110:
                imger[i][j]=0
            else:
                imger[i][j]=255
    return imger