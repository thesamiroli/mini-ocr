import cv2
import numpy
import preprocessing as pp

import segmentation as sg
import digit_classifier

def main(image, what):
    inputImage = image
    grayedImage = pp.grayConversion(inputImage)
    print("Grayed")
    #medFilteredImage = pp.medFilter(grayedImage)

    binarizedImage = pp.binarization(grayedImage)
    print("binarized")

    heighty, widthx = binarizedImage.shape
    value = sg.segFun(heighty, widthx, binarizedImage, what)

    cv2.waitKey(0)
    return value

