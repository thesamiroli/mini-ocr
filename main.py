import cv2
import numpy
import preprocessing as pp
import segmentation as sg
import segmentation3 as sg3
import segmentation4 as sg4
import digit_classifier

def main(image, what):
    inputImage = image
    grayedImage = pp.grayConversion(inputImage)
    print("Grayed")
    #medFilteredImage = pp.medFilter(grayedImage)

    binarizedImage = pp.binarization(grayedImage)
    print("binarized")

    heighty, widthx = binarizedImage.shape
    #value = sg3.segFun(heighty, widthx, binarizedImage)
    value = sg4.segFun(heighty, widthx, binarizedImage, what)

    cv2.waitKey(0)
    return value

