import cv2
import preprocessing as pp
import segmentation as sg

def main(image, what):
    inputImage = image
    grayedImage = pp.grayConversion(inputImage)
    print("Image grayed")

    #grayedImage = pp.medFilter(grayedImage)

    binarizedImage = pp.binarization(grayedImage)
    print("Image Binarized")

    heighty, widthx = binarizedImage.shape
    value = sg.segFun(heighty, widthx, binarizedImage, what)

    cv2.waitKey(0)
    return value

