import cv2
import numpy
import preprocessing as pp
import segmentation as sg

inputImage = cv2.imread("ts.jpg")

grayedImage = pp.grayConversion(inputImage)

#medFilteredImage = pp.medFilter(grayedImage)

binarizedImage = pp.binarization(grayedImage)

cv2.imshow("Binarized image ", binarizedImage)

heighty, widthx = binarizedImage.shape

sg.segment(binarizedImage)

cv2.waitKey(0)