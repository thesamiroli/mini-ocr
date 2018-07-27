import cv2
import numpy
import preprocessing as pp

inputImage = cv2.imread("smalltext.jpg")

grayedImage = pp.grayConversion(inputImage)

medFilteredImage = pp.medFilter(grayedImage)

binarizedImage = pp.binarization(medFilteredImage)

cv2.imshow("Binarized image ", binarizedImage)
cv2.waitKey(0)