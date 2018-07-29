import numpy as np
import cv2

def segment(binarizedImage):
    #getting the contours
    _, contours, _ = cv2.findContours(binarizedImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #initializing a 2-d list (rows = no. of characters, cols = pixel values of each characters)
    features = np.random.rand(len(contours), 784)


    idx = 0
    for c in contours:

        # get the x,y coordinate , height and width of the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a rectangle to visualize the bounding rect
        cv2.rectangle(binarizedImage, (x-8, y-8), (x + w+16, y + h+16), (0, 255, 0), 1)
        roi = binarizedImage[y-8:y + h+16, x-8:x + w+16]
        print(roi.shape)

        #giving name to images
        imgs = str(idx) + '.jpg'
        imgs2 = str(idx) + '-resized.jpg'

        #write the image, resize the region of interest, write the resized image again
        cv2.imwrite(imgs, roi)
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_CUBIC)
        roi = cv2.bitwise_not(roi, roi)
        cv2.imwrite(imgs2, roi)
        print(roi.shape)

        #reading the  resized image back - features haru extract garna
        ft = cv2.imread(imgs2, 0)
        ft = ft.flatten()
        features[idx] = ft

        idx += 1


    #randomly check the features of a character
    print(features[0])


    contour = str(len(contours))
    print("Numbers of characters found : " +contour)

    cv2.imshow( " Segmented Image ", binarizedImage)

    cv2.waitKey(0)


