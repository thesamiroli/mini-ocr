import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

predictedDigit = " "
def LeNet(width, height, channels, output):
    model = Sequential()

    # Convulation
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), input_shape=(width, height, channels)))

    # ReLU Activation
    model.add(Activation('relu'))

    # Pooling
    model.add(MaxPool2D(pool_size=(2, 2)))

    # Convolution
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2)))

    # ReLU Activation
    model.add(Activation('relu'))

    # Pooling
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    # Hidden Layer
    model.add(Dense(100))

    model.add(Activation('relu'))

    model.add(Dense(output))

    model.add(Activation('softmax'))

    return model

def classify(image):
    global predictedDigit
    testFile = image
    testFile = testFile.reshape(1, 28, 28, 1)
    single_prediction = model.predict(testFile)
    #print("Your Predicted Number is: {}".format(np.argmax(single_prediction, axis=1)))
    predictedDigit = predictedDigit + " "+ str(np.argmax(single_prediction, axis=1)[0])
    print("Predicted digit : ", predictedDigit)
    return predictedDigit


model = LeNet(28, 28, 1, 10)


model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.load_weights('cnn_mnist.h5')

