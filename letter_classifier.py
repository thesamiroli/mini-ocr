from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import numpy as np
import set_character as sc

predictedValue = " "
def LeNet():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(26, activation='softmax'))
    return model

def opToChar(output):
    value = sc.convert(output)
    return value

def classify(testFile):
    global predictedValue
    testFile = testFile.reshape(1, 28, 28, 1)
    single_prediction = model.predict(testFile)
    predictedValue = predictedValue + " " + opToChar((np.argmax(single_prediction, axis=1)[0]))
    print("Predicted letter : ", predictedValue)
    return predictedValue

model = LeNet()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('weights/my_model_weights2.h5')

