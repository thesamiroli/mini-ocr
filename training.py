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

#---------------- 1 Preprocessing our data -----------------

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#The first five rows of our training data
print(train_data.head(5))

#Checking the shape of testing data and training data
print(train_data.shape)
print(test_data.shape)

#Storing pixel array in form of length, width and channel (It includes the pixel values)
features = train_data.iloc[:, 1:].values.reshape(len(train_data), 28, 28, 1)

#Now, storing the label
labels = train_data.iloc[:, 0].values

#Now, Converting labels to categorical features, One-hot encoding.
#Converts a class vector (integers) to binary class matrix.
labels = keras.utils.to_categorical(labels, num_classes=10)

#Converting our features and labels into a numpy array
features = np.array(features)
labels = np.array(labels)

#Lets see the shape of features and labels
print(features.shape)
print(labels.shape)

#Splitting the training data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.20, random_state=1)

#Checking training and testing shape
print(x_train.shape)
print(x_test.shape)

# ----------------------- 2 Creating our model -------------------------

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

model = LeNet(28, 28, 1, 10)

#If you want to look at the details of the model
#model.summary()

# --------------------- 3 Training our model ----------------
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
epochs = 20
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)

#Saving and loading the saved weights
model.save_weights('cnn_mnist.h5')
model.load_weights('cnn_mnist.h5')


# ---------------------4 Testing our model ---------------
test_data.head(5)

test_set = test_data.iloc[:, :].values.reshape(len(test_data), 28, 28, 1)
print(test_set.shape)

prediction = model.predict(test_set)
prediction = np.argmax(prediction, axis=1)
print(prediction)

#Writing the prediction into a csv file
df = pd.DataFrame({
    'S.N.' : list(range(1,len(prediction)+1)),
    'Output' : prediction
})

df.to_csv("prediction.csv", index=False, header=True)
# -------------------- 4.1 Testing a single value --------------
#test_x = test_data.iloc[24, :].values.reshape(1,28, 28, 1)
testFile = cv2.imread("1.jpg", 0)

testFile2 = cv2.imread("2.jpg", 0)

testFile3 = cv2.imread("3.jpg", 0)

#testFile4 = cv2.imread("b.jpg", 0)

testFile = testFile.reshape(1, 28, 28, 1)
single_prediction = model.predict(testFile)
print("Your Predicted Number is: {}".format(np.argmax(single_prediction, axis=1)))

testFile2 = testFile2.reshape(1,28,28,1)
single_prediction = model.predict(testFile2)
print("Your Predicted Number is: {}".format(np.argmax(single_prediction, axis=1)))

testFile3 = testFile3.reshape(1,28,28,1)
single_prediction = model.predict(testFile3)
print("Your Predicted Number is: {}".format(np.argmax(single_prediction, axis=1)))
'''
testFile4 = testFile4.reshape(1,28,28,1)
single_prediction = model.predict(testFile4)
print("Your Predicted Number is: {}".format(np.argmax(single_prediction, axis=1)))
'''


'''
test_x = test_data.iloc[24, :].values
test_x = np.array(test_x)
test_x = test_x.reshape(28, 28)
plt.imshow(test_x, cmap='gray');
'''


