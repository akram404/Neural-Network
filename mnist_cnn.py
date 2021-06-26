# this Convolutional neural network is for mnist hand-written digits dataset
import numpy as np
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPool2D

(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX/255.0

trainX = np.expand_dims(trainX, axis=-1)
print(trainX.shape)
print(trainY.shape)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation="relu", input_shape = (28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation="relu"))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation="relu", input_shape = (28,28,1)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(10,activation="softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model.fit(trainX, trainY, epochs=10, validation_split=0.1)
