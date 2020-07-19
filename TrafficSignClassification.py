# Libraries
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

# Reading the input images and putting them into a numpy array
data = []
labels = []
files = 43
cur_path = os.getcwd()

for i in range(files):
    path = cur_path + "\Dataset\Train\{0}".format(i)
    #print(path)
    images = os.listdir(path)
    for a in images:
        try:
            p = '\\' + path + '\\' + a
            #print(p)
            image = cv2.imread(p)
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Splitting the images into train and test
print(data.shape, labels.shape)
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

# Converting the labels into one hot encoding
Y_train = to_categorical(Y_train, 43)
Y_test = to_categorical(Y_test, 43)

# # Definition of the CNN model
def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 5)))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    # Compilation of the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

model = build_model()

# # Train the model
X_train = X_train.resize(len(X_train), len(X_train), 1)
X_test = X_test.resize(len(X_test), len(X_test), 1)
history = model.fit(X_train, Y_train, batch_size=64, epochs=15, validation_data=(X_test, Y_test))