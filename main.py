import os
import cv2.cv2 as cv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras_preprocessing import image
from keras.utils import np_utils
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from skimage.transform import resize
from sklearn.model_selection import train_test_split

assetsPath = "./assets"
framesPath = assetsPath + "/frames"
videosPath = assetsPath + "/videos"
csvPath = assetsPath + "/csv"

csvFile = csvPath + "/mapping.csv"

if os.path.isdir(framesPath) is not True:
    os.mkdir(framesPath)

count = 0
videoFile = videosPath + "/Tom and jerry.mp4"
cap = cv.VideoCapture(videoFile)
frameRate = cap.get(5)
x = 1

if os.path.isfile(framesPath + "/frame0.jpg") is not True:
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if ret is not True:
            break
        if frameId % math.floor(frameRate) == 0:
            filename = framesPath + "/frame%d.jpg" % count;
            count += 1
            cv.imwrite(filename, frame)
    cap.release()
    print("Fin du découpage de la vidéo 1")

img = plt.imread(framesPath + "/frame0.jpg")

data = pd.read_csv(csvFile)
data.head()

X = []
for img_name in data.Image_ID:
    img = plt.imread(framesPath + '/' + img_name)
    X.append(img)
X = np.array(X)

y = data.Class
dummy_y = np_utils.to_categorical(y)

image = []
for i in range(0, X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224, 224)).astype(int)
    image.append(a)
X = np.array(image)

X = preprocess_input(X)
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)

X_train = X_train.reshape(208, 7*7*512)
X_valid = X_valid.reshape(90, 7*7*512)

train = X_train / X_train.max()
X_valid = X_valid / X_train.max()

# Création du model
model = Sequential()
model.add(InputLayer((7*7*512,)))
model.add(Dense(units=1024, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.summary()

# Compilation du model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrainement du model
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))