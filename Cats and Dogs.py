import tensorflow as tf
import pickle
import random
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

DATADIR = "Pets"
CATEGORIES = ["Dog", "Cat"]

# Trying to normalize data, image size
training_data = []
IMG_SIZE = 50


"""def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        # Iterate through images in folder of pets
        for img in os.listdir(path):
            # Convert to greyscale, simple data set information
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()
# print(len(training_data))
random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

# Feature set
X = []

# Labels
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
    # print(label)

# Saving normalized data to prevent above from rerunning
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

y = np.array(y)
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()"""

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1)
