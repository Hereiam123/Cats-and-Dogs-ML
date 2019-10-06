import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

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

# Saving normalized data to prevent above from rerunning
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()"""
