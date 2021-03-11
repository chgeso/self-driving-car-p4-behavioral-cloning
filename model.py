import csv
import cv2
import numpy as np

rows = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rows.append(row)

images = []
angle_measurements = []
for row in rows:
    for idx in range(3):
        source_path = row[idx]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = "./data/IMG/" + filename
        image = cv2.cvtColor(cv2.imread(local_path), cv2.COLOR_BGR2RGB)
        images.append(image)
    
    # Create adjusted steering measurements for the side camera images.
    correction = 0.15 # This is a prameter to tune for left and right angle measurements.
    angle_measurement_center = float(row[3])
    angle_measurements.append(angle_measurement_center)
    angle_measurements.append(angle_measurement_center + correction)
    angle_measurements.append(angle_measurement_center - correction)

augmented_images = []
augmented_angle_measurements = []
for image, angle_measurement in zip(images, angle_measurements):
    augmented_images.append(image)
    augmented_angle_measurements.append(angle_measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_angle_measurement = angle_measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_angle_measurements.append(flipped_angle_measurement)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_angle_measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D

model = Sequential()

# To parallelize image normalization.
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# To choose an area of interest that excludes the sky and the hood of the car.
model.add(Cropping2D(cropping=((70,25),(0,0))))

# NVIDIA CNN Architecture
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
