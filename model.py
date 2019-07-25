import os
import csv
import math
import numpy as np
import cv2
import numpy as np
import sklearn
#import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Lambda, Activation
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D, Cropping2D

BASE_FOLDER = './data_self/'
MEASUREMENTS_FILE = BASE_FOLDER + 'driving_log.csv'
IMAGE_FOLDER = BASE_FOLDER + 'IMG/'

samples = []
with open(MEASUREMENTS_FILE) as csvfile:
    print(MEASUREMENTS_FILE)
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
			


train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32, testData = True):
    num_samples = len(samples)
    correction = 0.1
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = IMAGE_FOLDER +batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
				
                #print(name)
                
                images.append(center_image)
                angles.append(center_angle)
                
                #Data Augmentation
                if (testData == True):		
                    #Add Flipped Image
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle * -1.0)
                    
                    name = IMAGE_FOLDER+batch_sample[1].split('/')[-1]
                    left_image = cv2.imread(name)
                    left_angle = center_angle + correction
                    
                    #images.append(cv2.flip(left_image, 1))
                    #angles.append(left_angle)
                    
                    name = IMAGE_FOLDER+batch_sample[2].split('/')[-1]
                    right_image = cv2.imread(name)	
                    right_angle = center_angle - correction
                    
                    #images.append(cv2.flip(right_image, 1))
                    #angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size, testData = True)
validation_generator = generator(validation_samples, batch_size=batch_size, testData = False)

#ch, row, col = 3, 80, 320  # Trimmed image format
row, col, ch = 160, 320, 3  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/255. - 0.5,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5, activation='relu'))
model.add(Convolution2D(36,5,5, activation='relu'))
#model.add(Convolution2D(48,5,5, activation='relu'))
#model.add(Convolution2D(64,3,3, activation='relu'))
#model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
#model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)
			
model.save("./model.h5")
			
