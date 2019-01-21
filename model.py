import csv
import cv2
import numpy as np
from scipy import ndimage


samples=[]           # list to store the lines from the csv file
with open('data/driving_log.csv') as csvFile:
    reader=csv.reader(csvFile)     # using csvReader to read the csv file
    headers = next(reader)         # ignore the headings
    for line in reader:            # iterate through the lines and append the line one by one to list.
        samples.append(line)
        
from sklearn.model_selection import train_test_split              
train_samples, validation_samples = train_test_split(samples, test_size=0.2)        # split the training and validation sample. use 20% for the validation sample.

import sklearn
from random import shuffle

# define the generator to be used during training with default batch size of 32.
def generator(samples, batch_size=32):                
    num_samples = len(samples)       # total number of samples collected in csv
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)     # shuffle the sample at the beginning
        for offset in range(0, num_samples, batch_size):    # for th
            batch_samples = samples[offset:offset+batch_size]

            images = []      # list to store the images
            measurements = []   # list to store the steering angle
            for batch_sample in batch_samples:    # for each batch of the samples iterate over each sample 
                for i in range(3):         # iterate for the center , left and right camera images
                    name = 'data/IMG/'+batch_sample[i].split('/')[-1]     # get the absolute file by splitting the given file location url in csv 
                    image = ndimage.imread(name)   # read the image in the RGB format
                    images.append(image)           # add the image to the list
                correction = 0.2          # this is the correction value to be tuned
                steering_centre=float(batch_sample[3])   # retrieve the steering angle from the csv.
                steering_left=steering_centre+correction  # add the correction value to get the steering_left angle
                steering_right=steering_centre-correction  # subtract the correction value to get the steering_right angle
                # add these angles to the lists
                measurements.append(steering_centre)  
                measurements.append(steering_left)
                measurements.append(steering_right)
            
            # create the lists for augmented data
            augmented_images,augmented_measurements=[],[]
            for image,measurement in zip(images,measurements):     # iterate through the dataset
                augmented_images.append(image)   # append the image to the list
                augmented_measurements.append(measurement)    # append the steering angle to the list
                augmented_images.append(np.fliplr(image))     # add the flipped image to the list
                augmented_measurements.append(measurement*-1.0)   # add the corresponding steering value of the flipped image to the list
            X_train=np.array(augmented_images)      # convert the lists to numpy array
            y_train=np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)   # this yields the data

# compile and train the model using the generator function
train_generator = generator(train_samples,batch_size=1)
validation_generator = generator(validation_samples,batch_size=1)

# Setup Keras
from keras.models import Sequential,Model
from keras.layers import Flatten, Dense,Lambda,Cropping2D
from keras.layers.core import  Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# build the model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))   # normalize the image
model.add(Cropping2D(cropping=((70,25), (0,0))))    # crop the image from the up and down portion the image
# convolution layer 1 with (5,5) filter size, followed by subsampling of (2,2) 
model.add(Conv2D(6, 5, 5, subsample=(2, 2),activation='relu'))  # use the RELU activation
# convolution layer 2 with (5,5) filter size,10 output units followed by subsampling of (2,2) 
model.add(Conv2D(10, 5, 5, subsample=(2, 2),activation='relu'))
# convolution layer 3 with (5,5) filter size, 14 output units followed by subsampling of (2,2) 
model.add(Conv2D(14, 5, 5,subsample=(2, 2), activation='relu'))
# convolution layer 4 with (3,3) filter size
model.add(Conv2D(16, 3, 3, activation='relu'))
# convolution layer 5 with (3,3) filter size
model.add(Conv2D(16, 3, 3, activation='relu'))
# flatten the layer
model.add(Flatten())
# fully connected layer with 100 units
model.add(Dense(100))
# fully connected layer with 50 units
model.add(Dense(50))
# fully connected layer with 10 units
model.add(Dense(10))
# final output unit
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')  # use the adam optimizer.
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=2)
model.save('model.h5')     # save the model