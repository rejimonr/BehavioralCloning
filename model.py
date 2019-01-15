###This will load the collected data, build the model, train the model and save it for use later.
##Heavy use of code and ideas from the lessons and NVIDA paper

#Load all the needed modules
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
from os import listdir
import cv2
import tensorflow as tf

##Preprocess method to crop, resize and convert to YUV space
def preprocess(img):
    #Crop the top part
    new_img = img[50:140,:,:]
    #print("cropping")
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    #print("resizing")
    #Convert to YUV space
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2YUV)
    return new_img

##Flip randomly and adjust the angle accordingly
def random_flip(img,angle):
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        angle = -angle
    return img,angle

##Generator to get the on the fly data in batches
##Takes in the loaded image and driving data log to yield batches
##Does the preprocessing of images and random flipping
def generate_data(image_paths, angles, batch_size=128,validation=False):
    #print("gen called",validation)
    image_paths, angles = shuffle(image_paths, angles)
    X,y = ([],[])
    while True:       
        for i in range(len(angles)):
            img = mpimg.imread(image_paths[i])
            angle = angles[i]
            img = preprocess(img)
            if not validation:
                img, angle = random_flip(img, angle)
            X.append(img)
            y.append(angle)
            if len(X) == batch_size:
                #print(len(X),"yielding",validation)
                yield (np.array(X), np.array(y))
                X, y = ([],[])
                image_paths, angles = shuffle(image_paths, angles)


    

####ENSURE OWN DATA IS UPLOADED TO THE BELOW FOLDER IF NEEDED####
'''Recording of the data from online simulator was extremely slow. Based on suggestions from the group I donwloaded the simulator and 
recorded the data in my local system. This data had to be uploaded into the /opt/ filesystem consdering the size. 
This data gets wiped off if we leave the classroom. Before using the own data we need to run the scripts dl.sh and dl_img.sh to get the data.
The IMG data has to be unzipped as well.
'''
####

base_data_dir='/opt/carnd_p3'
#base_data_dir='/home/workspace/CarND-Behavioral-Cloning-P3'
own_data_path='/own/'
udacity_data_path = '/data/'
csv_path = ['/opt/carnd_p3/own/driving_log.csv', '/opt/carnd_p3/data/driving_log.csv']
data_path = ['/own/IMG/', '/data/IMG/']
url_split = ['\\','/'] #Had to have this hack because my local system had windows paths

image_paths = []
angles = []

#Added this switch to control which data to pick. 
#Used only udacity data originally to code up the model since there were issues in recording own data and uploading it
#If True then the appropriate data is included. First value is for own data and second for udacity sample data
use_own_data = [True, True]

#print(listdir('/opt'))
print("udacity data",listdir(base_data_dir + udacity_data_path))
if use_own_data[0] == True : print("own data",listdir(base_data_dir + own_data_path))

#debug = 1;

##Combine the Udacity sample data and own data to get the samples


for j in range(2):
    if use_own_data[j] == False: #Skip appropriately 
        continue
    # Import driving data from csv files
    with open(csv_path[j], newline='') as f:
        driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=',', quoting=csv.QUOTE_NONE))

    # Gather data - image paths and angles for center, left, right cameras in each row
    for row in driving_data[1:]:
        # get center image path and angle
        file_name = row[0].split(url_split[j])[-1]
        image_paths.append(base_data_dir + data_path[j] + file_name)
        angles.append(float(row[3]))
        #if debug ==1:
            #print(row[0],file_name,base_data_dir + data_path[j] + file_name)
            #debug = 2;
        # get left image path and angle and add adjustment to the angle values
        file_name = row[1].split(url_split[j])[-1]
        image_paths.append(base_data_dir + data_path[j] + file_name)
        angles.append(float(row[3])+0.25)
        # get left image path and angle
        file_name = row[2].split(url_split[j])[-1]
        image_paths.append(base_data_dir + data_path[j] + file_name)
        angles.append(float(row[3])-0.25)

image_paths = np.array(image_paths)
angles = np.array(angles)
print("shapes", image_paths.shape,angles.shape)
print(image_paths[0], angles[0])

# split into train/test sets and then get a smaller validation set
image_paths_train, image_paths_test, angles_train, angles_test = train_test_split(image_paths, angles,
                                                                                  test_size=0.05)
image_paths_train, image_paths_val, angles_train, angles_val = train_test_split(image_paths_train,angles_train,test_size=0.2)

print('Train:', image_paths_train.shape, angles_train.shape)
print('Test:', image_paths_test.shape, angles_test.shape)


##Build the model
##Replicated the NVIDA model as is with the addition of elu activation functions and dropouts for the fully connected layers

model = Sequential()
##Normalization layer
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))
# Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride and add two 3x3 convolution layers (output depth 64, and 64)
model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='elu'))
model.add(Conv2D(64, (3, 3), activation='elu'))
# Add a flatten layer
model.add(Flatten())
# Add three fully connected layers (depth 100, 50, 10) and dropouts
model.add(Dense(100, activation='elu'))
model.add(Dropout(0.50))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.50))
model.add(Dense(10, activation='elu'))
model.add(Dropout(0.50))

# Add a fully connected output layer
model.add(Dense(1))

#Build the model with adam optimizer and mse as the loss function. Again credits to the class room lessons
model.compile(optimizer='adam', loss='mse')

##Train the model and compare loss
bsize = 64  #Parameter which was used for tuning

#Generate the training, validation and test data using the generators
train_gen = generate_data(image_paths_train, angles_train, validation=False, batch_size=bsize)
val_gen = generate_data(image_paths_val, angles_val, validation=True, batch_size=bsize)
test_gen = generate_data(image_paths_test, angles_test, validation=True, batch_size=bsize)

##Added to save the models after each epoch
checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

#train and get the history saved. Epochs was tuned
history = model.fit_generator(train_gen, validation_data=val_gen, validation_steps=int(len(angles_val)/bsize), steps_per_epoch=int(len(angles_train)/bsize), epochs=5, verbose=1, callbacks=[checkpoint])
print('Test Loss:', model.evaluate_generator(test_gen, 128))
#print(model.summary())


##Save the model for use by drive.py later
model.save('./model.h5')
