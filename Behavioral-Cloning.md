# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Behavioral-Cloning.md file summarizing the result (this file)
* video.mp4 which is the recording of the autonomous driving for the first test track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I replicated the NVIDA model with some minor adjustments to use for this project as the use case and data were similar. 
The final model looks like below
    Image normalization (Simple normalization using Lambda function)
    Convolution: 5x5, filter: 24, strides: 2x2, activation: elu
    Convolution: 5x5, filter: 36, strides: 2x2, activation: elu
    Convolution: 5x5, filter: 48, strides: 2x2, activation: elu
    Convolution: 3x3, filter: 64, strides: 1x1, activation: elu
    Convolution: 3x3, filter: 64, strides: 1x1, activation: elu
    Flatten Layer
    Fully connected: neurons: 100, activation: elu
    Dropout
    Fully connected: neurons: 50, activation: elu
    Dropout
    Fully connected: neurons: 10, activation: ELU
    Dropout
    Fully connected: neurons: 1 (output)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting for all the fully connected layers. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. 

#### 4. Appropriate training data

I combined the sample training data provided with my own data captured using the local instance of the simulator. I captured about 4 laps of (including some partial) central lane driving data, about 2 laps of opposite side driving data, 1+ instances of driving around turns and 1+ instances of driving in the mountain track (this was the one available in the local instance of the simulator). I captured some more data than required to compensate for all the biases as well to improve the model considering my lousy simulator driving. 
 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

As discussed above I started with replicating the NVIDA model as is with just the first normalization layer added. I modified the activation function to elu from relu since it was suggested to be more quick in training. I also added dropouts for the fully connected layer to reduce overfitting which was happening in the initial test runs. 

#### 2. Final Model Architecture

Here is how the final model looks
    Image normalization (Simple normalization using Lambda function)
    Convolution: 5x5, filter: 24, strides: 2x2, activation: elu
    Convolution: 5x5, filter: 36, strides: 2x2, activation: elu
    Convolution: 5x5, filter: 48, strides: 2x2, activation: elu
    Convolution: 3x3, filter: 64, strides: 1x1, activation: elu
    Convolution: 3x3, filter: 64, strides: 1x1, activation: elu
    Flatten Layer
    Fully connected: neurons: 100, activation: elu
    Dropout
    Fully connected: neurons: 50, activation: elu
    Dropout
    Fully connected: neurons: 10, activation: ELU
    Dropout
    Fully connected: neurons: 1 (output)


#### 3. Creation of the Training Set & Training Process

As discussed above I used the combination of my own recorded data and the sample data provided to compensate for inherent biases and to compensate for my bad driving in the simulator. 

As suggested in the lessons and NVIDA paper I added the below preprocessing steps. 
   -Crop the image to avoid the top part of the images
   -Resized the image to the NVIDA model input size
   -Converted the image to YUV space

(Own training data was captured in the local system since online simulator was extremely slow for me. This was stored on Google Drive and loaded before training using dl.sh and dl_img.sh scripts. IMG data was unzipped before use)

Since the data was huge in size I used a generator function (again from the suggestions in the lessons) to feed data in batches for training, validation and testing. 

I split the data initially into training and testing sets. I then split the training data to get an validation set. 

The generator in addition to calling the preprocessing of the images also introduced random flipping on the fly for training data. (Spent considerable amount of time on this step as a bug was causing my code to load all the data instead of batch size)

Multiple iterations were done with the changing the epoch #s, batch size to arrive at what I thought was the best model. Initial autonomous run resulted in car crashing into the waters. More fine tuning was done before arriving at the current model which was able to drive the first test track successfully. (There is still lot more improvement to make on this one as we can see from the video. The model did not work well on the jungle track. I need to feed some more data and see if the model will learn this one.)
