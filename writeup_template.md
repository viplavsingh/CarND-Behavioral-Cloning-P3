# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_driving.jpg "center driving"
[image3]: ./examples/recover_from_left.jpg "Recovery Image from left"
[image4]: ./examples/recover_from_left1.jpg "Recovery Image from left "
[image5]: ./examples/recover_from_right.jpg "Recovery Image from right"
[image6]: ./examples/original_image.jpg "Normal Image"
[image7]: ./examples/flipped_image.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
Initially I tried the lenet architecture for the problem. The lenet based model performed well on the track 1 in the simulator but performed poorly on the track 2.
Then I thought of using the more powerful model architecture i.e. Nvidia based architecture. But since this architecture had too many output units at each layer. I made some modification
to the architecture which had less number of output units and satisfies the problem requirement. This still does not perform good on track 2 but performs better than lenet.
The final model architecture I have used is the modification of Nvidia based architecture. 

My final model consisted of the following layers:
 
Input is 160x320x3 RGB image.
1. Lambda Layer: 160x320x3 RGB image is preprocessed by dividing the pixel vlaues by 255 and then subtracting 0.5.(i.e (x / 255.0) - 0.5).
2. Cropping layer: Image is then cropped from up and down to help the model not distract itself. Cropping2D(cropping=((70,25), (0,0)))
3. Convolution layer 1: filter of (5x5) size with 6 output units.
	i. Activation layer: Relu
	ii.Subsampling: stride of (2x2) with same padding.
4. Convolution layer 2: filter of (5x5) size with 10 output units.
	i. Activation layer: Relu
	ii. Subsampling: stride of (2x2) with same padding.
5. Convolution layer 3: filter of (5x5) size with 14 output units.
	i. Activation layer: Relu
	ii. Subsampling: stride of (2x2) with same padding.
6. Convolution layer 4: filter of (3x3) size with 16 output units.
	i. Activation layer: Relu
5. Convolution layer 3: filter of (3x3) size with 16 output units.
	i. Activation layer: Relu
6. fully connected layer1: outputs with 100 units
7. fully connected layer2: outputs with 50 units
8. fully connected layer 3: outputs with 10 units.
9. 1 final output. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

I have not used the dropout layers in the model since the model is not overfitting. The training and validation losses are low and within the same range. I have used the udacity data to
train the model. I also used the training data collected by me to ensure that model was not overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I used the training data provided by the udacity to train the model.
But I also collected the training data using the simulator to see how the model architecure performs and it does not overfit.
Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
I used 2 laps of center lane driving and 1 lap of recovering from the left and right sides of the road.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the lenet based architecture. It worked well on the track 1 in the simulator but performed poorly on the track 2 
data. So, idea was to create a more generalized model which can work on any track. So I moved to more powerful model architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set using train_test_split() function.
Then I used the nvidia based model to train on these training images. I noticed the training and validation loss were not reducing after 3 epochs.
So I used an epoch of 3.

Then to check whether the model architecture is not overfitting, I used other dataset. It was not overfitting.

The final step was to run the simulator to see how well the car was driving around track one. After some time, I noticed car went off the track. The I used the left and right camera images
also in the dataset and also augmented the data by flipping.   

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

1. Lambda Layer: 160x320x3 RGB image is preprocessed by dividing the pixel vlaues by 255 and then subtracting 0.5.(i.e (x / 255.0) - 0.5).
2. Cropping layer: Image is then cropped from up and down to help the model not distract itself. Cropping2D(cropping=((70,25), (0,0)))
3. Convolution layer 1: filter of (5x5) size with 6 output units.
	i. Activation layer: Relu
	ii.Subsampling: stride of (2x2) with same padding.
4. Convolution layer 2: filter of (5x5) size with 10 output units.
	i. Activation layer: Relu
	ii. Subsampling: stride of (2x2) with same padding.
5. Convolution layer 3: filter of (5x5) size with 14 output units.
	i. Activation layer: Relu
	ii. Subsampling: stride of (2x2) with same padding.
6. Convolution layer 4: filter of (3x3) size with 16 output units.
	i. Activation layer: Relu
5. Convolution layer 3: filter of (3x3) size with 16 output units.
	i. Activation layer: Relu
6. fully connected layer1: outputs with 100 units
7. fully connected layer2: outputs with 50 units
8. fully connected layer 3: outputs with 10 units.
9. 1 final output.


#### 3. Creation of the Training Set & Training Process

I used 2 laps of the center lane driving and 1 lap of recovering from the left and right sides. 
if the training data is all focused on driving down the middle of the road, the model won’t ever learn what to do if it gets off to the side of the road.
So to capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to what to do if it gets off to the side of the road. These images show what a recovery looks like :

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would help in generalising the model. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 7904  number of data points. After taking the images from the left and right camera and flipping it became 47424. I then preprocessed this data by 
using the lambda layer of the model (x / 255.0) - 0.5. After the normalization, the images are cropped from the up and down portion of the image so that the model doesnot distract itself. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the saturating training and validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
