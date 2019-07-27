# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center_2019_07_25_19_34_50_080.jpg "Grayscaling"
[image3]: ./examples/center_2019_07_27_18_34_22_466.jpg "Recovery Image"
[image4]: ./examples/center_2019_07_27_18_34_22_606.jpg "Recovery Image"
[image5]: ./examples/center_2019_07_27_18_34_22_745.jpg "Recovery Image"
[image6]: ./examples/center_2019_07_25_19_34_50_080.jpg "Normal Image"
[image7]: ./examples/Figure_1.png "Flipped Image"

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

My model consists of a 2 Layers of convolution neural network with 5x5 kernel sizes and depths between 24 and 36 (model.py lines 95-96) 
Followed by Another 2 layers of convolution neural network with kernel sizes 3 X 3 and depths 48 and 64.
Maxpooling layer is deployed.

The model includes RELU layers to introduce nonlinearity (code line 95-100), and the data is normalized in the model using a Keras lambda layer (code line 91). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 2 Dropout layers have been deployed to reduce overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 107).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, flippping the image and using right and left camera. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia I thought this model might be appropriate because it is widely used in self driving cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

Then I augmented data by flipping image and using left and right camera.

The final step was to run the simulator to see how well the car was driving around track one. I have used dataset provided by udacity for track1. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I added data set for recovery.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 88-108) consisted of a convolution neural network with the following layers and layer sizes 
* Normalization
* Crop image
* Convolution Layer Number of Convolution Filter(24)
* Canvolution Layer Number of Convolution Filter(36)
* Dropout (30%)
* Convolution Layer Number of Convolution Filter(48)
* Maxpooling
* Canvolution Layer Number of Convolution Filter(64)
* Maxpooling
* Dropout (30%)
* Flatten
* Dense Size(100)
* Dense Size(50)
* Dense Size(10)
* Dense Size(1)



#### 3. Creation of the Training Set & Training Process

I have used the data set provided by the Udacity to train my model. However there was issues while recovery. Have added few recordings for recovery.

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to track. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would provide more datapoints and hence help in training my model better. Have also used right and left camera images for data augmentation. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]



After using data augmentation and using left and right camera images I had 43328 data points. I then preprocessed this data by cropping the data and normalizing the image.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the video. I used an adam optimizer so that manually training the learning rate wasn't necessary.
