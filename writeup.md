# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/network.png "Model Visualization"
[image2]: ./images/center_2016_12_01_13_30_48_287.jpg "Grayscaling"
[image3]: ./images/left_2016_12_01_13_30_48_287.jpg "Recovery Image"
[image4]: ./images/center_2016_12_01_13_30_48_287.jpg "Recovery Image"
[image5]: ./images/right_2016_12_01_13_30_48_287.jpg "Recovery Image"
[image6]: ./images/center_2018_07_05_00_22_14_552.jpg "Normal Image"
[image7]: ./images/center_2018_07_05_00_22_14_552_fliped.jpg "Flipped Image"
[image8]: ./images/loss.png "Flipped Image"

## Rubric Points
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My final model consisted of the following layers:

| Layer  |Description  |
|---|---|
| Input | 3@66x200 YUV image  |
| Normalize | 3@66x200  |
| Convolution 5x5 | 24@31x98. activation='relu' |
| Convolution 5x5 | 36@14x47, activation='relu' |
| Convolution 5x5 | 48@5x22, activation='relu' |
| Convolution 3x3 | 64@3x20, activation='relu' |
| Convolution 3x3 | 64@1x18, activation='relu' |
| Dropout | keep_prob=0.5 |
| Flatten | |
| Fully connected 1 | input 1164, output 100 |
| Fully connected 2 | input 100, output 50 |
| Fully connected 3 | input 50, output 10 |
| Fully connected 4 | input 10, output 1 |

This network was referred to [this paper](https://arxiv.org/pdf/1604.07316v1.pdf).

#### 2. Attempts to reduce overfitting in the model

* The model contains dropout layers in order to reduce overfitting ([model.py](https://github.com/atinfinity/CarND-Behavioral-Cloning-P3/blob/master/model.py) lines 29). 
* The model was trained and validated on different data sets to ensure that the model was not overfitting ([model.py](https://github.com/atinfinity/CarND-Behavioral-Cloning-P3/blob/master/model.py) 104-114). 
* The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([model.py](https://github.com/atinfinity/CarND-Behavioral-Cloning-P3/blob/master/model.py) line 109).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to extract feature(lane edge etc...) and classification of steering angle.

My first step was to use a convolution neural network model similar to the image clafssication. I thought this model might be appropriate because I think that two stages.

- Feature extraction(lane edge etc...)
- Classification(steering angle)

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 

I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the following point.

- Network
  - use `Dropout` in my network
- Data augumentation
  - use images of 3 camera
  - imaege flipping(horizontal)

The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture ([model.py](https://github.com/atinfinity/CarND-Behavioral-Cloning-P3/blob/master/model.py) lines 17-36) consisted of a convolution neural network with the following layers and layer sizes. 

| Layer  |Description  |
|---|---|
| Input | 3@66x200 YUV image  |
| Normalize | 3@66x200  |
| Convolution 5x5 | 24@31x98. activation='relu' |
| Convolution 5x5 | 36@14x47, activation='relu' |
| Convolution 5x5 | 48@5x22, activation='relu' |
| Convolution 3x3 | 64@3x20, activation='relu' |
| Convolution 3x3 | 64@1x18, activation='relu' |
| Dropout | keep_prob=0.5 |
| Flatten | |
| Fully connected 1 | input 1164, output 100 |
| Fully connected 2 | input 100, output 50 |
| Fully connected 3 | input 50, output 10 |
| Fully connected 4 | input 10, output 1 |

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric) And, this figure was cited from [this paper](https://arxiv.org/pdf/1604.07316v1.pdf).

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the sides of the road back to center. 

These figures show images of multiple cameras(left, center, right).

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help with the left turn bias. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 6 number of data points. I then preprocessed this data by the following method.

- use images of 3 camera
  - 3x data augumentation
- imaege flipping(horizontal)
  - 2x data augumentation

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here is loss curve(training and validation). From this figure, I think that over or under fitting has not occurred.

![alt text][image8]

### Evaluation of Track2
The simulator contains two tracks. So, I evaluated `Track2`, too.  
At first, I collected data of `Track2` from simulator. As a result, I collected 35901  images and angles. And, I increased the data by data augumentation(x2).

Here is loss curve(training and validation). From this figure, I think that over or under fitting has not occurred. And, you can watch the result on xxxx. 

I think that `Track2` has the following difficult condition.

- include shade scenes
- include steep slope

So, I think that I should increased data to improve result by data augumentation(shade, shift, brightness etc...).