# **Behavioral Cloning** 

## Writeup


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./leNet.png "LeNet"
[image2]: ./cnn-architecture.png "Architecture"
[image3]: ./brightness.png "Brightness"
[image4]: ./shadow.png "Shadow"
[image5]: ./shiffting.png "Shiffting"
[image6]: ./epoch5.png "5 Epochs"



## Rubric Points
I started to implement a simple model, using only the center camera and augmented the dataset by flipping the images as being a quick way to augment the data.
Then I trained the model using the LeNet architecture. I directly saw that the accuracy was lower on the validation set and high on the training set. By "low", I mean that the difference was distiguishable but in the reality the loss on the validation set was very small, around 0.0772 and of ~0.0423 on the training set.
![alt text][image1]

Instead of testing different regularization technics like L2, dropout, etc, I prefered to focus on a different network architecture. I chose to implement the NVidia architecture described in the ["End to End Learning for Self-Driving Cars"](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper.
![alt text][image2]

The architecture consists of 5 convolutions and 3 connected layers. In contrary to the LeNet architecture, the model seemed to not overfit and the accuracy was very high on both training and validation set.

As normalization step, at this point I used the following formulae:
```
(x / 255.0) - 0.5
```
After this first 2 training steps, the car was able to drive autonomously but not too long, after several turns it started to go off the road and couldn't recover/


The next step I took can be splitted into several points:
* comparing the quality of driving with the arrows, mouse and gamepad, I ended up with the last. 
- I have driven the car for around 2 laps in one direction and one lap in the opposite direction
- I recorded data when the car is driving from the side of the road back toward the center line. As suggested, I turned off data recording when weaving out to the side, and turned it back on when steering back to the middle.
- finally, I have focused on driving smoothly around curves for half of a lap

The whole was done on the first trackonly.

* I applied several data augmentation technics:
- brightness to simulate different light conditions
![alt text][image3]
- shadow augmentation. I have observed that the second track contains a lot of portion of the road with shadows.
![alt text][image4]
- horizontal and vertical shiffting to simulate the effect of car being at different positions on the road, and add an offset corresponding to the shift to the steering angle.
![alt text][image5]
- flipping images to simulate the car driving in the opposite direction that the one given by default in the simulator
- cropping as not all of these pixels contain useful information, in order to focus on only the portion of the image that is useful for predicting a steering angle. 

The shadow and shiffting technics were reported to be often effective.

* I'm using the all 3 cameras (left, center, right ones) in order to simulate the effect of the car wandering off the side, and recovering. I added an angle correction equal to 10% of the angle obtained from the center camera to the left camera and subtracted this same correction from angle obtained from the right camera.

* I combined each of the augmentation technics above by using generators
* I changed the normalization to
```
 x/127.5 - 1.
```
* NVidia architecture was used and the model was trained on several machines at the same time: Amazon GPU instance for more epochs and on a CPU machine for loss epoch. After more that 2 days of training and experimenting, the model was robust and fast enough by using a batch size of 32 (image and labels-steering angles for each batch). The above preprocessing steps were applied on the CPU.

* I reduced the number of epochs from 8 to 5 first and then to 3.
* Finally, as the cropping step was not applied using the Cropping2D Keras layer, I changed a litle the drive.py script in order to scale the images sent by the cameras to the same size as the images we had during the training after the cropping step.

The car is able to drive for hours on the first track.


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `Car Behavioral Cloning.ipynb` containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_nvidia_gen.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_nvidia_gen.h5
```

#### 3. Submission code is usable and readable

The `Car Behavioral Cloning.ipynb` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 3 convolution layers with a 5x5 filter and a stride of 2, 2 convolution layers with a 3x3 filter and a stride of 1

The model includes RELU layers to introduce nonlinearity (cell 41), and the data is normalized in the model using a Keras lambda layer (cell 41). 

#### 2. Attempts to reduce overfitting in the model

The model doesn't contains dropout layers as no overfitting was observed with such architecture.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, left, center and right camera with angle adjustments and reverse direction driving .

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to tune the data transformation mostly and to see how well it helped to drive the car in autonomous mode as well as reducing the training time by kipping the validatoon error low, reducing the batch size from 128 to 32 as well as lowering the epoch finally helped to achieve good results.

My first step was to use a convolution neural network model similar to the NVidia architecture I thought this model might be appropriate because ["End to End Learning for Self-Driving Cars"](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper shows quite interesting results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model (LeNet) had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I could modify the model so that it contains dropout but I chosed to focus on a different architecture.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (cell 41) consisted of a convolution neural network described by NVidia.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the off road. 

I didn't repeat this process on track two in order to get more data points. But it could be agood experiment.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the following tests.
![alt text][image2]

Here 3 epochs were used but I tested also 8 epochs which gave me similar results.

I used an adam optimizer so that manually training the learning rate wasn't necessary.


## What next
The model never saw images from the second track. After training the model on the first track, the car drive directly off the road on teh second track.
We could improve the model 
- by adding noising technics and playing with the shadow augmentation 
- by driving on this second track for generalization

We could finally tune the model with dropout to prevent any potential overfit. What is also quite interesting is that training on g2.2xlarge(GPU) Amazon instance was notmuch faster when using generators to create training and validation samples. The reason is apparently due to the augmentation preformedin the CPU and we could rather use tensorflow API in order to augment the dataset on the GPU. We can also try tensorflow tfrecords storing images in a queue.