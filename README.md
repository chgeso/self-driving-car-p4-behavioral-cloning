**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model.
* drive.py for driving the car in autonomous mode.
* model.h5 containing a trained convolution neural network.
* run1 containing images captured by autonomous driving.
* video.py for making a video out of the images in run1 directory.
* README.md summarizing the results.

#### 2. Submission includes functional code
First of all, you can make model.h5 by executing model.py script like below.
```
python3 model.py
```
It is used for a provided Udacity simulator and drive.py file, the car can be driven autonomously around the track by executing 
```
python3 drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for collecting three camera images and each angle measurements, augmenting the images, a modified network model based on NIVIDA CNN Architecture for training the images.   
 The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

First of all, I refer to NIVIDA CNN architecture. Therefore, my model consists like below.   
01. A convolutional neural network with 24 filters, 5x5 kernel, 2x2 stride, and RELU activation function.     
02. A convolutional neural network with 36 filters, 5x5 kernel, 2x2 stride, and RELU activation function.     
03. A convolutional neural network with 48 filters, 5x5 kernel, 2x2 stride, and RELU activation function.     
04. A convolutional neural network with 64 filters, 3x3 kernel, and RELU activation function.     
05. A convolutional neural network with 64 filters, 3x3 kernel, and RELU activation function.     
06. Flatten connected layer with output 100.   
07. Flatten connected layer with output 50.    
08. Flatten connected layer with output 10.    
09. Flatten connected layer with output 1.     
   
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting after each convolution layer and flatten connected layer. But, the rate is different between the two type layers.    

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 74). However, for choose an area of interest in the images, you have to tune the cropping parameters about top and bottom. In this project you can leave the left and right to 0.   

#### 4. Appropriate training data

I chose the training data provided by Udacity originally. I used all of the images taken by the three cameras (center, left, and right). And then I augmented the images by flipping.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to NVIDIA CNN Architecture.

My first step was to augment the images by using flipping and then applied normalization and cropped for choosing the interest area without sky and hood of the car.   

My second step was to use the convolution layers and flatten connected layers similar to the NVIDIA CNN Architecture. I thought this model might be appropriate because it is easy to understand how to construct the model. The model is familiar with the previous project with LeNet architecture.

To combat the overfitting, I modified the original NVIDIA CNN Architecture by adding dropout layer after above each layer sequence.  

Then I compiled with Adam Optimizer and MSE loss function.   

The final step was to run the simulator to see how well the car was driving around track one. There was one spot where the vehicle nearly fell off the track after cross over the bridge. And the speed is mostly same. 
To improve the driving behavior in these cases, I need to collect good data by driving on training mode of simulator.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is like below.   
01. A lambda layer to parallelize image normalization.   
02. A cropping layer with top=70, bottom=25, left=0, and right=0.
03. A convolutional neural network with 24 filters, 5x5 kernel, 2x2 stride, and RELU activation function.   
04. A droup out layer with 0.2 rate.   
05. A convolutional neural network with 36 filters, 5x5 kernel, 2x2 stride, and RELU activation function.   
06. A droup out layer with 0.2 rate.   
07. A convolutional neural network with 48 filters, 5x5 kernel, 2x2 stride, and RELU activation function.   
08. A droup out layer with 0.2 rate.   
09. A convolutional neural network with 64 filters, 3x3 kernel, and RELU activation 0function.   
10. A droup out layer with 0.2 rate.   
11. A convolutional neural network with 64 filters, 3x3 kernel, and RELU activation function.   
12. A droup out layer with 0.2 rate.   
13. Flatten connected layer with output 100.   
14. A droup out layer with 0.5 rate.   
15. Flatten connected layer with output 50.   
16. A droup out layer with 0.5 rate.   
17. Flatten connected layer with output 10.   
18. A droup out layer with 0.5 rate.   
19. Flatten connected layer with output 1.   
20. Adam Optimizer and mean squared error loss. 

#### 3. Creation of the Training Set & Training Process

First of all I used the training data provided from Udacity. The most of the images taken a center camera are in the center between the lane lines. This means that the trained behavior would keep staying in the center. When I collect all of the images taken the three camera, I set the angle measurements for left and right camera images by adding or subtracting a parameter value called "correction". It is a parameter to tune for left and right angle measurements.      

And then to augment the data set, I also flipped images and angles. The angles of flipped images are multiplied by -1 to the original angles. 

After the collection process, I preprocessed this data by lambda layer to normalization and by cropping layer to see only interest area of the images.   

Finally, randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the cheat sheets from Udacity. I used an adam optimizer so that manually training the learning rate wasn't necessary.
