
# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_2018_01_09_17_44_10_503.jpg "Center Image"
[image2]: ./examples/left_2018_01_09_17_44_10_503.jpg "Left Image"
[image3]: ./examples/right_2018_01_09_17_44_10_503.jpg  "Right Image"
[image4]: ./examples/right_2018_01_09_17_45_03_249.jpg "Recovery Image"
[image5]: ./examples/index.png "Loss Curves"
[image6]: ./examples/flipped_image.jpg "Flipped Image"

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

All these files can be found [here](https://github.com/AnkurSatya/Behavioral-Cloning)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Collecting training Data

![image1]

Choosing the data was an important step and a bit tricky. 
Initially I collected data(center driving) just by going clockwise around the track for 1.5 lap and same for counter-clockwise. When the model created with this data was run on the simulator, vehicle fell off at some spots (near the bridge, in the lake).

To improve the driving behavior in these cases, I slowed down the vehicle near the curves to take more data and to introduce the smooth turning behavior. Additionally, recovery data was also collected. 

**Recovery data**- went off the center and then started recording while steering back to the center. This was especially very necessary near the bridge and this did help the vehicle to stay on the track.

![image4]

The image above shows the vehicle moving away from the offset position.

The model with this data helped the vehicle to stay on the road and take smooth turns.

#### 2. Pre processing

**Normalization**

The input image was normalized to remove the bias created by a higher intensity pixel value. In general, it also handles the biases created by the different environmental conditions while capturing image(cloudy weather, unusual brightness) but in the simulator, the conditions were same throughout.

**Cropping**

Image was then cropped from the top to remove the sky and nearby trees, as they were not useful for our task. Also, image was cropped from the bottom to remove the hood of the car.

**Multiple Cameras**
Outputs from all three cameras were used for training. Data was divided into left, right and center images. For evaluating the angles associated with left and right images, a correction factor of 0.2 was used. 

**Center Image**
![image1]

**Left Image**

![image2] 

**Right Image**

![image3] 


Correction Factor=0.2

| Image | Angle	|
|:-----:|:-----:| 
| Left| Center angle+0.2|
|Right| Center angle-0.2|

**Data Augmentation**

To create more data, flipping technique was used. Every center image in the training set was flipped about the vertical axis. And the angle associated with it was the negative of the unflipped image.

*Original Image*
![image1]

*Flipped Image*
![image6]


After the augmentation process, the number of images were 30,948 and for validation set- 1976.

#### 3. Model Architecture

**Initial Model**

At first I used a network similar to the **LeNet**. Losses were decreasing but soon reached plateau which indicated towards underfitting. The reason for undefitting was quite clear, from the traffic sign classifier project, the model did not have sufficient depth to incorporate the complex incoming images. 

**Refined Model**

Next step was to increase the depth of the architecture.
But increasing depth alone makes the model more computationally expensive. So, increasing the width as well is a good practice and it has two benefits- 
1. less computation(explained below)
2. helps the model to gradually move from low level to high level features.

Instead of using one 5x5 convolutional filter, two 3x3 convolutional filters were used. This reduces the number of parameters by a factor of **2.7**. This can be easily understood with the convolution theory.

The architecture is described below and it was inspired by [Nvidia's](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) architecture of end-to-end learning.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   					    | 
|1.Normalization        | Lambda layer to normalize the inputs b/w -0.5 & 0.5|
|2.Cropping             | 50 pixels from top, 20 from bottom.           |  
|3.Convolution 5x5    	| 2x2 stride, outputs 43x158x24 	            |
|  Activation			| Exponential Linear Unit(ELU)			     	|
|4.Convolution 5x5	    | 2x2 stride, outputs 20x77x36 				    |
|  Activation           | ELU                                           |
|5.Convolution 5x5	    | 2x2 stride, outputs 8x37x48                   |
|  Activation           | ELU                                           |
|6.Convolution 3x3      | 1x1 stride, outputs 6x35x64                   |
|  Activation           | ELU                                           |
|7.Convolution 3x3      | 1x1 stride, outputs 4x33x64                   |
|  Activation           | ELU                                           |
|8.Flattening           | Flattens the output of Layer 7 to 8448 units  |
|9.Fully connected      | Size- 100 units                               |
|  Activation           | ELU                                           |
|10.Fully connected     | Size-50 units    							    |
|  Activation           | ELU                                           |
|11.Fully connected     | Size-10 units                                 |   
|  Activation           | ELU                                           |
|12.Fully connected-Output| Size-1 unit 							    |


The ELU activation function was choosen because of the nature of the regression output. The output can be negative and positive but ReLU only gives out positive values whereas ELU is exponential in the negative x region.


#### 4. Reducing overfitting

As the number of paramters are large, overfitting is unavoidable. To combat this, **l2 regularizer** was used with it's coefficient value being 0.001. Every layer had this regularizer.

More data created by data augmentation also helped in reducing overfitting.

#### 5. Optimizer tuning

To train the model, an adam optimizer was used. This optimizer reduces the number of parameters to be tuned. There is no need to decay learning rate as the momentum and cache paramters take care of that in adam optimizer.

| Parameter |Value| 
|:---------:|:---:| 
| Learning Rate| 0.001|
|beta 1-momentum| 0.9|
|beta 2- cache|0.99|
|learning rate decay|0.0|

The Ideal number of epochs were 10 because after that the loss curves reach plateau.

![image5]




```python

```
