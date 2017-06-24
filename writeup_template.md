Note to the reviewer - this model is trained on and tuned to *Fantastic quality* mode.  It will likely do poorly on other graphical quality levels.

# Behavioral Cloning

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[original]: ./examples/original.png "Original Image"
[cropped]: ./examples/cropped.png "Cropped"
[canny]: ./examples/canny.png "Canny"
[canny_exclusion]: ./examples/canny_exclusion.png "Canny with excluded ROI"
[filtered_steering]: ./examples/filtered_steering.png "Filtered steering"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. `model.py` and `tools.py` together show the pipeline I used for training and validating the model, and it contains comments to explain how the code works.  As well, there are three Jupyter notebooks included.  The most recentl notebook, `prototyping_170623.ipynb`, is where I actually did the model development and has some of my notes on that.  Then end model was trained there, but can be replicated with the `model.py` implementation.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I ultimately interpreted the NVIDIA architecture and found that interpretation to be a good fit.  If you chase down the [NVIDIA paper](https://arxiv.org/pdf/1604.07316v1.pdf]), you'll find they assiduously avoid going into detail on how their convolutional layers work.

I initially replicated their network with only convolutional and dense layers, but found my results were not pleasing.  I suspect that NVIDIA was using some intermediate layers between convolutions, so I added pooling and relu layers between the first 2 convolutions, which improved performance drastically. The dimension changes from using a (1,1) stride convolution and 2x2 Max Pooling match their changes, suggesting this may be on the right track.

Here are the layers

- Cropping2D(cropping=((59, 20), (0, 0)), input_shape=(160, 320, 4))
- Lambda(lambda x: (x / 255.0) - 0.5)
- Convolution2D(24, 5, 5, subsample=(1, 1), border_mode='valid')
- MaxPooling2D(pool_size=(2, 2), border_mode='valid')
- Activation('relu')
- Convolution2D(36, 5, 5, subsample=(1, 1), border_mode='valid')
- MaxPooling2D(pool_size=(2, 2), border_mode='valid')
- Activation('relu')
- Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid')
- Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid')
- Convolution2D(96, 3, 3, subsample=(1, 1), border_mode='valid')
- Convolution2D(110, 3, 3, subsample=(1, 1), border_mode='valid')
- Flatten()
- Dense(100)
- Dense(50)
- Dropout(0.5)
- Dense(10)
- Dropout(0.5)
- Dense(1)

#### 2. Attempts to reduce overfitting in the model

Dropout layers on the last two fully connected layers help regularize.  I tried more layers but found it hampered training too much.

I also trained the final model on 52,800 images, with forward and backward data from both tracks to minimize the overfit.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

I found that the angle adjustment for the left and right camera data was critical.  I probably spent the most time of all getting that angle correct.  When the adjustment angle was very high, I found the car swerved left to right a lot.  When it was too low, I noticed that the car would tend to get close to a curb and stay close, eventually leading to an error.

So I tried searching the space of angles and found that 0.3 and 0.4 were best.  With these angles, the car calmed down, drove mostly straight, and avoide the edges of the road (mostly.)

While finding the best angle adjustment was when I made a lot of use of loading previously trained models, freezing the feature extraction layers, and just training the mid late layers.  This helped speed up the process of finding the best setting.

#### 4. Appropriate training data

- I trained forward and backward on both tracks.
- Specifically trained on recovery events, and developed a whole methodology for that.
- After models failed, I would collect recovery data to specifically target that type of failure.
- I 6x every drive by using all 3 cameras and switching images.
- I eventually trained and tested only on Fantastic quality mode, based on the tip from Paul Heraty and the assumption that the DNN might do better at feature identification on a HQ setting.
- If I recorded a drive and messed up at some point, I would go through and delete the center frames that included the mistake.  My script then throws out those frames, the corresponding left and right frames, and the associated data.  This gave me a nice way to spot clean my data.

For capturing recovery, I would first position the car in a circumstance (ex: vehicle about to go off road way) I wanted to correct.  Then I would set up the wheel angle at a corrective angle.  Then I would start recording, and I would slowly move the vehicle through a corrective action toward the centerline.  Since the model did not make use of speed or throttle, only the relationship between the scene and the steering angle matters.  Thus, by going slow, I would get many more frames of that correction, and that improved my data collection process.

I would typically repeat this until I had 10 or 20 Mb of data of correction, and the create a new zip archive of that data set.  By labeling the data archives, and creating the option to load some data archives but not others, I allowed myself to selectively train.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Much of this is articulated above.  You can see some of the evolution in my approach in `prototyping_170623.ipynb`.  In terms of network architecture I evolved it like this:

- 2 layer fully connected
- 3 layer fully connected
- DNN with convolutions and tons of dropout
- DNN with convolutions, reasonable dropout, and canny layer
- Naive NVIDIA architecture + canny layer
- Final NVIDIA architecture + canny layer

I saw big improvements when I did the following:

- switched to a DNN
- reduced dropout
- found a good steering angle
- added maxpool and relu to NVIDIA architecture
- added lots of training data
- adding specific data targeting errors

Early models trained many epochs but as the data pool grew, I never went longer than 2 epochs, but of course the network is seeing a lot of data in just one epoch in that scenario.

As mentioned in passing above, I supplemented the BGR layers with a 4th channel, a canny edge detection on the BGR channel.  I don't know if this has impact, but I discovered late in the game that I had messed that process up, and when I repaired and retuned the parameters of the canny layer, I saw a nice improvement in driving behavior.

I used early stopping, typically with `patience=0` when the data pool had grown.

#### 2. Final Model Architecture

See above!

#### 3. Creation of the Training Set & Training Process

For training data processing, I tried the following, of which, I only used some in the final processing pipeline:

- Cropping [Layer 0 of final model]
- Normalization [added as Layer 1 of final model based on reviewer feedback]
- Canny edge detection as a 4th channel [included in final image processing pipeline]
- Canny edge detection restricted to a region of interest [not included in final pipeline]
- Low pass filter applied to steering angles [not included in final pipeline]

Let me quickly run through examples of each:

##### Cropping

As recommended, I excluded most of the sky, and a good portion of the bottom of the image as well. Specifically, I chopped 59 px off the top of the image and 20 off the bottom, leaving the horizontal dimensions untouched.  Here's the original followed by the cropped version:

![original image][original]
![cropped image][cropped]

##### Normalization

Standard, included in the model as Layer 1 now by request.  I'd show you the effect, but it would just look like a very dark image!

##### Canny edge detection & Canny with ROI

I realize that convolutions are very capable of edge detection, but I figured it would be worthwhile to prime the model by explicitly providing edges.  I spent some time tuning to get lower and upper thresholds for `cv2.Canny` of `142` and `233` respectively.  Here's the original followed by the final:

![original image][original]
![canny edged image][canny]

I also considered excluding the pathch immediately in front of the car from the canny channel.  The reason being that edge detection was picking up the texture of the road (one downside of using the high quality graphics settings) and I thought that might be confusing to the DNN.  However, I ultimately excluded it, having faith that the middle and final layers would ignore the texture.  Here's an example:

![canny with exclusion][canny_exclusion]

##### Low pass filter applied to steering angles

When I plotted out steering angles, I was shocked how uneven my steering was.  As a remediation for that, I tried applying a low pass filter: a 6th order Butterworth, applied in a forward pass and backward pass both 3rd order to avoid phase shift.  I experimented with various cutoff frequencies, but I did not see substantial improvements, although it was early in the model building, so I left it out. In the end, I was uncomfortable with the possibility that I might give bad guidance through this filtering to the DNN.  Here's a chart of the actual steer data vs filtered data:

![filtered steering][filtered_steering]

#### 4. Advanced Track!

Very tricky for even me to drive.  The blind hills are brutal.  For some reason my model just drives right into the barriers at the start.  My ambition is to come back and fix this later and see if I can get a decent lap time!