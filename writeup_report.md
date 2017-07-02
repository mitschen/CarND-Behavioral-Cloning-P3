#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/final_sample_linearscale.png "Samples in linear scale"
[image2]: ./figures/final_sample_logscale.png "Samples in log-scale"
[image3]: ./figures/viewport.png "Viewport"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

After finishing the introduction part it was clear for me to follow the suggestion of reusing the nvidia network. Enclosed you'll find the table which summarizes the keras summary.


|Layer (type)                   |  Output Shape        |  Param #   |  Connected to        | Remark             
|:-:-:-:-:-:|
|cropping2d_1 (Cropping2D)       | (None, 48, 320, 3)  |  0         | cropping2d_input_1[0][0]         | reduce viewport to a size of 48x320, starting with px 80.
| lambda_1 (Lambda)               | (None, 48, 320, 3) |   0         |  cropping2d_1[0][0]               | apply normalization 0-mean
| convolution2d_1 (Convolution2D) | (None, 44, 316, 24) |  1824       | lambda_1[0][0]                   | 5x5x3 -> 24, relu activation
|maxpooling2d_1 (MaxPooling2D)   | (None, 43, 158, 24)  | 0          | convolution2d_1[0][0]       | 2x2 with a stride of (1,2)     
|convolution2d_2 (Convolution2D)  |(None, 39, 154, 36) |  21636     |  maxpooling2d_1[0][0]             | 5x5x24 -> 36, relu activation
|maxpooling2d_2 (MaxPooling2D)    |(None, 38, 77, 36)  |  0         |  convolution2d_2[0][0]            | 2x2 with a stride of (1,2)
|convolution2d_3 (Convolution2D)  |(None, 34, 73, 48)  |  43248     |  maxpooling2d_2[0][0]             | 5x5x36 -> 48, relu activation
|maxpooling2d_3 (MaxPooling2D)    |(None, 33, 36, 48)  |  0         |  convolution2d_3[0][0]            | 2x2 with a stride of (1,2)
|convolution2d_4 (Convolution2D)  |(None, 31, 34, 64)  |  27712      | maxpooling2d_3[0][0]             | 3x3x48->64, relu activation
|maxpooling2d_4 (MaxPooling2D)    |(None, 15, 17, 64)  |  0          | convolution2d_4[0][0]            | 2x2 with stride (2,2)
|convolution2d_5 (Convolution2D)  |(None, 13, 15, 64)  |  36928      | maxpooling2d_4[0][0]             | 3x3x48->64, relu activation
|maxpooling2d_5 (MaxPooling2D)    |(None, 6, 7, 64)    |  0          | convolution2d_5[0][0]             | 2x2 with stride (2,2
|flatten_1 (Flatten)              |(None, 2688)        |  0          | maxpooling2d_5[0][0]            | Flatten the output  
|dense_1 (Dense)                  |(None, 1164)        |  3129996    | flatten_1[0][0]                  | Fully Connected
|dense_2 (Dense)                  |(None, 100)         |  116500     | dense_1[0][0]                    | Fully Connected
|dense_3 (Dense)                  |(None, 50)          |  5050       | dense_2[0][0]                    | Fully Connected
|dense_4 (Dense)                  |(None, 10)          |  510        | dense_3[0][0]                    | Fully Connected
|dense_5 (Dense)                  |(None, 1)           |  11         | dense_4[0][0]                    | single output angle

Total params: 3,383,415
Trainable params: 3,383,415
Non-trainable params: 0

The cropping-layer reduces the viewport for the network. The intention is to not confuse the network with trees, water, hills, .... The resulting viewport of the car is shown in the figure below.

![alt text][image3]

So as you can see, I've reduced the view port to a very limeted part only. This reduced the input dimensions to a triple of 48, 320 with a depth of 3.

The next step - the so called lambda layer was applying a normalization to the pixel data - meaning a mean around 0 with a standard derivate of 1. As we've seen in the previous project - this is one of the key-steps in order to get a good regression network.

Then a bunch of convolutions are following - according to the [nvidia reference](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I didn't add any strides to the convolutions itself - instead I've added a maxpooling layer with strides after each conv. As you can see in the sources and accoring to the dimensions of the output shape in the model table above, I've chosen a stride in the x-dimension of 2 while staying with a stride of 1 for y-dimension. The reason for that is that our input shape is already very tight (48 px) and is shrinking very fast so that further convolutions couldn't be applied. For each of the convolution layers i've used the relu activation function.

Right behind the convolutions i've added the three fully connected layers according to our reference model which finally will end in one output neuron - representing the angle.

The whole model pipeline described here in short could be reviewed in the **trainModel** function.



####2. Dataprocessing, training and overfitting

Playing around with the simulator i've figured out, that recording a track will always result in overwriting existing training data. So to not do things twices, i've decided to apply some preprocessing before starting the training. 

First of all i did a preprocessing of the data - I'll go in details in the section data collection below. The result of the preprocessing step is that any collected figure is stored in a seperate folder with a increasing number, the kind of camera and the angle in filename:

`53633_r_-300.jpg => no 53633, r - right camera, angle of (-300 / 1000) * 25 in angle = -7.5°`

By applying this preprocessing - anytime i was recording new data, the data was appended instead of overwriting the existing ones. So with every testdrive i was able to increase my testset. Please refer to the method **augmentation** for reviewing this preprocessing step.

For sure by increasing the testset all the time, the training time was increasing as well. So i decided to not start every iteration from the scratch but instead re-using the model i've already trained to improve it. The intention was, that in case that i'm facing a situation during autonomous mode the network couldn't handle, i was able to collect some more specific data for the problematic part of the track and apply a finetuning to the model using the new samples only. This allows me reduce the time of training (which anyway took very long on my machine). Furthermore it allows me reduce the number of epochs: i've trained the model in one epoch - see how it scales, added some more testsequences and retrain it again for one epoch. 
The save/load methods are part of the method **trainModel**. As you can see I'm always storing the new model with a new filename instead of overwriting the existing one. The reason for that is that i wanted to backup my progress. Training a model tough on a certain situation might mess up the whole model - with the backup i was able to restore an older model in retrain it.

For training the model, i've implemented a generator see **generator** method. The generator is more or less the template from the introduction of the project. The only difference is that i've already applied a color-space convertion from BGR2RGB. The hint for that i got from the [helpful guide](https://slack-files.com/T2HQV035L-F50B85JSX-7d8737aeeb) mentioned in the project description. The batch-size i've choosen was 256 images instead of the suggested 32 - again reduction of file i/o and therefore speedup.


How did I prevent overfitting - I didn't. I started first with a dropout layer after each fully connected. The result was, the car takes a try on it offroad capabilities. The dropout I was using had a drop propability of 0.5. After removing the dropout, the car was staying on the road. With the resulting loss for training/ validation (see end of section) it was clear that I'm devinetly not running in an overfitted situation.


At the end of my development, i've started the network training from the scratch using 4 epochs with a training/validation set of a total of 84.5K samples. The evaluation results in a loss of 0.0044 for training and a loss in valiation of 0.0043.

####3. Data augmentation

I started collecting data for one round and reviewed the results. I noticed that the steering angle of 0 was appearing very often. My assumption was, that the straight segments of the track will not really cause that much problems while the curves will always result in a disaster. So I tried to reduce the occurences of steering angle 0 by driving 3 laps with mouse control - always keep the mouse button pressed and therefore almost provide some steering angle. The result of these rounds shows a much better distribution of samples for angles.

According to the suggestion in the project introduction i furthermore played around with the data itself. I took the images from left and right camera to the training set. Furthermore i've taken each image and flipped it horizontally. Each sample was stored on HDD as seperate image with a special filename pattern: 

`id _ camera _ angle*1000.jpg`

The id is a increasing number which prevents overwriting of existing samples in case of a new track recording. Camera is a one or two character letter: (c)enter, (c)enter(m)irror, (r)ight, (r)ight(m)irror, ... mirror means in this case that the image is flipped horizontally.
And finally the angle as read from the csv file multiplied by 1000 - this multiplicate helps me a write valid filenames (not floating point filenames).


Each image flipped (m) has a negative angle in filename. To adjust the angle for left/ right camera i've applied an additional offset of 0.02 to the angle. This is done in the method **getAngleFromFilename** which is called in the **generator** method. I deduced this guess 0f 0.02 by checking some steering angles for different images.


The final sample distribution looks like that:
![alt text][image1]
![alt text][image2]

As you can see, the samples are symmetric due to the flipping of data. The total number of samples i've used for training and validation is about 84.5K.

With the first autonomous run the car stay's very well on track but then fails in the first curve after the bridge. So I recorded the segment right after the bridge till about the start of the autonomous drive several times (~4 times in both directions) and retrained the model.

####4. Summary and result
With the nvidia network architecture for autonomous driving I was able to train a network which successfully pass the track. I've collected data using the simulation and increased my testsamples by refering to the cameras on left and right side as well as to flip all samples horizontally and add these to the sample data.