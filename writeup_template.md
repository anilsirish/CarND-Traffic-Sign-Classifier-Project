# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Initial_Distib.png "Data Visualization 1"
[image2]: ./Final_Distib.png "Data Visualization 1"
[image3]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./add_images/walk.jpg "Traffic Sign 1"
[image5]: ./add_images/unknown.jpg "Traffic Sign 2"
[image6]: ./add_images/exclam.jpg "Traffic Sign 3"
[image7]: ./add_images/30.jpg "Traffic Sign 4"
[image8]: ./add_images/stop.jpg "Traffic Sign 5"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### This writeup include all the rubric points and how I've addreses each one. 

Here is a link to my [project code](https://github.com/anilsirish/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

You can also view the HTML export of notebook file here [HTML](http://htmlpreview.github.io/?https://github.com/anilsirish/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validatation examples = 4410
* Image data shape = (32, 32, 3)
* Number of unique classes = 43

#### 2. Visualization of dataset

Here is a chart showing how initial training data is distributed over different clasees. It shows that there are very few training images for certain classes compared to others. To avoid any potential for biased predictions by our model, it is better to add more training images for classes with low count. 

![Initial Distribution][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing of image data

I've used following techniques to preprocess the image data

1. Converto to grayscale. As the color attribute of image is not important to classify the data, it is better to convert to grayscale to improve performance and acuracy of the model. Here is an example of a traffic sign image before and after grayscaling.

![Gray Scale Image][image3]

2. Normalization. Image data needs to be normalized so that the data has mean zero and equal variance. I've first used the formulae `(pixel - 128)/ 128` but noticed that the normalizing with `pixel / 255 * 0.8 + 0.1` has improved acuracy of prediction.

3. Generate additional data. As shown in above histogram, training set contains very few training images for ertain classes. To improve acuracy of prediction I've decided to generate more training images by rotating existing images at various angles. I've chosen rotation as it is easier to do but chosing other techniques such as generating “jittered” copies would have been better. Following is the histogram of training data distribution over different classes.

![Final Distribution][image2]


#### 2. My final model consisted of the following layers:


Layer 1: Convolutional layer with input 32x32x1 and Output 28x28x6.
         Activation with Relu.
         Max Pooling, input 28x28x6 and output 14x14x6.
         
Layer 2: Convolutional layer with output 10x10x16.
         Activation with Relu.
         Max Pooling, output 5x5x16.

Layer 3: Fully Connected, Input is flatten data from previous layer, i.e. input = 400 (5x5x16), output = 120.
         Activation with Relu.
         Dropout, to prevent overfitting with keep probability of 0.7 for training.
         
Layer 4: Fully Connected, input 120, output 84.
         Activation with Relu
         Dropout, to prevent overfitting.
         
Layer 5: Fully Connected (Logits), input 84, output 43 i.e. numer of classes.


#### 3. Training the model

To train the model, I used following

* Optimizer - AdamOptimizer
* No. of epochs - 50
* Batch size - 200
* Learning Rate - 0.0009

#### 4. Approach and Final results

It was not easy to get the final acuracy 0.93 but after trial and error achieved slightly better accuracy then 0.93.

My final model results were:
* validation set accuracy of 0.942
* test set accuracy of 0.933

The accuracy of model could have improved further but as I am rushing to complete other projects, stopped after 0.933. Probably adding another layer of Convolution would have helped to extract more features of training data there by increasing accuracy. Also to generate additional data, jittered copies of existing images would have been better compared to simple rotation.

I've chosen the same architecture that was taught in classroom i.e. LeNet as this model looked to work better with images. 

I used almost the same architecture as the LeNet but with dropout to avoid overfitting. Dropout was added after I noticed that the accuracy during training is better but dropped when testing with test data. Dropout with keep probability of 0.5 during training has helped to get consistent testing accuracy.


### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![Traffic Sign 1][image4] ![Traffic Sign 2][image5] ![Traffic Sign 3][image6] 
![Traffic Sign 4][image7] ![Traffic Sign 5][image8]


The model was able to correctly guess all 5 traffic signs, which gives an accuracy of 100%. Are the test images chosen were too easy to clasify? May be, but I am happy with 100% accuracy :) 

Following are the top 5 softmax probabilities of above images.

 1. [ 1.000,  0.000,  0.000,  0.000,  0.000] 
 2. [ 0.995,  0.005,  0.000,  0.000,  0.000] 
 3. [ 0.994,  0.006,  0.000,  0.000,  0.000] 
 4. [ 0.993,  0.007,  0.000,  0.000,  0.000]
 5. [ 1.000,  0.000,  0.000,  0.000,  0.000] 

Please refer to the output of cells 39 & 40 (in IPython notebook or HTML version) where these softmax probabilities and final performance are calculated.

 [HTML Version of Notebook](http://htmlpreview.github.io/?https://github.com/anilsirish/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)

