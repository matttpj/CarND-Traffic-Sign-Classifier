# Traffic Sign Classifier

## Writeup by Matthew Jones

### Project: CarND-Traffic-Sign-Classifier-P3
---

**Build a Traffic Sign Classifer**

The main steps for this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

<br/>
<img src="./output_images/training_sample_images.jpg" width=40% height=40%>
<img src="./output_images/training_traffic_sign_frequency.jpg" width=40% height=40%>
<br/>

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/matttpj/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

See below for an exploratory visualization of the training data set, including a random selection of 20 images and a histogram showing the frequency distribution of images by traffic sign type (class id).
<br/>
* __Sample images (Training Set)__  
<img src="./output_images/training_sample_images.jpg" width=40% height=40%>
* __Frequency distribution of images by class id (Training Set)__  
<img src="./output_images/training_traffic_sign_frequency.jpg" width=40% height=40%>
<br/>

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Below steps were inspired by the Udacity programme materials and from reading the whitepaper on [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Pierre Sermanet and Yann LeCun.

The following pre-processing steps were performed on the training data (../data/train.p).
 * __Convert images to grayscale__  
 _The above referenced Sermanet and LeCun whitepaper mentions on page 2 that using grayscale images improved ConvNet performance_
 * __Augment the number of images by a scale factor__
 _Many of the image classes had less than ~250 samples which was insufficient to train the model without giving undue bias to the image classes with greater than ~1250 samples; so I augmented the data with increased emphasis on scaling numbers of those images with fewer samples by creating additional samples with random variants of shear, blur and gamma transformations. This was done by:_
  * __Separate the traffic signs into separate arrays__
  * __Calculate the no. of signs per traffic sign class__
  * __Calculate the mean no. of signs per class__
  * __Calculate the mean per sign class / images per sign class__  
  _This will show for each image class the factor by how much it is greater or lesser compared to the average_  
  * __Select an initial Scale Factor multiplier to boost the signs with fewer samples over ~ 2,000 sample threshold to reduce the model bias__  
  * __Generate additional images per class as determined by final Scale Factor__    
  _Final Scale Factor = (Initial Scale Factor * avg_per_sign / images_per_sign)_    
 _Initial Scale Factor was modified by trial and error to find a value that improved ConvNet accuracy_  

The above augmentation method was derived from method described here _https://github.com/eqbal/CarND-Traffic-Sign-Classifier-Project_

Finally, as recommended by Udacity programme, I chose to Normalize the image to within the range (-1, +1) by scaling with the following adjustment to the X_train pixel data >>> **(X_train - 128) / 128**.  However, Normalization appeared to have very limited impact on the accuracy of the model.

The difference between the original data set and the augmented data set is illustrated by the below charts.
<br/>
* __Training data set: pre v. post image pre-processing and augmentation__
<img src="./output_images/training_pre_v_post_image_preprocessing.jpg" width=60% height=60%>
<br/>

Here are some example of augmented images created off the original image X_train[555], Class Id: 12 >> Priority road:  
* __Augmented image examples__
<img src="./output_images/training_augmented_images_sample_transformations.jpg" width=40% height=40%>
<br/>

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

* __LeNet training model__
<img src="./output_images/lenet.png" width=50% height=50%>
<br/>

Taking above LeNet model, as a starting point my final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<br/>
<img src="./test_images/01.jpg" width=30% height=30%>
<img src="./test_images/02.jpg" width=30% height=30%>
<img src="./test_images/03.jpg" width=30% height=30%>
<br/>
<img src="./test_images/04.jpg" width=30% height=30%>
<img src="./test_images/05.jpg" width=30% height=30%>
<br/>

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
