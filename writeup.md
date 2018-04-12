## Vehicle Detection Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_noncar.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/sliding_search.png
[image4]: ./output_images/final_detections.png
[image5]: ./output_images/heat_map.png
[image6]: ./output_images/cars_found.png
[image7]: ./output_images/final_detected.png
[video1]: ./output_images/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
For this project, I used the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite](http://www.gti.ssr.upm.es/data/Vehicle_database.html). Here's a set of random car and non-car images from the data sets.

![alt text][image1]

To extract the HOG features from the image I used the `get_hot_features()` function that was defined in the lectures.`get_hog_features()` takes an image and a given set of parameters (number of pixels per cell, number of orientations, cells per block, etc.) to extract HOG features using `hog()` from `skimage.features`.

I experimented with various color spaces and different parameter values to understand their influence on the output as well as the accuracy of the classifier being trained. 

Here's how the output looks like with a random car image and a random non-car image. There are marked differences in the HOG feature vectors. The parameters used are `orient=9, pix_per_cell=8, cell_per_block=2`.


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

The final choice for the HOG parameters was obtained through various trial and error steps. My major goals were to minimize the time required to generate the HOG features and maximize the detection chances for the classifier. I've tried various combinations of color spaces, hog channels, pixels per cell, cell  per block. I'm not sure if I have the best HOG features but they seem to be working with fewer false positives.

My current HOG parameters (can be found in code cell 6) are as follows:
`color_space = 'YCrCb'`
`orient = 8`
`pix_per_cell = 4`
`cell_per_block = 16`
`hog_channel = "ALL"`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM classifier using HOG features with all the color channels from the YCrCb color space (can be seen in the 8th code cell). I did not use any histogram or spatial features as they were increasing the number of false positives and the time taken to predict classes. I used the `train_test_split()` function to split my data set into train and test data sets (code cell 6). Once the split was done, I used the `StandardScaler()` function to scale the input training examples (code cell 7). I used the same scaler transform on the test set as well as the sliding window outputs from the video frames.

The SVM classifier did fairly well on the test data giving an accuracy of `0.9699`.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the faster search method described in the final chapters of the lesson. Once a frame has been extracted from the video, I computed the HOG for the entire image and extracted HOG features for specific locations on the image. My implementation can be found in code cell 9. 

We know that a car that is closer appears larger in an image than a car that is farther. Based on this observation, I restricted the sliding windows to a specific range of y pixel values. This eliminates searching for cars of a given scale in low probability locations. 

I checked for detections at different scales from 0.5 to 3.0 and found that my classifier has the best results in the 1.0 to 2.0 scale range. Based on this observation, I added multiple sliding window searches in that range.

I tried various overlap window ranges and found that the overlap (`cells_per_step` in my case) works best at 2. At 1 it is extremely slow and thresholding gets a little tricky and at 3 the number of detections falls sharply and I couldn't detect the white car in a large number of frames.

Here's an image showing detections on a test image. 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

After tinkering with all the parameters, I settled on using just the HOG features with YCrCB color space. Once that was established, I used a lot of the functions that were defined in class for sliding window search, generating heat maps and also for drawing bounding boxes.

Here's an image showing my final results on the test images.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
My video output can be found is named project_video_output.mp4 and can be found in the zipped folder. I think the detections are decent. There are a few frames where the white is not detected at all.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. I used a dequeue data structure to hold historical detectons from the past six frames. I created a heatmap based on both current and the previous five positive detections and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

I also added a confidence level parameter to the detection output of the SVM classifier.

`confidence = svc.decision_function(test_features)`

Through experimentation I found out that a confidence level of 20 or greater made sure that there were fewer false positives.

Here's an example result showing the heatmap from one of the test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on original image:

### Here is the heat map of an input test image:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()`:
![alt text][image6]

### Here the resulting bounding boxes on the input image:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The implementation for this project was fairly straight forward and I reused a lot of the functions and implementations from the lecture notes. 

The major issues I had were with respect to the false positive detections and coming up with various ideas to overcome those false positives. My biggest concern with this pipeline is that it can process 1 frame per second and is nowhere close to being realtime. I also think that my pipeline will fail at detecting white cars in bright backgrounds. 

To make this pipeline more robust, I am planning to switch from HOG feature based detections to a Deep Neural network (possibly faster R-CNN). The SVM classifier has an accuracy of 96.99% on my test inputs but is unable to classify with the same level of accuracy in the video. I think a deep learning based approach may come up with better features on its own for identification. I'm also planning to add the Udacity training data set and see if there is a difference in accuracy due to the additional data.


