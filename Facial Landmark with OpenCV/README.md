# 👋 Hello

# Facial Landmark Detection with OpenCV

![picture](https://paperswithcode.com/media/tasks/Screenshot_2019-11-22_at_20.21.00_jiyQulV.png)

# Introduction

We are going to use dlib and OpenCV to detect facial landmarks in an image.

Facial landmarks are used to localize and represent salient regions of the face, such as:

  - Eyes
  - Eyebrows
  - Nose
  - Mouth
  - Jawline

Facial landmarks have been successfully applied to `face alignment`, `head pose estimation`, `face swapping`, `blink detection` and much more.

## What are Facial Landmarks ?

Detecting facial landmarks is a subset of the shape prediction problem. Given an input image (and normally an ROI that specifies the object of interest), a shape predictor attempts to localize key points of interest along the shape.

In the context of facial landmarks, our goal is detect important facial structures on the face using shape prediction methods.

Detecting facial landmarks is therefore a two step process:

`Step #1`: Localize the face in the image.
`Step #2`: Detect the key facial structures on the face ROI.

Face detection (Step #1) can be achieved in a number of ways.

We could use OpenCV’s built-in Haar cascades.

We might apply a pre-trained HOG + Linear SVM object detector specifically for the task of face detection.

Or we might even use deep learning-based algorithms for face localization.

In either case, the actual algorithm used to detect the face in the image doesn’t matter. Instead, what’s important is that through some method we obtain the face bounding box (i.e., the (x, y)-coordinates of the face in the image).

Given the face region we can then apply Step #2: detecting key facial structures in the face region.

There are a variety of facial landmark detectors, but all methods essentially try to localize and label the following facial regions:

  - Mouth
  - Right eyebrow
  - Left eyebrow
  - Right eye
  - Left eye
  - Nose
  - Jaw

The facial landmark detector included in the dlib library is an implementation of the [One Millisecond Face Alignment with an Ensemble of Regression Trees paper by Kazemi and Sullivan (2014)](https://pdfs.semanticscholar.org/d78b/6a5b0dcaa81b1faea5fb0000045a62513567.pdf).

This method starts by using:

A training set of labeled facial landmarks on an image. These images are manually labeled, specifying specific (x, y)-coordinates of regions surrounding each facial structure.
Priors, of more specifically, the probability on distance between pairs of input pixels.
Given this training data, an ensemble of regression trees are trained to estimate the facial landmark positions directly from the pixel intensities themselves (i.e., no “feature extraction” is taking place).

The end result is a facial landmark detector that can be used to detect facial landmarks in real-time with high quality predictions.

For more information and details on this specific technique, be sure to read the **paper by Kazemi and Sullivan** linked to above, along with the official dlib announcement.

## Understanding dlib’s facial landmark detector

The pre-trained facial landmark detector inside the dlib library is used to estimate the location of 68 (x, y)-coordinates that map to facial structures on the face.

The indexes of the 68 coordinates can be visualized on the image below:

![picture](https://www.researchgate.net/profile/Sushant_Gautam/publication/328043674/figure/fig1/AS:677615992057856@1538567650004/Visualizing-68-facial-coordinate-points-from-the-iBUG-300-W-dataset-5.jpg)

These annotations are part of the 68 point [iBUG 300-W dataset](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) which the dlib facial landmark predictor was trained on.

It’s important to note that other flavors of facial landmark detectors exist, including the 194 point model that can be trained on the [HELEN dataset](http://www.ifp.illinois.edu/~vuongle2/helen/).

Regardless of which dataset is used, the same dlib framework can be leveraged to train a shape predictor on the input training data — this is useful if you would like to train facial landmark detectors or custom shape predictors of your own.
