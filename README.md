# Face-mask detector
Version: 1.0\
App_link: https://share.streamlit.io/willstonewill/face-mask-detector/main

# Introduction

This is the capstone project of [Lighthouse Lab](https://www.lighthouselabs.ca/) Data Science bootcamp(Oct. 2021 cohort). Aiming to present data science skills learned during the three months intensive bootcamp. Version 1.0 was finished within 2 weeks. 

The purpose of this project is to create a model that has the ability of classifying whether who in an image are wearing masks and who are not.

## Datasets

The datasets used in this project come from:
Version 1.0: https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset

It offers in total 11,798 images splitting into 3 folders: training, validating and testing.

This dataset was used for the second step of the whole model: mask classification.

## How it works

1. Original image as an input
<img src="/readme_img/step1.png" width="200"/>
2. Capture faces
<img src="/readme_img/step2.png" width="200"/>
3. Mask classification
<img src="/readme_img/step3.png" width="200"/>
4. Labelling and output
<img src="/readme_img/step4.png" width="200"/>

## Model - Face detection

This is the first step of the whole model. The model in this step is from the package [“mtcnn”](https://github.com/ipazc/mtcnn) which is the implementation of “Multi-task Cascaded Convolutional Neural Networks” from Zhang, K et al.(2016)

It is a model pre-trained on [FaceNet](https://github.com/davidsandberg/facenet/tree/master/src/align) and consists of three parts:
- P-Net or proposal network:\
In this part, many candidates of face are found, proposed and calibrated.
- R-Net or refine network:\
In this part, candidates are further calibrated with bounding box regression and non-maximum suppression(NMS).
- O-Net or output network:\
In this part, a face bounding box with 5 face landmarks is created as output.

Here in our project, only the face bounding box and the probability of confidence were used to determine the threshold of a detected face. It can provide an accuracy around **95%~98%** depending on the datasets.

## Model - Mask classification

This is the second step of the model. The input of this step are the cropped and resized faces detected from the first step “face detection”. To achieve a better performance, three models were created for this step and the best performing one was selected as the final model.

- Baseline model: \
\
This model is created as a benchmark and baseline. It is combined purely by dense layers. The testing accuracy is *51.3%* which is slightly or no better than random guess. Confusion matrix below:
<img src="/readme_img/cm_baseline.png" width="300"/>

- CNN model: \
\
Consists of several convolutional layers and dense layers. It provides a huge increase of accuracy upto *96.5%*. But it has a false positive issue which is especially severe under the scope of this project. Confusion matrix blow:
<img src="/readme_img/cm_cnn.png" width="300"/>

- Transfer learning model with VGG19(final model): \
\
A methodology of transfer learning is applied in this model on the VGG19 architecture. Typical transfer learning process is introduced and followed here. 
1. First, freeze all weights of the pre-trained VGG19 architecture and create a deep neural network structure containing the VGG19 as an instance and some dense layers on top of it. 
2. Second thing is to train this deep neural network by using the datasets mentioned above in the “Datasets” section. 
3. Third, unfreeze some layers of the VGG19(in my case the last 3 dense layers) and then retrain the whole model with the same training data(fine-tuning). \
\
This transfer learning technique can be helpful to keep VGG19 which was trained on a huge dataset of images(ImageNet) as a generic model, at the same time, make it specific to our topic/data.\
As a result, this method offers an accuracy of **99.0%** before fine-tune and **99.8%** after fine tuning.

The confusion matrix:

<img src="/readme_img/cm_vgg.png" width="300"/> 

Accuracy and loss plots:

<img src="/readme_img/acc.png" width="300"/>

<img src="/readme_img/loss.png" width="300"/>

Classification report and roc/auc plot:

<img src="/readme_img/clf_report.png" width="300"/>

<img src="/readme_img/roc_auc.png" width="300"/>

## Next steps:

1. Optimize the model structure. Current two steps model has problems in the fact that wearing a mask can impede first step “face detection” from working as expected as the mask would unavoidably hide part of the face.
2. Gathering more generalised data. The datasets used in version 1.0 contain only one well cropped face per file.
3. Enrich the input data format. It currently supports jpg or jpeg format images as input. While png, gif and video formats should be included in future versions.
4. Extend prediction categories. A three-category of “Correctly masked”, “Incorrectly masked”, “Unmasked” may be more appropriate. 

## Relative resources:

1. [“How to perform face detection with deep learning”](https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/)
2. [“Tensorflow tutorial - Transfer learning and fine tuning”](https://www.tensorflow.org/tutorials/images/transfer_learning)
3. [“Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks”](https://arxiv.org/abs/1604.02878)
4. [“Very Deep Convolutional Networks for Large-Scale Image Recognition”](https://arxiv.org/abs/1409.1556)
