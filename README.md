# Dog_Breed_Identification
capstone for galvanize

## Introduction:
The goal of this project is to create a classifier capable of determining a dog's breed from a photo. With a training set and a test set of images of dogs. Each image has a filename that is its unique id. The dataset comprises 120 breeds of dogs. 

## Data Description:
The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. 

* Number of categories: 120
* Number of images: 20,580
* Annotations: Class labels

http://vision.stanford.edu/aditya86/ImageNetDogs/

## Data Preparation
1. load each image into numpy array with standardized dimensions 500x500x3
2. map file name of each image with corresponding labels (breed type) 
3. exclude images that is not in the labels.txt

## Model Used 
* classify_mlp - Multilayer perceptron: CV accuracy ~2%
* classify_cnn - Convolutional neural network: CV accuracy ~20%
* classify_net - modified Convolutional neural network: CV accuracy ~90%


## Model Architecture

       BN
        |
    __________
    |  Conv   |
    |   |     |
    |  Max    |
    | Pooling |  X 5 layers
    |   |     |
    |  BN     |
    |___|_____|  
        |
      Dense
     
     
## Reference:
*https://keras.io/getting-started/sequential-model-guide/

*https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model




