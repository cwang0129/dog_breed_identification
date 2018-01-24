# Dog_Breed_Identification
capstone for galvanize


## Introduction:
On reddit.com, there is a section called "Awwwww". Many people will ask question about what kind of dog is this for one particular post. It inspired me to creat a bot who can answer this question automatically. 


## Data Description:
The Stanford Dogs dataset contains images of 120 breeds of dogs from around the world. 

* Number of categories: 120
* Number of images: 20,580
* Annotations: Class labels

http://vision.stanford.edu/aditya86/ImageNetDogs/

## Data Preparation
1. load each image into numpy array with standardized dimensions 500*500*3
2. map file name of each image with corresponding labels (breed type) 
3. exclude images that is not in the labels.txt

## Model Used 
classify_mlp - Multilayer perceptron: CV accuracy ~2%
classify_cnn - Convolutional neural network: CV accuracy ~20%
classify_net - modified Convolutional neural network: CV accuracy ~90%


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
     
     




