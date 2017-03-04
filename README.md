# Project 3: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---
* Implementation of behavioural cloning model
* Details of Augmentation Process
* Video generation
* Discussion

### Project files:
- `model.py` contains class Model with static methods for generation model and trainings:
    - _DenseNet_ based model
    - _SimpleNet_ neural network model
- `dataproc.py` contains two classes:
    - `Augment` is responsible for augmentation:
        * Height Cropping
        * Left-Right flipping
        * Brightness adjusting
        * Appliance of shadow mask on an input image
    - `TrackDataset` is responsible for:
        * Loading dataset
        * Augmenting images
        * Splitting dataset on testing and validation datasets
        * Batch generation for training and validation
- `common.py` contains auxiliary function for loading images and changing color map.
- `video.mp4` video of first track challenge.
- `model.h5` best _keras_ model

### Implementation

#### DenseNet based model

After reading NVIDIA's paper about driver behaviour cloning, I have decided to use my own neural network based on DenseNet. I came up with this solution:
* Batch size is **128**
* Input image shape **(12, 128, 3)**
* Input image in **RGB** color map
* **3** DenseNet blocks
* **3** layers per each block with growing factor **12**
* Batch normalization before each activation function
* **ELU** activation function was used, because as I think standard **ReLU** function will not work for _regression problem_. I think that the best activation function for regression challenges is **Leaky ReLU**, but I did not try it in this work.
* **3** fully connected layers
* Adam optimizer with **1E-03** learning rate and **1E-08** epsilon

It happened that this model is hardly trainable and the converge takes too long on my `GTX-1080`, after 10 epochs (5 hours) I had **0.15** validation error and results of driving were really poor, so that even on first turn the car went into the river.

I was really disappointed with gained results, but I didn't want to go with NVIDIA solution. After playing with different configurations I stayed with this one, and let's call it as _SimpleNet_:
* Batch size is **128** - _Again_
* Input image shape **(12, 128, 3)** - _Again_
* Input image in **RGB** color map - _Again_
* **4** convolution layers with max pooling:
    -
* __ELU__ activation function.
* 3 fully connected layers

##### Architectures

**SimpleNet** model architecture:
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 124, 124, 8)   608         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 62, 62, 8)     0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 58, 58, 8)     1608        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 58, 58, 8)     0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 29, 29, 8)     0           activation_1[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 26, 26, 16)    2064        maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 13, 13, 16)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 9, 9, 16)      6416        maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1296)          0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 128)           166016      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            6450        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 183,683
Trainable params: 183,683
Non-trainable params: 0
____________________________________________________________________________________________________
None
```

**DenseNet** based architecture:
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 128, 128, 3)   0                                            
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 128, 128, 16)  432         input_1[0][0]                    
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 128, 128, 16)  512         convolution2d_1[0][0]            
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 128, 128, 16)  0           batchnormalization_1[0][0]       
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 128, 128, 12)  1728        activation_1[0][0]               
____________________________________________________________________________________________________
merge_1 (Merge)                  (None, 128, 128, 28)  0           convolution2d_1[0][0]            
                                                                   convolution2d_2[0][0]            
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 128, 128, 28)  512         merge_1[0][0]                    
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 128, 128, 28)  0           batchnormalization_2[0][0]       
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 128, 128, 12)  3024        activation_2[0][0]               
____________________________________________________________________________________________________
merge_2 (Merge)                  (None, 128, 128, 40)  0           convolution2d_1[0][0]            
                                                                   convolution2d_2[0][0]            
                                                                   convolution2d_3[0][0]            
____________________________________________________________________________________________________
batchnormalization_3 (BatchNorma (None, 128, 128, 40)  512         merge_2[0][0]                    
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 128, 128, 40)  0           batchnormalization_3[0][0]       
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 128, 128, 12)  4320        activation_3[0][0]               
____________________________________________________________________________________________________
merge_3 (Merge)                  (None, 128, 128, 52)  0           convolution2d_1[0][0]            
                                                                   convolution2d_2[0][0]            
                                                                   convolution2d_3[0][0]            
                                                                   convolution2d_4[0][0]            
____________________________________________________________________________________________________
batchnormalization_4 (BatchNorma (None, 128, 128, 52)  512         merge_3[0][0]                    
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 128, 128, 52)  0           batchnormalization_4[0][0]       
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 128, 128, 52)  2704        activation_4[0][0]               
____________________________________________________________________________________________________
averagepooling2d_1 (AveragePooli (None, 64, 64, 52)    0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
batchnormalization_5 (BatchNorma (None, 64, 64, 52)    256         averagepooling2d_1[0][0]         
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 64, 64, 52)    0           batchnormalization_5[0][0]       
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 64, 64, 12)    5616        activation_5[0][0]               
____________________________________________________________________________________________________
merge_4 (Merge)                  (None, 64, 64, 64)    0           averagepooling2d_1[0][0]         
                                                                   convolution2d_6[0][0]            
____________________________________________________________________________________________________
batchnormalization_6 (BatchNorma (None, 64, 64, 64)    256         merge_4[0][0]                    
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 64, 64, 64)    0           batchnormalization_6[0][0]       
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 64, 64, 12)    6912        activation_6[0][0]               
____________________________________________________________________________________________________
merge_5 (Merge)                  (None, 64, 64, 76)    0           averagepooling2d_1[0][0]         
                                                                   convolution2d_6[0][0]            
                                                                   convolution2d_7[0][0]            
____________________________________________________________________________________________________
batchnormalization_7 (BatchNorma (None, 64, 64, 76)    256         merge_5[0][0]                    
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 64, 64, 76)    0           batchnormalization_7[0][0]       
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 64, 64, 12)    8208        activation_7[0][0]               
____________________________________________________________________________________________________
merge_6 (Merge)                  (None, 64, 64, 88)    0           averagepooling2d_1[0][0]         
                                                                   convolution2d_6[0][0]            
                                                                   convolution2d_7[0][0]            
                                                                   convolution2d_8[0][0]            
____________________________________________________________________________________________________
batchnormalization_8 (BatchNorma (None, 64, 64, 88)    256         merge_6[0][0]                    
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 64, 64, 88)    0           batchnormalization_8[0][0]       
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 64, 64, 88)    7744        activation_8[0][0]               
____________________________________________________________________________________________________
averagepooling2d_2 (AveragePooli (None, 32, 32, 88)    0           convolution2d_9[0][0]            
____________________________________________________________________________________________________
batchnormalization_9 (BatchNorma (None, 32, 32, 88)    128         averagepooling2d_2[0][0]         
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 32, 32, 88)    0           batchnormalization_9[0][0]       
____________________________________________________________________________________________________
convolution2d_10 (Convolution2D) (None, 32, 32, 12)    9504        activation_9[0][0]               
____________________________________________________________________________________________________
merge_7 (Merge)                  (None, 32, 32, 100)   0           averagepooling2d_2[0][0]         
                                                                   convolution2d_10[0][0]           
____________________________________________________________________________________________________
batchnormalization_10 (BatchNorm (None, 32, 32, 100)   128         merge_7[0][0]                    
____________________________________________________________________________________________________
activation_10 (Activation)       (None, 32, 32, 100)   0           batchnormalization_10[0][0]      
____________________________________________________________________________________________________
convolution2d_11 (Convolution2D) (None, 32, 32, 12)    10800       activation_10[0][0]              
____________________________________________________________________________________________________
merge_8 (Merge)                  (None, 32, 32, 112)   0           averagepooling2d_2[0][0]         
                                                                   convolution2d_10[0][0]           
                                                                   convolution2d_11[0][0]           
____________________________________________________________________________________________________
batchnormalization_11 (BatchNorm (None, 32, 32, 112)   128         merge_8[0][0]                    
____________________________________________________________________________________________________
activation_11 (Activation)       (None, 32, 32, 112)   0           batchnormalization_11[0][0]      
____________________________________________________________________________________________________
convolution2d_12 (Convolution2D) (None, 32, 32, 12)    12096       activation_11[0][0]              
____________________________________________________________________________________________________
merge_9 (Merge)                  (None, 32, 32, 124)   0           averagepooling2d_2[0][0]         
                                                                   convolution2d_10[0][0]           
                                                                   convolution2d_11[0][0]           
                                                                   convolution2d_12[0][0]           
____________________________________________________________________________________________________
batchnormalization_12 (BatchNorm (None, 32, 32, 124)   128         merge_9[0][0]                    
____________________________________________________________________________________________________
activation_12 (Activation)       (None, 32, 32, 124)   0           batchnormalization_12[0][0]      
____________________________________________________________________________________________________
globalaveragepooling2d_1 (Global (None, 124)           0           activation_12[0][0]              
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 500)           62500       globalaveragepooling2d_1[0][0]   
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           50100       dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             101         dense_2[0][0]                    
====================================================================================================
Total params: 189,373
Trainable params: 187,581
Non-trainable params: 1,792
____________________________________________________________________________________________________
None
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 128, 128, 3)   0                                            
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 128, 128, 16)  432         input_1[0][0]                    
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 128, 128, 16)  512         convolution2d_1[0][0]            
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 128, 128, 16)  0           batchnormalization_1[0][0]       
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 128, 128, 12)  1728        activation_1[0][0]               
____________________________________________________________________________________________________
merge_1 (Merge)                  (None, 128, 128, 28)  0           convolution2d_1[0][0]            
                                                                   convolution2d_2[0][0]            
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 128, 128, 28)  512         merge_1[0][0]                    
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 128, 128, 28)  0           batchnormalization_2[0][0]       
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 128, 128, 12)  3024        activation_2[0][0]               
____________________________________________________________________________________________________
merge_2 (Merge)                  (None, 128, 128, 40)  0           convolution2d_1[0][0]            
                                                                   convolution2d_2[0][0]            
                                                                   convolution2d_3[0][0]            
____________________________________________________________________________________________________
batchnormalization_3 (BatchNorma (None, 128, 128, 40)  512         merge_2[0][0]                    
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 128, 128, 40)  0           batchnormalization_3[0][0]       
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 128, 128, 12)  4320        activation_3[0][0]               
____________________________________________________________________________________________________
merge_3 (Merge)                  (None, 128, 128, 52)  0           convolution2d_1[0][0]            
                                                                   convolution2d_2[0][0]            
                                                                   convolution2d_3[0][0]            
                                                                   convolution2d_4[0][0]            
____________________________________________________________________________________________________
batchnormalization_4 (BatchNorma (None, 128, 128, 52)  512         merge_3[0][0]                    
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 128, 128, 52)  0           batchnormalization_4[0][0]       
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 128, 128, 52)  2704        activation_4[0][0]               
____________________________________________________________________________________________________
averagepooling2d_1 (AveragePooli (None, 64, 64, 52)    0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
batchnormalization_5 (BatchNorma (None, 64, 64, 52)    256         averagepooling2d_1[0][0]         
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 64, 64, 52)    0           batchnormalization_5[0][0]       
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 64, 64, 12)    5616        activation_5[0][0]               
____________________________________________________________________________________________________
merge_4 (Merge)                  (None, 64, 64, 64)    0           averagepooling2d_1[0][0]         
                                                                   convolution2d_6[0][0]            
____________________________________________________________________________________________________
batchnormalization_6 (BatchNorma (None, 64, 64, 64)    256         merge_4[0][0]                    
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 64, 64, 64)    0           batchnormalization_6[0][0]       
____________________________________________________________________________________________________
convolution2d_7 (Convolution2D)  (None, 64, 64, 12)    6912        activation_6[0][0]               
____________________________________________________________________________________________________
merge_5 (Merge)                  (None, 64, 64, 76)    0           averagepooling2d_1[0][0]         
                                                                   convolution2d_6[0][0]            
                                                                   convolution2d_7[0][0]            
____________________________________________________________________________________________________
batchnormalization_7 (BatchNorma (None, 64, 64, 76)    256         merge_5[0][0]                    
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 64, 64, 76)    0           batchnormalization_7[0][0]       
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 64, 64, 12)    8208        activation_7[0][0]               
____________________________________________________________________________________________________
merge_6 (Merge)                  (None, 64, 64, 88)    0           averagepooling2d_1[0][0]         
                                                                   convolution2d_6[0][0]            
                                                                   convolution2d_7[0][0]            
                                                                   convolution2d_8[0][0]            
____________________________________________________________________________________________________
batchnormalization_8 (BatchNorma (None, 64, 64, 88)    256         merge_6[0][0]                    
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 64, 64, 88)    0           batchnormalization_8[0][0]       
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 64, 64, 88)    7744        activation_8[0][0]               
____________________________________________________________________________________________________
averagepooling2d_2 (AveragePooli (None, 32, 32, 88)    0           convolution2d_9[0][0]            
____________________________________________________________________________________________________
batchnormalization_9 (BatchNorma (None, 32, 32, 88)    128         averagepooling2d_2[0][0]         
____________________________________________________________________________________________________
activation_9 (Activation)        (None, 32, 32, 88)    0           batchnormalizatio
```
