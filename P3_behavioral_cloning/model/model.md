
# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report (This is that written report.)

## Rubric Points
** Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  **

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md - summary of results
* model.ipynb - source document for the writeup.md and model.py. 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Overall process

1. Create training data by driving around the track.
2. Design and test different model archetectures.
3. Tune model parameters.
4. Test model on track. 


## 1. Creation of the Training Set & Training Process

To capture training data I drop around track 1 three times using center lain driving. To help the car recover if it got off the track I collected training data driving on the left and right side of the track for a total of 6 laps. To create even more training data I drove the same center, left and right courses going in the reverse direction. 

Here is the code I used to visualize the collected training data.


```python
import cv2
import os
import pandas as pd
import numpy as np
from PIL import Image
import random
from matplotlib import pyplot as plt
%matplotlib inline
```


```python
#Constants
DATA_PATH = '/home/ubuntu/udacity_projects/P3_behavioral_cloning/driving_data/'
SIDE_CAMERA_ADJUSTMENT = .06
SIDE_DRIVING_ADJUSTMENT = .3
```


```python
def convert_paths(df, folder):
    #convert paths of datalog to those on local computer
    for i in range(len(df)):
        for col in ['img_center', 'img_left', 'img_right']:
            file = df.ix[i, col].split('/')[-1]
            img_path = os.path.join(folder, 'IMG', file)
            df.ix[i, col] = img_path
    return df

def load_datasets(folder_nums):
    #load datasets from multiple folders
    dataframes = []
    for num in folder_nums:
        folder = os.path.join(DATA_PATH, 'track_'+str(num))
        df = pd.read_csv(os.path.join(folder, 'driving_log.csv'))
        df.columns = ['img_center', 'img_left', 'img_right', 'steering', 'throttle', 'break', 'speed']
        df = convert_paths(df, folder)
        dataframes.append(df)
    df = pd.concat(dataframes)
    return df

#Load the datasets.
df_center = load_datasets([11, 14, 18])
df_left = load_datasets([13, 15])
df_right = load_datasets([12, 16])

df = pd.concat([df_center, df_left, df_right])
df = df.reset_index(drop=True)
df['steering'].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f49bc5455f8>




![png](output_6_1.png)


Since I didn't want the car to go to drive on the side of track, I adjusted the training data where I drove on the side to have higher steering angles back toward the center of the track. The training data collected on the right side had an adjustment of + .3 and the left training data had an adjustment of -.3.


```python
#Here we adjust the steering angles of the training data where I 
#drove on the side of the track. 
df_left['steering'] = df_left['steering'] + SIDE_DRIVING_ADJUSTMENT
df_right['steering'] = df_right['steering'] - SIDE_DRIVING_ADJUSTMENT
df = pd.concat([df_center, df_left, df_right])
df = df.reset_index(drop=True)
df['steering'].plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f49bc545080>




![png](output_8_1.png)



```python
df['steering'].hist(bins=81)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f49b95b9710>




![png](output_9_1.png)



```python
#df_curves['steering'].hist(bins=81)
```


```python
def get_data(df, row, img_col, steering_adjust):
    img_path = df[img_col][row]
    #print(img_path)
    img = Image.open(img_path)
    img_arr = np.array(img)
    raw_angle = df['steering'][row]
    #print('raw_angle ', raw_angle)
    angle = raw_angle + steering_adjust
    #print('adjusted angle', angle)
    return img_arr, angle
```

To prepare the data for training the neural network I used a generator to create batches of image and steering angle pairs. For each data row the generator selects a random camera angel (left,center or right) and adjustes the steering angel to bring the car back to center and flips the image horizontally and reverses the steerign angle half the time to create more data inputs to help avoid over fitting. 


```python
def data_gen(df, batch_size=64, sample_ix=None):
    arg_choices = [
                    {'img_col': 'img_center', 'adjust': 0.0},
                    {'img_col': 'img_left',   'adjust': SIDE_CAMERA_ADJUSTMENT},
                    {'img_col': 'img_right',  'adjust': -SIDE_CAMERA_ADJUSTMENT},]    

    while True:
        images=[]
        angles=[]
        rows = sample_ix[:batch_size]
        for i in rows:
            #randomly select a camera angle.
            args = random.choice(arg_choices)
            #get the numpy array and adjust the steering angle.
            img, angle = get_data(df, i, args['img_col'], steering_adjust=args['adjust'])

            #half the time flip the steering angle. 
            flip = random.randint(0, 1)
            if flip == 1:
                img = np.fliplr(img)
                angle = -angle
                
            images.append(img)
            angles.append(angle)

        #create the batch of images and steering images for keras.
        X_train = np.array(images)
        y_train = np.array(angles)
        yield X_train, y_train
```

To avoid overfitting we split the data set into a training set that we use to train the model and a validation set that the model has never seen that is used to test if the model has learned something other than memorizing the input data. 


```python
#create and index of the data rows
df = df.reset_index(drop=True)
n = len(df)
sample_ix = list(range(n))
random.shuffle(sample_ix)

#separate index into train and validation
train_n = round(n * .8)
train_ix = sample_ix[:train_n]
val_ix = sample_ix[train_n:]

#create data generators with those indexes
train_gen = data_gen(df, sample_ix=train_ix)
val_gen = data_gen(df, sample_ix=val_ix)
```

Here are some examples of images and steering angles used in the trainign data.


```python
X, y = next(train_gen)
i = 0
plt.imshow(X[i])
print(y[i])
```

    -0.20226415



![png](output_17_1.png)



```python
i = 2
plt.imshow(X[i])
print(y[i])
```

    0.17396226



![png](output_18_1.png)


## Build a model

Now that we have a data pipeline we want to build a network archetecture that will learn how to drive. To do this I used a similar archetecture as I've used on my Donkey Car that I race in DIY Robocars races. 

It has Several layers of convolutions and dense layers with a single output for the angle. Some of the characteristics of model include.
* It has elu and relu activation functions to introduce non-linearity. 
* The convolutions use strides that are longer in the X dimension.
* To avoid over fitting I use L2 regularizers on most of the layers. 
* It uses an adam optimizer since this has become the standard for Convlution based nets.


```python
import keras
from keras.layers import Input, Dense, merge
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, AveragePooling2D, MaxPool2D, SimpleRNN, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Cropping2D, Lambda
from keras.regularizers import l2

def build_model1():
    '''
    Function to create models with convolutional heads and dense tails.
    Accepts dictionaries defining the conv and dense layers.
    '''

    img_in = Input(shape=(160, 320,3), name='img_in')
    x = Lambda(lambda x: (x / 255.0) - 0.5)(img_in)
    x = Cropping2D(cropping=((60, 20), (0, 0)))(x)
    #x = AveragePooling2D(2,2)(x)
    x = Convolution2D(16, (4,4), strides=(3,4), activation='elu', kernel_regularizer=l2(0.001))(x)
    x = Convolution2D(32, (4,4), strides=(1,2), activation='elu', kernel_regularizer=l2(0.001))(x)
    x = Convolution2D(48, (4,4), strides=(1,2), activation='elu', kernel_regularizer=l2(0.001))(x)
    x = Convolution2D(64, (3,3), activation='elu', kernel_regularizer=l2(0.001))(x)
    x = Convolution2D(64, (3,3), activation='elu', kernel_regularizer=l2(0.001))(x)
    x = Convolution2D(64, (3,3), activation='elu', kernel_regularizer=l2(0.001))(x)
    x = Flatten(name='flattened')(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.002))(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.002))(x)
    x = Dense(10, activation='relu', kernel_regularizer=l2(0.002))(x)
    angle = Dense(1)(x)
    
    model = Model(inputs=[img_in], outputs=[angle])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

model = build_model1()
```

    Using TensorFlow backend.


Here is the structure of the model and the number of parameters.


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    img_in (InputLayer)          (None, 160, 320, 3)       0         
    _________________________________________________________________
    lambda_1 (Lambda)            (None, 160, 320, 3)       0         
    _________________________________________________________________
    cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 26, 80, 16)        784       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 23, 39, 32)        8224      
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 20, 18, 48)        24624     
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 18, 16, 64)        27712     
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 16, 14, 64)        36928     
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 14, 12, 64)        36928     
    _________________________________________________________________
    flattened (Flatten)          (None, 10752)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                688192    
    _________________________________________________________________
    dense_2 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                330       
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 11        
    =================================================================
    Total params: 825,813
    Trainable params: 825,813
    Non-trainable params: 0
    _________________________________________________________________


To train the net I feed use the training and validation generator to train and test the network. Callbacks are used to save the model with the lowest validation error and stop when the validation error stops improving. 


```python
#checkpoint to save model after each epoch
save_best = keras.callbacks.ModelCheckpoint('model.hdf5', monitor='val_loss', verbose=1, 
                                      save_best_only=True, mode='min')

#stop training if the validation error stops improving.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=.0005, patience=2, 
                                     verbose=1, mode='auto')

callbacks_list = [save_best, early_stop]

model.fit_generator(train_gen, 100, epochs=50, validation_data=val_gen, 
                    callbacks=callbacks_list, validation_steps=40)
```

    Epoch 1/50
     99/100 [============================>.] - ETA: 0s - loss: 0.4914Epoch 00000: val_loss improved from inf to 0.37319, saving model to model.hdf5
    100/100 [==============================] - 119s - loss: 0.4897 - val_loss: 0.3732
    Epoch 2/50
     99/100 [============================>.] - ETA: 0s - loss: 0.2516Epoch 00001: val_loss improved from 0.37319 to 0.25847, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.2511 - val_loss: 0.2585
    Epoch 3/50
     99/100 [============================>.] - ETA: 0s - loss: 0.1674Epoch 00002: val_loss improved from 0.25847 to 0.20346, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.1672 - val_loss: 0.2035
    Epoch 4/50
     99/100 [============================>.] - ETA: 0s - loss: 0.1236Epoch 00003: val_loss improved from 0.20346 to 0.16933, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.1234 - val_loss: 0.1693
    Epoch 5/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0972Epoch 00004: val_loss improved from 0.16933 to 0.15190, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0972 - val_loss: 0.1519
    Epoch 6/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0813Epoch 00005: val_loss improved from 0.15190 to 0.13552, saving model to model.hdf5
    100/100 [==============================] - 17s - loss: 0.0812 - val_loss: 0.1355
    Epoch 7/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0713Epoch 00006: val_loss improved from 0.13552 to 0.13187, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0712 - val_loss: 0.1319
    Epoch 8/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0615Epoch 00007: val_loss improved from 0.13187 to 0.12012, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0615 - val_loss: 0.1201
    Epoch 9/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0579Epoch 00008: val_loss did not improve
    100/100 [==============================] - 16s - loss: 0.0579 - val_loss: 0.1204
    Epoch 10/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0522Epoch 00009: val_loss improved from 0.12012 to 0.11295, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0521 - val_loss: 0.1129
    Epoch 11/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0477Epoch 00010: val_loss improved from 0.11295 to 0.11060, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0477 - val_loss: 0.1106
    Epoch 12/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0458Epoch 00011: val_loss improved from 0.11060 to 0.10860, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0458 - val_loss: 0.1086
    Epoch 13/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0447Epoch 00012: val_loss improved from 0.10860 to 0.10560, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0447 - val_loss: 0.1056
    Epoch 14/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0421Epoch 00013: val_loss improved from 0.10560 to 0.10110, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0421 - val_loss: 0.1011
    Epoch 15/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0398Epoch 00014: val_loss did not improve
    100/100 [==============================] - 16s - loss: 0.0398 - val_loss: 0.1046
    Epoch 16/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0378Epoch 00015: val_loss improved from 0.10110 to 0.09922, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0378 - val_loss: 0.0992
    Epoch 17/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0371Epoch 00016: val_loss did not improve
    100/100 [==============================] - 16s - loss: 0.0370 - val_loss: 0.0998
    Epoch 18/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0347Epoch 00017: val_loss improved from 0.09922 to 0.09764, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0347 - val_loss: 0.0976
    Epoch 19/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0355Epoch 00018: val_loss did not improve
    100/100 [==============================] - 16s - loss: 0.0355 - val_loss: 0.0990
    Epoch 20/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0349Epoch 00019: val_loss improved from 0.09764 to 0.09450, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0349 - val_loss: 0.0945
    Epoch 21/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0340Epoch 00020: val_loss did not improve
    100/100 [==============================] - 16s - loss: 0.0340 - val_loss: 0.0950
    Epoch 22/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0328Epoch 00021: val_loss did not improve
    100/100 [==============================] - 16s - loss: 0.0327 - val_loss: 0.0973
    Epoch 23/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0322Epoch 00022: val_loss improved from 0.09450 to 0.09385, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0323 - val_loss: 0.0939
    Epoch 24/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0311Epoch 00023: val_loss did not improve
    100/100 [==============================] - 16s - loss: 0.0311 - val_loss: 0.0951
    Epoch 25/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0308Epoch 00024: val_loss improved from 0.09385 to 0.09332, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0307 - val_loss: 0.0933
    Epoch 26/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0301Epoch 00025: val_loss improved from 0.09332 to 0.09295, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0301 - val_loss: 0.0930
    Epoch 27/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0307Epoch 00026: val_loss improved from 0.09295 to 0.09289, saving model to model.hdf5
    100/100 [==============================] - 16s - loss: 0.0307 - val_loss: 0.0929
    Epoch 28/50
     99/100 [============================>.] - ETA: 0s - loss: 0.0298Epoch 00027: val_loss did not improve
    100/100 [==============================] - 16s - loss: 0.0298 - val_loss: 0.0964
    Epoch 00027: early stopping





    <keras.callbacks.History at 0x7f4952263ef0>



Here's the result of the driving auto pilot. 


```python
%%HTML
<video width="320" height="240" controls>
  <source src="track_1_video.mp4" 
    type="video/mp4">
</video>
```


<video width="320" height="240" controls>
  <source src="track_1_video.mp4" 
    type="video/mp4">
</video>


## Future Improvements.

Unfortunately I'm rushed to finish my last homeworks assigments so I'm not experimenting as much as I'd like. Later I'd like to come back on try the following things.

* Use image augmentation (blur, shift, ..) to make the model more robust.
* train the throttle so the car slows down on the corners and speeds up on the straights.
* See how small the model could get and still work. Try using Devol (genetic algo) to find the best network.


```python

```
