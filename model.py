
# coding: utf-8

# In[1]:

import zipfile

zip_ref=zipfile.ZipFile('images_bridge.zip')
zip_ref.extractall()


# # Reading CSV File

# In[14]:

import numpy as np
import csv
import cv2
from sklearn.model_selection import train_test_split

samples=[]

with open('driving_log_bridge.csv', 'r') as f:
    reader=csv.reader(f)
    
    for row in reader:
        samples.append(row)
        
with open('driving_log_given.csv', 'r') as f:
    reader=csv.reader(f)
    
    for row in reader:
        samples.append(row)

train_samples, validation_samples=train_test_split(samples, test_size=0.06)   
print(len(train_samples)*3)
print(len(validation_samples)*3)


# # Creating an image generator

# In[37]:

from sklearn.utils import shuffle
import matplotlib.pyplot as plt
    
def generator(samples_list, batch_size=32):
    num_samples=len(samples_list)
    
    while 1:
        shuffle(samples_list)
        for offset in range(0,num_samples, batch_size ):
            batch_samples=samples_list[offset:offset+batch_size]
            images=[]
            angles=[]
            
            for batch_sample in batch_samples:
                root='IMG_given/'
                
                center_image = cv2.imread(root+batch_sample[0].split('/')[-1])
                center_image=cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                
                left_image =cv2.imread(root+batch_sample[1].split('/')[-1])
                left_image=cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                
                right_image = cv2.imread(root+batch_sample[2].split('/')[-1])
                right_image=cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
                
                center_angle=float(batch_sample[3])
                correction=0.2
                
                # Angle correction for left and right camera outputs.
                left_angle=center_angle+correction
                right_angle=center_angle-correction
                
                # Appending to the image batch.
                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])
                
                aug_center=cv2.flip(center_image,1)
                aug_center_angle=center_angle*(-1)
                
                images.append(aug_center)
                angles.append(aug_center_angle)
                                        
            X_train=np.array(images, dtype=np.float)
            y_train=np.array(angles, dtype=np.float)
            yield shuffle(X_train, y_train)


# In[38]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

a=next(generator(samples, 32))
print(a[0].shape)
plt.imshow(a[0][20])
print(a[1].shape)


# # Preprocessing and Model Architecture

# In[19]:

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, Activation
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2

batch_size=32

model=Sequential()

# Preprocessing

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0)))) # Dims changed from (35,40)


model.add(Convolution2D(24, 5, 5, subsample=(2,2), W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Convolution2D(36, 5, 5, subsample=(2,2), W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Convolution2D(48,5,5,subsample=(2,2), W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Convolution2D(64, 3, 3, W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Convolution2D(64, 3, 3, W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Flatten())

model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Dense(1))

# Defining generators.

train_generator=generator(train_samples, batch_size)
validation_generator=generator(validation_samples, batch_size)


# # Training the model.

# In[20]:

model.compile(loss='mse', optimizer='adam')

history_object=model.fit_generator(train_generator,samples_per_epoch=len(train_samples)*4, nb_epoch=10, 
                                   validation_data=validation_generator, nb_val_samples=len(validation_samples))

model.save('model_prefinal_plus_bridge_test.h5')
print('Done!')


# # Plotting the losses.

# In[30]:

import matplotlib.pyplot as plt
# %matplotlib inline

curve1=plt.plot(history_object.history['loss'], label='Training Loss')
curve2=plt.plot(history_object.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Error')

