
# coding: utf-8

# # Face Recognition
# 
# 

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks import *

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

np.set_printoptions(threshold=np.nan)


# In[2]:


def faceRecoModel(input_shape):
    """
    Implementation of the Inception model used for FaceNet
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
        
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # First Block
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 1, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides = 2)(X)
    
    # Second Block
    X = Conv2D(64, (1, 1), strides = (1, 1), name = 'conv2')(X)
    X = BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)

    # Second Block
    X = Conv2D(192, (3, 3), strides = (1, 1), name = 'conv3')(X)
    X = BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D(pool_size = 3, strides = 2)(X)
    
    # Inception 1: a/b/c
    X = inception_block_1a(X)
    X = inception_block_1b(X)
    X = inception_block_1c(X)
    
    # Inception 2: a/b
    X = inception_block_2a(X)
    X = inception_block_2b(X)
    
    # Inception 3: a/b
    X = inception_block_3a(X)
    X = inception_block_3b(X)
    
    # Top layer
    X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first')(X)
    X = Flatten()(X)
    X = Dense(128, name='dense_layer')(X)
    
    # L2 normalization
    X = Lambda(lambda  x: K.l2_normalize(x,axis=1))(X)

    # Create model instance
    model = Model(inputs = X_input, outputs = X, name='FaceRecoModel')
        
    return model


# In[3]:


FRmodel = faceRecoModel(input_shape=(3, 96, 96))


# In[4]:


def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, here it is a dummy variable required by Keras
    y_pred -- python list containing three objects:
            anchor -- the embeddings for the anchor images, of shape (None, 128)
            positive -- the embeddings for the positive images, of shape (None, 128)
            negative -- the embeddings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
   
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))

    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
   
    
    return loss


# In[5]:


FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])


# In[6]:


load_weights_from_FaceNet(FRmodel)


# In[8]:


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

X_train = X_train.transpose((0, 3, 1, 2))
X_test = X_test.transpose((0, 3, 1, 2))
#Y_train = Y_train.transpose()
#Y_test = Y_test.transpose((0, 3, 1, 2))

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


# In[33]:


# Example of a picture
index = 12
plt.imshow(X_train.transpose((0,2,3,1))[index])
print ("y = " + str(Y_train[index, :]))


# In[99]:


# Build database from X_train and Y_train
# database.shape = (m, 128)

m = X_train.shape[0]
database_emb = np.zeros((m, 128))
database_names = Y_train

for i in range(m):
    image = X_train[i,:,:,:]
    embedding = img1_to_embedding(image)
    database_emb[i,:] = embedding


# In[105]:


#KNN on database with new image
new_emb = img_to_embedding("images/test_4.jpg")


# In[ ]:


def img_to_embedding(image_path):
    img1 = cv2.imread(image_path, 1)
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = FRmodel.predict_on_batch(x_train)
    return embedding


# In[59]:


def img1_to_embedding(img1):
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = FRmodel.predict_on_batch(x_train)
    return embedding


# In[43]:


def verify(image_path, identity):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    X_train --
    Y_train --
    
    Returns:
    dist --
    door_open -- True, if the door should open. False otherwise.
    """
    
    # Compute the embedding for the image
    embedding = img_to_embedding(image_path)
    
    # Compute distance with identity's image
    dist = np.linalg.norm(database[identity] - embedding)
    
    if dist < 0.5:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
        
    return dist, door_open


# In[44]:


verify("images/test_4.jpg", "kian")


# In[50]:


img1 = cv2.imread('images/test_a.jpg', 1)
img2 = cv2.imread('images/test_p.jpg', 1)
img3 = cv2.imread('images/test_n.jpg', 1)


# In[51]:


# Read a sample image as input to test the model
img = img1[...,::-1]
img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
x_train = np.array([img])
y_a = FRmodel.predict_on_batch(x_train)
print (y_a)


# In[48]:


# Read a sample image as input to test the model
img = img2[...,::-1]
img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
x_train = np.array([img])
y_p = FRmodel.predict_on_batch(x_train)
print (y_p)


# In[49]:


# Read a sample image as input to test the model
img = img3[...,::-1]
img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
x_train = np.array([img])
y_n = FRmodel.predict_on_batch(x_train)
print (y_n)


# In[ ]:


diff_ap = np.linalg.norm(y_a - y_p)
diff_an = np.linalg.norm(y_a - y_n)
diff_pn = np.linalg.norm(y_n - y_p)
print(diff_ap, diff_an, diff_pn)


# In[100]:


def who_is_it(image_path, database):
    
    # Compute the embedding for the image
    embedding = img_to_embedding(image_path)
    
    # Find the closest embedding
    closest = None
    min_dist = 100
    for (name, emb) in database.items():
        dist = np.linalg.norm(embedding - emb)
        print(dist, name)
        if dist < min_dist:
            min_dist = dist
            identity = name
            
    if min_dist > 0.6:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
    
    return min_dist, identity


# In[101]:


who_is_it('images/test_5.jpg', database)


# In[ ]:


FRmodel.fit(x = X_train, y = Y_train, epochs = 40, batch_size = 16)

