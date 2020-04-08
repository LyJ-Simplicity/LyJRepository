import tensorflow as tf
import os
import random
import numpy as np

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

seed = 42
np.random.seed = seed

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 4

TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)

print('Resizing training images and masks')
for n, id_ in tqdm(enumerate(train_ids), total = len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img,(IMG_HEIGHT, IMG_WIDTH),mode = 'constant', preserve_range = True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1),dtype = np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_,(IMG_HEIGHT, IMG_WIDTH), mode = 'constant',
                                      preserve_range = True), axis = -1)
        mask = np.maximum(mask,mask_)

    Y_train[n] = mask

#test images

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
sizes_test = []
print('Resizing testing images')
for n, id_ in tqdm(enumerate(test_ids), total = len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()

"""
#Let's see if things look all right by drawing some random images and their associated masks;
#Check if training data looks all right
ix = random.randint(o,len(train_ids))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
"""

#Build the model

inputs = tf.keras.layers.Input((IMG_WIDTH , IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
C1 = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(s)
C1 = tf.keras.layers.Dropout(0.1)(C1)
C1 = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(C1)
P1 = tf.keras.layers.MaxPooling2D(2,2)(C1)

C2 = tf.keras.layers.Conv2D(128,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(P1)
C2 = tf.keras.layers.Dropout(0.1)(C2)
C2 = tf.keras.layers.Conv2D(128,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(C2)
P2 = tf.keras.layers.MaxPooling2D(2,2)(C2)

C3 = tf.keras.layers.Conv2D(256,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(P2)
C3 = tf.keras.layers.Dropout(0.2)(C3)
C3 = tf.keras.layers.Conv2D(256,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(C3)
P3 = tf.keras.layers.MaxPooling2D(2,2)(C3)

C4 = tf.keras.layers.Conv2D(512,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(P3)
C4 = tf.keras.layers.Dropout(0.2)(C4)
C4 = tf.keras.layers.Conv2D(512,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(C4)
P4 = tf.keras.layers.MaxPooling2D(2,2)(C4)

C5 = tf.keras.layers.Conv2D(1024,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(P4)
C5 = tf.keras.layers.Dropout(0.3)(C5)
C5 = tf.keras.layers.Conv2D(1024,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(C5)

#Expansive path
U6 = tf.keras.layers.Conv2DTranspose(512,(2,2), strides = (2,2), padding = 'same')(C5)
U6 = tf.keras.layers.concatenate([U6, C4])
C6 = tf.keras.layers.Conv2D(512,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(U6)
C6 = tf.keras.layers.Dropout(0.2)(C6)
C6 = tf.keras.layers.Conv2D(512,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(C6)

U7 = tf.keras.layers.Conv2DTranspose(256,(2,2), strides = (2,2), padding = 'same')(C6)
U7 = tf.keras.layers.concatenate([U7, C3])
C7 = tf.keras.layers.Conv2D(256,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(U7)
C7 = tf.keras.layers.Dropout(0.2)(C7)
C7 = tf.keras.layers.Conv2D(256,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(C7)

U8 = tf.keras.layers.Conv2DTranspose(128,(2,2), strides = (2,2), padding = 'same')(C7)
U8 = tf.keras.layers.concatenate([U8, C2])
C8 = tf.keras.layers.Conv2D(128,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(U8)
C8 = tf.keras.layers.Dropout(0.1)(C8)
C8 = tf.keras.layers.Conv2D(128,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(C8)

U9 = tf.keras.layers.Conv2DTranspose(64,(2,2), strides = (2,2), padding = 'same')(C8)
U9 = tf.keras.layers.concatenate([U9, C1], axis = 3)
C9 = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(U9)
C9 = tf.keras.layers.Dropout(0.1)(C9)
C9 = tf.keras.layers.Conv2D(64,(3,3),activation = 'relu', kernal_initializer='he_normal', padding = 'same')(C9)

outputs = tf.keras.layers.Conv2D(1,(1,1), activation = 'sigmoid')(C9)

model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

##############################
#Model checkpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose = 1, save_best_only = True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir = 'log')]

results = model.fit(X_train, Y_train, validation_split = 0.1, batch_size = 16, epochs = 25, callbacks = callbacks)

##############################
#Perform a sanity check on some random training samples
#No need to load model if it is already in memory...
idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose = 1)
preds_val = model.predict(X_train[:int(X_train.shape[0]*0.9):], verbose = 1)
preds_test = model.predict(X_test, verbose = 1)

#Each pixel is given a value between 0 and 1,we set a threshold .5 to binarize.
#Threshold predictions to binarize the image
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

#Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

#Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()
