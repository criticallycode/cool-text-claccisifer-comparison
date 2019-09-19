# cool-text-claccisifer-comparison
A comparison of text classification methods.

import random as rn
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Flatten, Activation
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# separating training data into different folders

images = []
labels = []

img_size = 150

def training_data(im_class, dir):
    # tqdm is for the display of progress bars
    # for the image in the listed directory
    for img in tqdm(os.listdir(dir)):
        label = im_class
        path = os.path.join(dir, img)
        _, ftype = os.path.splitext(path)
        if ftype == ".jpg":
            # read in respective image as a color image
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            # rsize the respective image
            img = cv2.resize(img, (img_size, img_size))

            # make a numpy array out of the image
            images.append(np.array(img))
            labels.append(str(label))

training_data('Daisy', 'C:/Users/Daniel/Downloads/flowers-recognition/flowers/daisy')
print(len(images))

training_data('Sunflower', 'C:/Users/Daniel/Downloads/flowers-recognition/flowers/sunflower')
print(len(images))

training_data('Tulip', 'C:/Users/Daniel/Downloads/flowers-recognition/flowers/tulip')
print(len(images))

training_data('Dandelion', 'C:/Users/Daniel/Downloads/flowers-recognition/flowers/dandelion')
print(len(images))

training_data('Rose', 'C:/Users/Daniel/Downloads/flowers-recognition/flowers/rose')
print(len(images))

# do label encoding
encoder = LabelEncoder()
y_labels = encoder.fit_transform(labels)
# one hot encoding
y_labels = to_categorical(y_labels, 5)
x_features = np.array(images)
# data normalization
x_features = x_features/255

X_train, X_test, y_train, y_test = train_test_split(x_features, y_labels, test_size=0.25, random_state=27)

np.random.seed(27)
rn.seed(27)
# graph level random seed for TF/Keras
tf.set_random_seed(27)

# will make 5 x 2 subplots
fig, ax = plt.subplots(2, 5)
# set a specific size for the plot
fig.set_size_inches(15, 15)
# specify plotting on both axis
for i in range(2):
    for j in range(5):
        # get a random integer between one and the final label in the list of labels
        label = rn.randint(0, len(labels))
        ax[i, j].imshow(images[label])
        ax[i, j].set_title('Flower: '+labels[label])
plt.tight_layout()
plt.show()

# because the model is so large/deep it may be a good idea to perform some data augmentation in order
# combat overfitting
data_generator = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2,
                                    horizontal_flip=True)

# fit on the trainign data to produce more data with the specified transformations
data_generator.fit(X_train)

def create_model():
    # first specify the sequential nature of the model
    model = Sequential()
    # conv2d is the convolutional layer for 2d images
    # first parameter is the number of memory cells - let's just try 64 units for now
    # second parameter is the size of the "window" you want the CNN to use
    # the shape of the data we are passing in, 3 x 150 x 150
    model.add(Conv2D(64, (5, 5), input_shape=(150, 150, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3),  padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # flatten the data for the dense layer
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    # now compile the model, specify loss, optimization, etc
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()

batch_size = 64
num_epochs = 50

# fit the generator, since we used one in making new data

filepath = "weights_custom.hdf5"
callbacks = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
              ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=0.00001),
             EarlyStopping(monitor= 'val_loss', min_delta=1e-10, patience=15, verbose=1, restore_best_weights=True)]

train_records = model.fit_generator(data_generator.flow(X_train, y_train, batch_size=batch_size), epochs = num_epochs,
                          validation_data=(X_test, y_test), verbose= 1, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=callbacks)

# visualize training loss
# declare important variables
training_acc = train_records.history['acc']
val_acc = train_records.history['val_acc']
training_loss = train_records.history['loss']
validation_loss = train_records.history['val_loss']

# gets the lengt of how long the model was trained for
train_length = range(1, len(training_acc) + 1)

def plot_stats(train_length, training_acc, val_acc, training_loss, validation_loss):

    # plot the loss across the number of epochs
    plt.figure()
    plt.plot(train_length, training_loss, label='Training Loss')
    plt.plot(train_length, validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    plt.plot(train_length, training_acc, label='Training Accuracy')
    plt.plot(train_length, val_acc, label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def make_preds(model):
    # make predictions
    score = model.evaluate(X_test, y_test, verbose=0)
    print('\nAchieved Accuracy:', score[1],'\n')

    y_pred = model.predict(X_test)
    # evalute model predictions
    Y_pred_classes = np.argmax(y_pred, axis=1)
    Y_true = np.argmax(y_test, axis=1)
    confusion = confusion_matrix(Y_true, Y_pred_classes)
    print(confusion)

plot_stats(train_length, training_acc, val_acc, training_loss, validation_loss)
make_preds(model)

# Create the base model and add it to our sequential framework
resnet = ResNet50(weights='imagenet',
                 include_top=False,
                 input_shape=(150, 150, 3))
print(resnet.summary())
resnet_model = Sequential()
resnet_model.add(resnet)

# Add in our own densely connected layers (after flattening the inputs)
resnet_model.add(Flatten())
resnet_model.add(Dense(256, activation='relu'))
resnet_model.add(Dropout(0.2))
resnet_model.add(Dense(64, activation='relu'))
resnet_model.add(Dropout(0.2))
resnet_model.add(Dense(5, activation='softmax'))
resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(resnet_model.summary())

num_epochs = 10

filepath = "weights_resnet50.hdf5"
callbacks = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
              ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=0.00001),
             EarlyStopping(monitor= 'val_loss', min_delta=1e-10, patience=15, verbose=1, restore_best_weights=True)]

resnet_records = resnet_model.fit_generator(data_generator.flow(X_train, y_train, batch_size=batch_size), epochs = num_epochs,
                          validation_data=(X_test, y_test), verbose= 1, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=callbacks)

# visualize training loss
# declare important variables
training_acc = resnet_records.history['acc']
val_acc = resnet_records.history['val_acc']
training_loss = resnet_records.history['loss']
validation_loss = resnet_records.history['val_loss']

# gets the length of how long the model was trained for
train_length = range(1, len(training_acc) + 1)

plot_stats(train_length, training_acc, val_acc, training_loss, validation_loss)
make_preds(resnet_model)


Incep = InceptionV3(weights='imagenet',
                 include_top=False,
                 input_shape=(150, 150, 3))
Incep_model = Sequential()
Incep_model.add(Incep)
Incep_model.add(Flatten())
Incep_model.add(Dense(256, activation='relu'))
Incep_model.add(Dropout(0.2))
Incep_model.add(Dense(64, activation='relu'))
Incep_model.add(Dropout(0.2))
Incep_model.add(Dense(5, activation='softmax'))
print(Incep_model.summary())
Incep_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath = "weights_Incep.hdf5"
callbacks = [ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
              ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, mode='min', min_lr=0.00001),
             EarlyStopping(monitor= 'val_loss', min_delta=1e-10, patience=15, verbose=1, restore_best_weights=True)]

num_epochs = 10

Incep_records = Incep_model.fit_generator(data_generator.flow(X_train, y_train, batch_size=batch_size), epochs = num_epochs,
                          validation_data=(X_test, y_test), verbose= 1, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=callbacks)

# visualize training loss
# declare important variables
training_acc = Incep_records.history['acc']
val_acc = Incep_records.history['val_acc']
training_loss = Incep_records.history['loss']
validation_loss = Incep_records.history['val_loss']

# gets the length of how long the model was trained for
train_length = range(1, len(training_acc) + 1)

plot_stats(train_length, training_acc, val_acc, training_loss, validation_loss)
make_preds(Incep_model)
