import numpy as np
import sys
import os
from math import ceil

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, MaxPooling2D, Flatten, Input, Activation, BatchNormalization, Dropout, Reshape
import matplotlib.pyplot as plt
from tensorflow.keras.losses import categorical_crossentropy, MAPE
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def create_cnn(width, height, depth, filters=(16, 16, 16, 16, 16)):
    input_shape = (height, width, depth)
    chan_dim = -1

    inputs = Input(shape=input_shape)
    for (i, filter) in enumerate(filters):
        if i == 0:
            x = inputs
            #x = Conv3D(filter, (3, 3, 1), padding="same", activation="relu")(x)
            #x = BatchNormalization(axis=chan_dim)(x)
            #x = Reshape(target_shape=(height,width,filter))(x)
        #else:
            # CONV => RELU => BN => POOL
        x = Conv2D(filter, (3, 3), padding="same", activation="relu")(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization(axis=chan_dim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model

load_model = False
train_model = True
save_model = True
test_model = True

print('creating model...')
model = create_cnn(256, 636, 1)
opt = Adam(lr=1e-3, decay=1e-3/200)
model.compile(loss="mean_squared_error", optimizer=opt)
print(model.summary())

if load_model:
    model.load_weights('new_model.h5')

if train_model:
    image_folder = 'train_data'
    angle_folder = 'train_data'
    print('loading training data...')
    vera_train = [f for f in os.listdir(image_folder) if f.endswith('-images.npy')]
    vico_train = [f for f in os.listdir(angle_folder) if f.endswith('-angles.npy')]

    for im,an in zip(vera_train,vico_train):
        print(im)
        im_data = np.load('{0}/{1}'.format(image_folder,im))[...,None]
        an_data = np.load('{0}/{1}'.format(angle_folder,an))[...,1]
        print(im_data.shape,an_data.shape)

        print('training model...')
        model.fit(im_data, an_data, validation_split=0.3, epochs=10)

if save_model:
    print('saving model...')
    model.save('new_model.h5', save_format='h5')

if test_model:
    image_folder = 'test_data'
    angle_folder = 'test_data'
    print('loading testing data...')
    vera_test = [f for f in os.listdir(image_folder) if f.endswith('-images.npy')]
    vico_test = [f for f in os.listdir(angle_folder) if f.endswith('-angles.npy')]
    fingers = ['thumb','index','middle','ring','pinky']

    for i,(im,an) in enumerate(zip(vera_test,vico_test)):
        print(im)
        im_data = np.load('{0}/{1}'.format(image_folder,im))[...,None]
        an_data = np.load('{0}/{1}'.format(angle_folder,an))[...,1]
        print(im_data.shape,an_data.shape)

        print('testing model...')
        predictions = model.predict(im_data)
        plt.subplot(len(vera_test),1,i+1)
        plt.plot(an_data),plt.plot(predictions)
        if i==0:
            plt.title("index finger angle")
        title = im.split('-')
        plt.ylabel('-'.join([title[i] for i in [0,2]]),rotation=90)
        print('{0} MSE {1}'.format(im,dict(MAPE(an_data,predictions))))
    plt.show()
