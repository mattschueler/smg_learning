import numpy as np
import sys
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, MaxPooling2D, Flatten, Input, Activation, BatchNormalization, Dropout, Reshape
import matplotlib.pyplot as plt
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


def create_cnn(width, height, depth, filters=(16, 16, 21, 64), regress=False):
    input_shape = (height, width, depth, 1)
    chan_dim = -1

    inputs = Input(shape=input_shape)


    # loop over the number of filters
    for (i, filter) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs
            x = Conv3D(filter, (5, 5, 4), strides = (1, 1, 4), padding="same", activation="relu")(x)
            x = BatchNormalization(axis=chan_dim)(x)
            x = Reshape(target_shape=(height,width,filter))(x)
        else:
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
    x = Dense(1, activation="relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model

print('loading data...')
ultrasound_data = np.load('ultrasound_data.npy')[...,None]
angle_data = np.load('angle_data.npy')

print('creating model...')
model = create_cnn(128, 310, 4, regress=True)
opt = Adam(lr=1e-3, decay=1e-3/200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
print(model.summary())

print("split data...")
data_len = ultrasound_data.shape[0]
split = int(data_len*0.7)
ang_train = angle_data[:split]
ang_test = angle_data[split:]
us_train = ultrasound_data[:split,...]
us_test = ultrasound_data[split:,...]

print(us_train.shape)
print(ang_train.shape)
print(us_test.shape)
print(ang_test.shape)

print('training model...')
model.fit(us_train, ang_train, validation_split=0.3, epochs=10)

print('saving model...')
model.save('complex_model.h5', save_format='h5')

print('predicting angles...')
predictions = model.predict(us_test)
plt.subplot(211),plt.plot(ang_test)
plt.subplot(211),plt.plot(predictions)
plt.subplot(212),plt.plot((ang_test-predictions)/ang_test)
plt.show()

"""
print('training model...')
model.fit(us_train, ang_train, validation_split=0.3, epochs=10, batch_size=8, verbose=2)

model.save('models/four_fingers/pinch_relax_convolutional.h5')
"""
