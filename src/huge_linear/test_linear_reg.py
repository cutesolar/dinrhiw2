#!/usr/bin/python3

# calculates linear regresion fit to data

import sys
import pandas as pd
import numbers
import numpy as np
import math
import tensorflow as tf
from sklearn.linear_model import LinearRegression

dfx = pd.read_csv("in_x.csv",header=None)
dfy = pd.read_csv("out_y.csv",header=None)

X = dfx.values
y = dfy.values

print("X size: " + str(X.shape))
print("y size: " + str(y.shape))

xsize = X.shape[1]
ysize = y.shape[1]
N = X.shape[0]

############################################################################
# linear model error is:

print("Fitting linear model to data..");

regmodel = LinearRegression()
reg = regmodel.fit(X, y)
yy = regmodel.predict(X)

mne = np.sum(np.absolute(np.subtract(y,yy)))/len(yy) # mean norm error
mse = np.sum(np.subtract(y,yy)**2)/len(yy)
rms = math.sqrt(mse)

print("MNE error of linear model: " + str(mne))
print("MSE error of linear model: " + str(mse))
print("RMS error of linear model: " + str(rms))

#############################################################################
# Neural network error is:

loss_fn = tf.keras.losses.MeanSquaredError()

nn_model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(xsize,)),
    tf.keras.layers.Dense(xsize), tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dense(xsize), tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dense(xsize), tf.keras.layers.LeakyReLU(alpha=0.1),
    tf.keras.layers.Dense(ysize)])

nn_model.compile(optimizer='adam',
                 loss=loss_fn,
                 metrics=['mean_absolute_error'])

nn_model.fit(X, y, epochs=1000)

