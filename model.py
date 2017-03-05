# The MIT License (MIT)
#
# Copyright 2017 Artem Artemev, im@artemav.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy.random as rnd

import common
import pickle
import time
import os

import importlib

import dataproc
import keras
import keras.callbacks as cb

from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Input, merge
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend
from keras.optimizers import Adam
from keras.regularizers import l2

importlib.reload(common)
importlib.reload(dataproc)

class BehaviorModel:
    @classmethod
    def densenet_model(cls,
            input_shape=(128,128,3),
            act='elu',
            decay=0.0001,
            learning_rate=0.001,
            show=True):
        xinput = Input(shape=input_shape)
        layers = 3
        filters = 16
        rate = 12
        x = Convolution2D(filters, 3, 3,
                          init='he_uniform',
                          border_mode='same',
                          bias=False,
                          W_regularizer=l2(decay))(xinput)
        for block_idx in range(2):
            x, filters = cls._denseblock(x, layers, filters, growth_rate=rate, decay=decay)
            x = cls._transition(x, filters, decay=decay)
        x, filters = cls._denseblock(x, layers, filters, growth_rate=rate, decay=decay)
        x = BatchNormalization(mode=0, axis=1,
                               gamma_regularizer=l2(decay),
                               beta_regularizer=l2(decay))(x)
        x = Activation(act)(x)
        x = GlobalAveragePooling2D(dim_ordering="tf")(x)
        x = Dense(500, activation=act, W_regularizer=l2(decay), b_regularizer=l2(decay))(x)
        x = Dense(100, activation=act, W_regularizer=l2(decay), b_regularizer=l2(decay))(x)
        x = Dense(1, activation='linear', W_regularizer=l2(decay), b_regularizer=l2(decay))(x)
        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1E-08)
        m = Model(input=[xinput], output=[x], name="densenet")
        m.compile(optimizer=adam, loss='mean_squared_error')
        if show == True:
            print(m.summary())
        return m
    @classmethod
    def model(cls, input_shape, act='elu', decay=0.0002, learning_rate=0.001, show=True):
        m = Sequential()
        pool = (2,2)
        stride = (1,1)
        mode = 'valid'
        m.add(Convolution2D(8, 5, 5,
             activation=act, input_shape=input_shape, border_mode=mode, subsample=stride))
        m.add(MaxPooling2D(pool_size=pool))
        m.add(Convolution2D(8, 5, 5,
             activation=act, border_mode=mode, subsample=stride))
        m.add(Activation(act))
        m.add(MaxPooling2D(pool_size=pool))
        m.add(Convolution2D(16, 4, 4,
             activation=act, border_mode=mode, subsample=stride))
        m.add(MaxPooling2D(pool_size=pool))
        m.add(Convolution2D(16, 5, 5,
             activation=act, border_mode=mode, subsample=stride))
        m.add(Flatten())
        m.add(Dense(128, W_regularizer=l2(decay), activation=act))
        m.add(Dense(50, W_regularizer=l2(decay), activation=act))
        m.add(Dense(10, W_regularizer=l2(decay), activation=act))
        m.add(Dense(1, activation='linear', init='he_normal'))
        m.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error')
        if show == True:
            print(m.summary())
        return m
    @classmethod
    def train(cls, model, data,
              batch_size=128,
              aug_factor=200,
              num_epochs=20,
              best_model_file='model.h5'):
        best_cb = cb.ModelCheckpoint(best_model_file,
                                     monitor='val_loss',
                                     save_best_only=True,
                                     mode='min',
                                     verbose=1)
        stop_cb = cb.EarlyStopping(monitor='val_loss',
                                   patience=15,
                                   min_delta=0,
                                   mode='auto',
                                   verbose=0)
        samples_per_epoch = batch_size * aug_factor
        valid_count = data.valid_sample_count()
        valid_count -= valid_count % batch_size
        data.init_batch_generator(batch_size=batch_size)
        train_gen = data.batch_generator(mode='train')
        valid_gen = data.batch_generator(mode='valid')
        logs = model.fit_generator(train_gen,
                                   samples_per_epoch,
                                   validation_data=valid_gen,
                                   nb_val_samples=valid_count,
                                   nb_epoch=num_epochs,
                                   callbacks=[best_cb, stop_cb])
        backend.clear_session()
    @classmethod
    def save_model(cls, model, filename='car_model.h5', as_json=True):
        if as_json == True:
            with open(filename + '.json', 'w') as fd:
                 fd.write(model.to_json())
        model.save(filename)
    @classmethod
    def _conv(cls, x, filters, act='elu', decay=0.0001):
        x = BatchNormalization(
                mode=0,
                axis=1,
                gamma_regularizer=l2(decay),
                beta_regularizer=l2(decay))(x)
        x = Activation(act)(x)
        x = Convolution2D(
                filters, 3, 3,
                init='he_uniform',
                border_mode='same',
                bias=False,
                W_regularizer=l2(decay))(x)
        return x
    @classmethod
    def _transition(cls, x, filters, act='elu', decay=0.0001):
        x = BatchNormalization(
                mode=0,
                axis=1,
                gamma_regularizer=l2(decay),
                beta_regularizer=l2(decay))(x)
        x = Activation(act)(x)
        x = Convolution2D(
                filters, 1, 1,
                init='he_uniform',
                border_mode='same',
                bias=False,
                W_regularizer=l2(decay))(x)
        x = AveragePooling2D((2,2), strides=(2,2))(x)
        return x
    @classmethod
    def _denseblock(cls, x, layers, filters, growth_rate, act='elu', decay=0.0001):
        list_feat = [x]
        for i in range(layers):
            x = cls._conv(x, growth_rate, act=act, decay=decay)
            list_feat.append(x)
            x = merge(list_feat, mode='concat', concat_axis=-1)
            filters += growth_rate
        return x, filters
