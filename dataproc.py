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

from sklearn.model_selection import train_test_split

class Augment():
    def __new__(cls, *args, **kvargs):
        raise ValueError('Not possible to create instance of this class')
    @staticmethod
    def setseed(seed=None):
        if seed == None:
           a = rnd.uniform(0,1)
           b = np.uint32(rnd.uniform(1, 1000))
           rnd.seed(np.uint32(time.time() * a) + b)
        else:
           rnd.seed(seed)
    @staticmethod
    def brightness(im, lbl, low=0.3, high=1.2, p=0.5, np=0.5):
        if rnd.choice([1,0], p=[p, np]) == 1:
            hsv = common.cvt_color(img, color='HSV', src='RGB')
            adjust = rnd.uniform(low=low, high=high)
            hsv[:,:,2] *= adjust
            return (cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB), lbl)
        return im, lbl
    @staticmethod
    def flip_horizontal(im, lbl, p=0.5, np=0.5):
        assert((p + np) == 1)
        if rnd.choice([1,0], p=[p, np]) == 1:
            return (np.fliplr(im), -lbl)
        return (im, lbl)
    @staticmethod
    def shadow(im, lbl, low=0.4, hight=0.9, p=0.5, np=0.5):
        if rnd.choice([1,0], p=[p, np]) == 1:
            height, width, _ = im.shape
            adjust = rnd.uniform(low=0, high=width)
            xu, xd = rnd.uniform(low=0, high=width, size=2)
            alpha = height / (xu - xd)
            beta = - (k * xu)
            side = rnd.uniform([0, 1])
            for row in height:
                col = np.uint32((col - beta) / alpha)
                if side == 0:
                    im[row,:col] *= adjust
                else:
                    im[row,col:] *= adjust
            return im.astype(np.uint8), lbl
        return im, lbl
    @staticmethod
    def crop_height(im, lbl, top=(0.3, 0.450), bottom=(0.075, 0.175), p=1, np=0):
        yt = np.uint32(rnd.uniform(*top))
        yb = np.uint32(rnd.uniform(*bottom))
        return im[yt:-yb], lbl

class TrackDataset():
    def __init__(self, data_path='data/', driving_log='driving_log.csv', img_path='IMG/'):
        self._images_path = os.path.join(data_path, img_path)
        csv_path = os.path.join(data_path, driving_log)
        self._load_steering_data(csv_path)
        self._split_valid_train()
        self._skewed_angles = None
        self._batch_x_shape = None
        self._batch_x_shape = None 
        self._batch_y_shape = None 
        self._mode = None
        self._color = None
    def init_batch_generator(self, batch_size=128, image_size=(128,128), mode='train', color='RGB'):
        self._batch_x_shape = (batch_size, *image_size)
        self._batch_x_shape = (batch_size, *image_size)
        self._batch_y_shape = (batch_size, 1)
        if mode != 'train' and mode != 'valid':
            raise ValueError('Unknown mode {0}'.format(mode))
        self._mode = mode
        self._color = color
    def batch_generator():
        batch_size = self._batch_x_shape[0]
        x = np.zeros(self._batch_x_shape)
        y = np.zeros(self._batch_y_shape)
        while self:
            #for i in range(batch_size):
            yield x, y
    def show_histograms(self):
        matplotlib.rc('xtick', labelsize=15)
        matplotlib.rc('ytick', labelsize=15)
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        fig.canvas.set_window_title('Initial dataset')
        ax1.set_title('normal scale')
        ax2.set_title('log scaled')
        ax2.set_yscale('log')
        self._dt.steering.hist(ax=ax1, alpha=0.5, bins=100)
        self._dt.steering.hist(ax=ax2, alpha=0.5, bins=100)
        fig.show()
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        fig.canvas.set_window_title('Augmented by steering angles dataset')
        ax1.set_title('normal scale')
        ax2.set_title('log scaled')
        ax2.set_yscale('log')
        self._data.steering.hist(ax=ax1, alpha=0.5, bins=100)
        self._data.steering.hist(ax=ax2, alpha=0.5, bins=100)
        fig.show()
    def _load_steering_data(self, csv_path):
        dt = pd.read_csv(csv_path)
        right_steering = self._adjust_angles(dt.steering, side='right')
        left_steering = self._adjust_angles(dt.steering, side='left')
        center = pd.DataFrame({'image':dt.center, 'steering':dt.steering})
        right = pd.DataFrame({'image':dt.right, 'steering':right_steering})
        left = pd.DataFrame({'image':dt.left, 'steering':left_steering})
        self._dt = dt
        self._data = pd.concat([center, left, right], ignore_index=True)
    def _split_valid_train(self, skewmax=1000):
        vals = self._data.steering.value_counts()
        self._skewed_angles = vals.index[vals.values >= skewmax]
        #nonskewed_data = 
        #self._xnorm, self._xval, self._ynorm, self._yval = train_test_split(test=0.15)
    def _adjust_angles(self, angles, side='right', adjust=0.25):
        if side == 'right':
            adjust *= -1
        elif side != 'left':
            raise ValueError('Unknown adjustment side')
        vals = np.arctan(np.tan(angles) + adjust)
        return vals.clip(lower=-1.0, upper=1.0)
    def load(self, filename='data/dataset.p'):
        with open(filename, 'rb') as fd:
             self.__dict__ = pickle.load(fd)
    def save(self, filename='data/dataset.p'):
        with open(filename, 'wb') as fd:
             pickle.dump(self.__dict__, fd)
