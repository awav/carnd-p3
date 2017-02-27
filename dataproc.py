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
import matplotlib.pyplot as plt
import numpy.random as rnd

import common
import pickle
import time
import os

import importlib

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
    def __init__(self, data_path='data/', driving_log='driving_log.csv', img_path='IMG/', show=True):
        imgs_path = os.path.join(data_path, img_path)
        csv_path = os.path.join(data_path, driving_log)
        dt = pd.read_csv(csv_path)
        right_steering = self._adjust_angles(dt.steering, side='right')
        left_steering = self._adjust_angles(dt.steering, side='left')
        center = pd.DataFrame({'image':dt.center, 'steering':df.steering})
        right = pd.DataFrame({'image':dt.right, 'steering':right_steering})
        left = pd.DataFrame({'image':dt.left, 'steering':left_steering})
        self._data = pd.concat([center, left, right], ignore_index=True)
    def _adjust_angles(angles, side='right', adjust=0.25):
        if side == 'right':
            adjust *= -1
        elif:
            raise ValueError('Unknown adjustment side')
        return np.arctan(np.tan(angles) + adjust)
    def load(self, filename='data/dataset.p'):
        with open(filename, 'rb') as fd:
             self.__dict__ = pickle.load(fd)
    def save(self, filename='data/dataset.p'):
        with open(filename, 'wb') as fd:
             pickle.dump(self.__dict__, fd)
