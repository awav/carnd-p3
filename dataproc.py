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

importlib.reload(common)

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
    #def brightness(im, lbl, low=0.25, high=1.15, prob=[0.5, 0.5]):
    def brightness(im, lbl, low=0.25, high=1.2, prob=[0.5, 0.5]):
        if rnd.choice([1,0], p=prob) == 1:
            hsv = np.float32(common.cvt_color(np.uint8(im), color='HSV', src='RGB'))
            adjust = rnd.uniform(low=low, high=high)
            hsv[:,:,2] *= adjust
            hsv = np.clip(hsv, 0, 255)
            return cv.cvtColor(np.uint8(hsv), cv.COLOR_HSV2RGB), lbl
        else:
            return im, lbl
    @staticmethod
    def flip_horizontal(im, lbl, prob=[0.5, 0.5]):
        if rnd.choice([1,0], p=prob) == 1:
            return np.fliplr(im), -lbl
        else:
            return im, lbl
    @staticmethod
    def shadow(im, lbl, low=0.25, high=0.7, prob=[0.5, 0.5]):
        if rnd.choice([1,0], p=prob) == 1:
            im = np.float32(im)
            height, width, _ = im.shape
            adjust = rnd.uniform(low=low, high=high)
            xu, xd = rnd.uniform(low=0, high=width, size=2)
            alpha = height / (xd - xu)
            beta = - (alpha * xu)
            side = rnd.choice([0, 1])
            for row in range(height):
                col = np.int32((row - beta) / alpha)
                if side == 1:
                    im[row,:col] *= adjust
                else:
                    im[row,col:] *= adjust
            return np.uint8(im), lbl
        else:
            return im, lbl
    @staticmethod
    def crop_height(im, top=(0.325, 0.475), bottom=(0.075, 0.175)):
        yt = int(rnd.uniform(low=top[0], high=top[1]) * im.shape[0])
        yb = int(rnd.uniform(low=bottom[0], high=bottom[1]) * im.shape[0])
        return im[yt:-yb]

class TrackDataset():
    def __init__(self, data_path='data/', driving_log='driving_log.csv', im_path='IMG/'):
        self._data_path = data_path
        self._images_path = os.path.join(data_path, im_path)
        csv_path = os.path.join(data_path, driving_log)
        self._load_steering_data(csv_path)
        self._split_skewed()
        self._split_valid_train()
        self._batch_x_shape = None
        self._batch_y_shape = None
        self._mode = None
        self._color = None
    def valid_sample_count(self):
        return self._valid.shape[0]
    def train_sample_count(self):
        return self._train.shape[0] + self._skewed_size
    def init_batch_generator(self, batch_size=128, image_size=(128,128,3), color='RGB'):
        self._batch_x_shape = (batch_size, *image_size)
        self._batch_y_shape = (batch_size, 1)
        self._color = color
    def batch_generator(self, mode='train'):
        if mode != 'train' and mode != 'valid' and mode != 'example':
            raise ValueError('Unknown mode {0}'.format(mode))
        skewed_size = np.uint32(self._train.steering.value_counts().max() *
                                self._skewed_count *
                                0.85)
        train_size = skewed_size + self._train.shape[0]
        batch_shape = self._batch_x_shape
        batch_size = self._batch_x_shape[0]
        im_shape = self._batch_x_shape[1:3]
        im_std = 1.0 / np.sqrt(batch_shape[1] * batch_shape[2] * batch_shape[3])
        while True:
            x = np.zeros(self._batch_x_shape)
            y = np.zeros(self._batch_y_shape)
            if mode == 'train' or mode == 'example':
                skewed_samples = self._sample_from_skewed(skewed_size)
                data = self._train.append(skewed_samples)
                data.reset_index(inplace=True, drop=True)
            else:
                data = self._valid
            batch = self._sample_batch(data, batch_size)
            for i in batch.index:
                filename = batch.loc[i, 'image']
                steering = batch.loc[i, 'steering']
                im_path = os.path.join(self._data_path, filename.strip())
                if not os.path.exists(im_path):
                    raise ValueError('Image not found {0}'.format(im_path))
                im = common.load_image(im_path, color=self._color)
                im, y_ = self._augment(im, steering)
                im = cv.resize(im, im_shape)
                if mode == 'example':
                    yield im, y_
                else:
                    x[i] = self.normalize_image(im.astype(np.float32), im_std=im_std)
                    y[i] = y_
            yield x, y
    @classmethod
    def normalize_image(cls, im, im_std=None):
        if im_std == None:
            im_std = 1.0 / np.sqrt(im.shape[0] * im.shape[1] * im.shape[2])
        xmean = np.mean(im)
        xstd = max(np.std(im, ddof=1), im_std)
        return (im - xmean) / xstd
    @classmethod
    def preprocess_image(cls, im, image_shape=(128,128)):
        im = Augment.crop_height(im, top=(0.45,0.45), bottom=(0.15,0.15))
        im = cv.resize(im, image_shape).astype(np.float32)
        return cls.normalize_image(im)
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
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        fig.canvas.set_window_title('Valid dataset distribution')
        ax1.set_title('normal scale')
        ax2.set_title('log scaled')
        ax2.set_yscale('log')
        self._valid.steering.hist(ax=ax1, alpha=0.5, bins=100)
        self._valid.steering.hist(ax=ax2, alpha=0.5, bins=100)
        fig.show()
    def _augment(self, im, steering):
        im = Augment.crop_height(im)
        im, steering = Augment.flip_horizontal(im, steering)
        im, steering = Augment.brightness(im, steering)
        im, steering = Augment.shadow(im, steering)
        return im, steering
    def _load_steering_data(self, csv_path):
        dt = pd.read_csv(csv_path)
        right_steering = self._adjust_angles(dt.steering, side='right')
        left_steering = self._adjust_angles(dt.steering, side='left')
        center = pd.DataFrame({'image':dt.center, 'steering':dt.steering})
        right = pd.DataFrame({'image':dt.right, 'steering':right_steering})
        left = pd.DataFrame({'image':dt.left, 'steering':left_steering})
        self._dt = dt
        self._data = pd.concat([center, left, right], ignore_index=True)
    def _split_valid_train(self, nonskewed_test_size=0.15):
        x, xval, y, yval = train_test_split(
             self._nonskewed_data.image,
             self._nonskewed_data.steering,
             test_size=nonskewed_test_size)
        _, counts  = np.unique(yval, return_counts=True)
        sample_size = np.uint32(self._skewed_count * counts.max() * 0.85)
        skewed_samples = self._sample_from_skewed(sample_size, drop=True)
        xval = xval.append(skewed_samples.image)
        yval = yval.append(skewed_samples.steering)
        self._train = pd.DataFrame({'image':x, 'steering':y})
        self._valid = pd.DataFrame({'image':xval, 'steering':yval})
        self._train.reset_index(inplace=True, drop=True)
        self._valid.reset_index(inplace=True, drop=True)
        self._skewed_size = np.uint32(
            self._train.steering.value_counts().max() *
            self._skewed_count *
            0.85)
    def _split_skewed(self, skewmax=1000):
        vals = self._data.steering.value_counts()
        ids = vals.index[vals.values >= skewmax]
        self._skewed_count = ids.shape[0]
        skewed_rows = np.where(self._data.steering.isin(ids) == True)[0]
        nonskewed_rows = np.where(self._data.steering.isin(ids) == False)[0]
        self._skewed_data = self._data.loc[skewed_rows]
        self._nonskewed_data = self._data.loc[nonskewed_rows]
        self._skewed_data.reset_index(inplace=True, drop=True)
        self._nonskewed_data.reset_index(inplace=True, drop=True)
    def _sample_from_skewed(self, size, drop=False):
        rows = rnd.choice(range(self._skewed_data.shape[0]), size=size, replace=False)
        sliced_data = self._skewed_data.loc[rows]
        if drop == True:
            self._skewed_data.drop(rows, inplace=True)
            self._skewed_data.reset_index(inplace=True, drop=True)
        sliced_data.reset_index(inplace=True, drop=True)
        return sliced_data
    def _sample_batch(self, train, size):
        rows = rnd.choice(range(train.shape[0]), size=size, replace=False)
        batch = train.loc[rows]
        batch.reset_index(inplace=True, drop=True)
        return batch
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
    def save_data(self, filename, data):
        with open(filename, 'wb') as fd:
             pickle.dump(data, fd)
