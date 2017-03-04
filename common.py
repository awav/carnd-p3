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

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import re

def load_image(filename, color='RGB'):
    im = cv.imread(filename, cv.IMREAD_COLOR)
    return cvt_color(im, color=color, src='BGR')

def cvt_color(im, color='RGB', src='BGR'):
    if src == 'BGR':
        colors = {'RGB':cv.COLOR_BGR2RGB,
                  'HLS':cv.COLOR_BGR2HLS,
                  'HSV':cv.COLOR_BGR2HSV,
                  'LUV':cv.COLOR_BGR2LUV,
                  'YUV':cv.COLOR_BGR2YUV,
                  'YCrCb':cv.COLOR_BGR2YCrCb}
    elif src == 'RGB':
        colors = {'BGR':cv.COLOR_RGB2BGR,
                  'HLS':cv.COLOR_RGB2HLS,
                  'HSV':cv.COLOR_RGB2HSV,
                  'LUV':cv.COLOR_RGB2LUV,
                  'YUV':cv.COLOR_RGB2YUV,
                  'YCrCb':cv.COLOR_RGB2YCrCb}
    else:
        raise ValueError("Source image can be either in RGB or BGR colormap")
    if color not in colors:
        raise ValueError("{0} color is not allowed".format(color))
    return cv.cvtColor(im, colors[color]).astype(np.uint8)

def list_images(folder, regex=None):
    return np.array([filename for filename in _list_images(folder)])

def _list_images(folder, regex=None):
    """
    Generator to list images in directory. Default regex parameter is None and
    it will search for png and jpg images.
    """
    if not os.path.isdir(folder):
        ## TOOD: log this message
        return
    folder = os.path.normpath(folder)
    folder_len = len(folder)
    default_pattern = '.*\.(png|jpg|jpeg|PNG|JPG|JPEG)$'
    regex = default_pattern if regex == None else regex
    regex_compiled = re.compile(regex)
    for subdir, mid, files in os.walk(folder):
        subdir = os.path.normpath(subdir)
        if subdir[:folder_len] != folder:
            continue
        if  files == []:
            continue
        for filename in files:
            if re.match(regex_compiled, filename) is None:
                continue
            yield os.path.join(subdir, filename)

def load_images(*dirs, color='RGB'):
    return np.array([im for im in _load_images(*dirs, color=color)])

def _load_images(*dirs, color='RGB'):
    """
    Generator to list images in specified dirs.
    By default it loads images in RGB.
    """
    subdirs = set()
    for folder in dirs:
        for filename in _list_images(folder):
            subdirname = os.path.dirname(filename)
            if subdirname not in subdirs:
                subdirs.add(subdirname)
                print(subdirname)
            yield load_image(filename, color=color)

def serialize(obj, filename):
    with open(filename, 'wb') as fd:
        pickle.dump(obj, fd)

def show_image(ims, ncols=1, nrows=1, window_title=None, titles=None, cmaps=None):
    """
    Show images as tiles in grid.
    """
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, squeeze=True)
    if window_title is not None:
        fig.canvas.set_window_title(window_title)
    size = len(ims)
    if ncols == 1 and nrows == 1:
        if cmaps is not None:
            axes.imshow(ims, cmap=cmaps)
        else:
            axes.imshow(ims)
        if titles is not None:
            axes.set_title(titles)
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
    elif ncols == 1 or nrows == 1:
        n = max(ncols, nrows)
        for i in range(n):
            ax = axes[i]
            if cmaps is not None:
                ax.imshow(ims[i], cmap=cmaps[i])
            else:
                ax.imshow(ims[i])
            if titles is not None:
                ax.set_title(titles[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    else:
        for r in range(nrows):
            for c in range(ncols):
                ax = axes[r][c]
                i = r * ncols + c
                if i >= size:
                    ax.axis('off')
                    continue
                if cmaps is not None:
                    ax.imshow(ims[i], cmap=cmaps[i])
                else:
                    ax.imshow(ims[i])
                if titles is not None:
                    ax.set_title(titles[i])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    fig.show()

def show_grid(ims, rows, cols):
    plt.figure(figsize = (rows, cols))
    gs = gridspec.GridSpec(rows, cols)
    gs.update(wspace=0.025, hspace=0.025)
    for i in range(rows * cols):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        ax.imshow(ims[i])
        #plt.subp
    plt.show()

def equalize_hist(im, show=False):
    im_eq = cv.cvtColor(im, cv.COLOR_RGB2YCrCb)
    zeros = np.zeros(im_eq.shape[:2])
    im_eq[:,:,0] = cv.equalizeHist(im_eq[:,:,0], zeros)
    im_eq = cv.cvtColor(im_eq, cv.COLOR_YCrCb2RGB)
    if show == True:
        show_images(im, im_eq, 'original', 'equalized', 'Histogram equalization')
    return im_eq

## In [5]: data = model.VehiclesDataset(); m = model.CarModel(); data = m.prepare(data); m.fit(data)
## data/vehicles/GTI_MiddleClose
## data/vehicles/GTI_Left
## data/vehicles/KITTI_extracted
## data/vehicles/GTI_Far
## data/vehicles/GTI_Right
## data/OwnCollection/vehicles/Right
## data/OwnCollection/vehicles/Left
## data/OwnCollection/vehicles/Far
## data/OwnCollection/vehicles/MiddleClose
## data/non-vehicles/GTI
## data/non-vehicles/Extras
## data/OwnCollection/non-vehicles/Right
## data/OwnCollection/non-vehicles/Left
## data/OwnCollection/non-vehicles/Far
## data/OwnCollection/non-vehicles/MiddleClose
## Test accuracy: 0.99821
## AUC score: 0.99997
