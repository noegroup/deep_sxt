# The MIT License (MIT)
#
# Copyright (c) 2022, Mohsen Sadeghi (mohsen.sadeghi@fu-berlin)
# Artificial Intelligence for the Sciences Group (AI4Science),
# Freie Universität Berlin, Germany.
#
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
import tensorflow.keras.layers as lay
import numpy as np
from .global_params import params_Conv2D

class MyBottleneckLayer(lay.Layer):

    def __init__(self, filters, kernel_size, activation=None, use_bias=None, **kwargs):

        super(MyBottleneckLayer, self).__init__(**kwargs)

#        n_filters_reduced = max(1, int(np.ceil(np.sqrt(filters))))
        n_filters_reduced = max(1, filters // 2)

        self.conv_1 = lay.Conv2D(filters=n_filters_reduced,
                                 kernel_size=1,
                                 activation=None,
                                 use_bias=use_bias,
                                 **params_Conv2D)

        self.conv_2 = lay.Conv2D(filters=n_filters_reduced,
                                 kernel_size=kernel_size,
                                 activation=None,
                                 use_bias=use_bias,
                                 **params_Conv2D)

        self.conv_3 = lay.Conv2D(filters=filters, kernel_size=1,
                                 activation=activation,
                                 use_bias=use_bias,
                                 **params_Conv2D)

    def call(self, x, **kwargs):

        return self.conv_3(self.conv_2(self.conv_1(x, **kwargs), **kwargs), **kwargs)
