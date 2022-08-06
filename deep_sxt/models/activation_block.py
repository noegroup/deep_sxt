# The MIT License (MIT)

# Copyright (c) 2022, Artificial Intelligence for the Sciences Group (AI4Science),
# Freie Universität Berlin, Germany.

# All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
import tensorflow.keras.layers as lay
from .global_params import params_batch_norm, params_Conv2D


class MyActivationBlock(lay.Layer):

    def __init__(self, activation, dropout, batch_norm, **kwargs):

        super(MyActivationBlock, self).__init__(**kwargs)

        if activation['type'] == 'leaky_relu':
            self.activation = lay.LeakyReLU(alpha=activation['leak_slope'])

        elif activation['type'] == 'relu':
            self.activation = lay.Activation('relu')

        elif activation['type'] == 'softplus':
            self.activation = lay.Activation('softplus')

        else:
            self.activation = lay.Activation('linear')

        if dropout['active']:
            self.dropout = lay.SpatialDropout2D(rate=dropout['rate'],
                                                data_format=params_Conv2D["data_format"])

        else:
            self.dropout = lay.Activation('linear')

        if batch_norm['active']:
            self.batch_norm = lay.BatchNormalization(**params_batch_norm)

        else:
            self.batch_norm = lay.Activation('linear')

    def call(self, x, **kwargs):

        return self.dropout(self.activation(self.batch_norm(x, **kwargs), **kwargs), **kwargs)
