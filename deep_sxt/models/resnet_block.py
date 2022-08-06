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
from .activation_block import MyActivationBlock
from .bottleneck_conv import MyBottleneckLayer
from .global_params import channel_axis


class MyResNetBlock(lay.Layer):

    def __init__(self, depth, n_filters, kernel_size, add_instead_of_concat=False,
                 activation=None, dropout=None, batch_norm=None,
                 **kwargs):

        super(MyResNetBlock, self).__init__(**kwargs)

        self.add_instead_of_concat = add_instead_of_concat

        self.conv = []
        self.activation = []

        for i in range(depth):
            self.conv.append(MyBottleneckLayer(filters=n_filters, kernel_size=kernel_size,
                                               activation=None, use_bias=False))

            self.activation.append(MyActivationBlock(activation=activation,
                                                     dropout=dropout,
                                                     batch_norm=batch_norm))

    def call(self, x, **kwargs):

        y = x

        for _conv, _act in zip(self.conv[:-1], self.activation[:-1]):
            x = _act(_conv(x, **kwargs), **kwargs)

        x = self.conv[-1](x, **kwargs)

        if self.add_instead_of_concat:
            y = tf.add(y, x)
        else:
            y = tf.concat([y, x], axis=channel_axis)

        return self.activation[-1](y, **kwargs)
