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


import tensorflow.keras.layers as lay
from .activation_block import MyActivationBlock
from .bottleneck_conv import MyBottleneckLayer


class MyMidBranchBlock(lay.Layer):

    def __init__(self, depth,
                 kernel_size,
                 n_filters,
                 activation=None, dropout=None, batch_norm=None,
                 **kwargs):

        super(MyMidBranchBlock, self).__init__(**kwargs)

        self.conv = []
        self.activation = []

        for i in range(depth):
            self.conv.append(MyBottleneckLayer(filters=n_filters,
                                               kernel_size=kernel_size,
                                               activation=None, use_bias=False,
                                               name=self.name + "_convolution_" + str(i)))

            self.activation.append(MyActivationBlock(activation=activation,
                                                     dropout=dropout,
                                                     batch_norm=batch_norm,
                                                     name=self.name + "_activation_" + str(i)))

    def call(self, x, **kwargs):

        for _conv, _act in zip(self.conv, self.activation):
            x = _act(_conv(x, **kwargs), **kwargs)

        return x
