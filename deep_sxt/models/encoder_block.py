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
from .resnet_block import MyResNetBlock
from .bottleneck_conv import MyBottleneckLayer
from .global_params import params_Conv2D


class MyEncoderBlock(lay.Layer):

    # Each stage is a series of n_blocks of resnets followed by pooling
    def __init__(self, n_stages, n_blocks, block_depth,
                 add_instead_of_concat,
                 kernel_sizes, n_filters, pooling,
                 activation=None, dropout=None, batch_norm=None,
                 **kwargs):

        super(MyEncoderBlock, self).__init__(**kwargs)

        assert len(kernel_sizes) == n_stages + 1
        assert len(n_filters) == n_stages + 1

        pool_size = (pooling['factor'], pooling['factor'])

        self.initial_conv = MyBottleneckLayer(filters=n_filters[0],
                                              kernel_size=kernel_sizes[0],
                                              activation=None, use_bias=True,
                                              name=self.name + "_initial_convolution")

        self.initial_activation = MyActivationBlock(activation=activation,
                                                    dropout=dropout, batch_norm=batch_norm,
                                                    name=self.name + "_initial_activation")

        self.resnet_blocks = []

        for i in range(n_stages):

            _resnet_block = []

            for j in range(n_blocks):
                _resnet_block.append(MyResNetBlock(depth=block_depth,
                                                   n_filters=n_filters[i + 1],
                                                   add_instead_of_concat=add_instead_of_concat,
                                                   kernel_size=kernel_sizes[i + 1],
                                                   activation=activation,
                                                   dropout=dropout, batch_norm=batch_norm,
                                                   name=self.name + "_resnet_block_" + str(i) + "_" + str(j)))

            self.resnet_blocks.append(_resnet_block)

        self.pool = []

        if pooling['type'] == 'max_pooling':
            for i in range(n_stages):
                self.pool.append(lay.MaxPooling2D(pool_size=pool_size, strides=None,
                                                  padding='same',
                                                  data_format=params_Conv2D["data_format"],
                                                  name=self.name + "_max_pooling_" + str(i)))

        elif pooling['type'] == 'average_pooling':
            for i in range(n_stages):
                self.pool.append(lay.AveragePooling2D(pool_size=pool_size, strides=None,
                                                      padding='same',
                                                      data_format=params_Conv2D["data_format"],
                                                      name=self.name + "_average_pooling_" + str(i)))
        else:
            raise ValueError("Unknown pooling type!")

    def call(self, x, **kwargs):

        x = self.initial_activation(self.initial_conv(x, **kwargs), **kwargs)

        skip_x = []

        for _resnet_blocks, _pool in zip(self.resnet_blocks, self.pool):

            for _resnet in _resnet_blocks:
                x = _resnet(x, **kwargs)

            skip_x.append(x)

            x = _pool(x)

        skip_x.reverse()

        return [x, skip_x]
