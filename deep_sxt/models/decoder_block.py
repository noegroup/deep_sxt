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
from .global_params import params_Conv2D, params_batch_norm, channel_axis
from .activation_block import MyActivationBlock
from .resnet_block import MyResNetBlock
from .bottleneck_conv import MyBottleneckLayer


class MyDecoderBlock(lay.Layer):

    # Each stage is an upsampling, potentially concat with skip connections
    # and finally followed by a series of n_blocks of resnets
    def __init__(self, n_stages, n_blocks, block_depth,
                 add_instead_of_concat,
                 kernel_sizes, n_filters, pooling,
                 use_skip_connections=False,
                 activation=None, dropout=None, batch_norm=None,
                 **kwargs):

        super(MyDecoderBlock, self).__init__(**kwargs)

        assert len(kernel_sizes) == n_stages + 1
        assert len(n_filters) == n_stages + 1

        self.use_skip_connections = use_skip_connections

        self.add_instead_of_concat = add_instead_of_concat

        self.conv_t = []
        self.batch_norm = []

        for i in range(n_stages):
            self.conv_t.append(lay.Conv2DTranspose(filters=n_filters[i],
                                                   kernel_size=kernel_sizes[i],
                                                   strides=pooling['factor'],
                                                   activation=None, use_bias=True,
                                                   name=self.name + "_transposed_convolution_" + str(i),
                                                   **params_Conv2D))

        self.resnet_blocks = []

        for i in range(n_stages):

            _resnet_block = []

            for j in range(n_blocks):

                _resnet_block.append(MyResNetBlock(depth=block_depth,
                                                   n_filters=n_filters[i],
                                                   add_instead_of_concat=self.add_instead_of_concat,
                                                   kernel_size=kernel_sizes[i],
                                                   activation=activation,
                                                   dropout=dropout,
                                                   batch_norm=batch_norm,
                                                   name=self.name + "_resnet_block_" + str(i) + "_" + str(j)))

            self.resnet_blocks.append(_resnet_block)

        self.final_conv = MyBottleneckLayer(filters=n_filters[-1],
                                            kernel_size=kernel_sizes[-1],
                                            activation=None, use_bias=False,
                                            name=self.name + "_final_convolution")

        self.final_activation = MyActivationBlock(activation=activation,
                                                  dropout=dropout, batch_norm=batch_norm,
                                                  name=self.name + "_final_activation")

    def call(self, x, skip_x, **kwargs):

        for _conv_t, _resnet_blocks, _skip in zip(self.conv_t,
                                                  self.resnet_blocks,
                                                  skip_x):

            x = _conv_t(x, **kwargs)

            if self.use_skip_connections:
                if self.add_instead_of_concat:
                    x = tf.add(_skip, x)
                else:
                    x = tf.concat([_skip, x], axis=channel_axis)

            for _resnet in _resnet_blocks:
                x = _resnet(x, **kwargs)

        return self.final_activation(self.final_conv(x, **kwargs), **kwargs)

