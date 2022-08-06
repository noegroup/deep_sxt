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
from .global_params import params_Conv2D, params_batch_norm, channel_axis
from .encoder_block import MyEncoderBlock
from .midbranch_block import MyMidBranchBlock
from .decoder_block import MyDecoderBlock
from .bottleneck_conv import MyBottleneckLayer


class MyDoubleDecoderNet(tf.keras.Model):

    def __init__(self, config=None, **kwargs):

        super(MyDoubleDecoderNet, self).__init__(**kwargs)

        tf.keras.backend.set_image_data_format(params_Conv2D["data_format"])

        if config is None:
            config = {'pooling': None,
                      'add_instead_of_concat_after_neck': False,
                      'encoder': None, 'neck': None, 'decoder_image': None, 'decoder_label': None}

        pooling = config['pooling']

        if pooling is None:
            pooling = {'type': 'max_pooling', 'factor': 2}

        self.add_instead_of_concat_after_neck = config['add_instead_of_concat_after_neck']

        m_f = config['m_f']

        if m_f is None:
            m_f = 25

        encoder_config = config['encoder']

        if encoder_config is None:
            encoder_config = {'kernel_sizes': [19, 15, 11],
                              'n_stages': 2, 'n_blocks': 3, 'block_depth': 2,
                              'add_instead_of_concat':False,
                              'n_filters': [m_f, 2 * m_f, 4 * m_f],
                              'activation': {'type': 'leaky_relu', 'leak_slope': 0.1},
                              'dropout': {'active': False},
                              'batch_norm': {'active': True}}

        neck_config = config['neck']

        if neck_config is None:
            neck_config = {'kernel_sizes': [1, 3, 5, 7, 9],
                           'depth': 4, 'n_filters': 8 * m_f,
                           'activation': {'type': 'relu'},
                           'dropout': {'active': False},
                           'batch_norm': {'active': True}}

        decoder_image_config = config['decoder_image']

        if decoder_image_config is None:
            decoder_image_config = {'kernel_sizes': [11, 15, 19],
                                    'n_stages': 2, 'n_blocks': 3, 'block_depth': 2,
                                    'add_instead_of_concat': False,
                                    'use_skip_connections': False,
                                    'n_filters': [4 * m_f, 2 * m_f, m_f],
                                    'activation': {'type': 'relu'},
                                    'dropout': {'active': False},
                                    'batch_norm': {'active': True}}

        print(f"skip connections in image decoder: {decoder_image_config['use_skip_connections']}")

        decoder_label_config = config['decoder_label']

        if decoder_label_config is None:
            decoder_label_config = {'kernel_sizes': [11, 15, 19],
                                    'n_stages': 2, 'n_blocks': 3, 'block_depth': 2,
                                    'add_instead_of_concat': False,
                                    'use_skip_connections': True,
                                    'n_filters': [4 * m_f, 2 * m_f, m_f],
                                    'activation': {'type': 'relu'},
                                    'dropout': {'active': False},
                                    'batch_norm': {'active': True}}

        print(f"skip connections in label decoder: {decoder_label_config['use_skip_connections']}")

        self.config = config

        self.encoder = MyEncoderBlock(**encoder_config, pooling=pooling, name="encoder")

        self.mid_branch = []

        neck_kernel_sizes = neck_config.pop('kernel_sizes')

        for i, _kernel_size in enumerate(neck_kernel_sizes):
            self.mid_branch.append(MyMidBranchBlock(**neck_config, kernel_size=_kernel_size,
                                                    name="mid_branch_" + str(i)))

        # Adding a decoder for image reproduction
        self.decoder_image = MyDecoderBlock(**decoder_image_config, pooling=pooling, name="image_decoder")

        _final_kernel_size = decoder_image_config["kernel_sizes"][-1]

        self.final_layers_image = []

        self.final_layers_image.append(MyBottleneckLayer(filters=m_f, kernel_size=_final_kernel_size,
                                                       activation=None, use_bias=True,
                                                       name="image_decoder_final_convolution_0"))

        self.final_layers_image.append(MyBottleneckLayer(filters=m_f, kernel_size=_final_kernel_size,
                                                       activation=None, use_bias=True,
                                                       name="image_decoder_final_convolution_1"))

        self.final_layers_image.append(MyBottleneckLayer(filters=1, kernel_size=_final_kernel_size,
                                                       activation=None, use_bias=True,
                                                       name="image_decoder_final_convolution_2"))

        # Adding a decoder for segmentation
        self.decoder_label = MyDecoderBlock(**decoder_label_config, pooling=pooling, name="label_decoder")

        _final_kernel_size = decoder_label_config["kernel_sizes"][-1]

        self.final_layers_label = []

        self.final_layers_label.append(lay.BatchNormalization(**params_batch_norm,
                                                       name="label_decoder_final_batchnorm"))

        self.final_layers_label.append(MyBottleneckLayer(filters=m_f, kernel_size=_final_kernel_size,
                                                       activation=None, use_bias=True,
                                                       name="label_decoder_final_convolution_1"))

        self.final_layers_label.append(lay.Conv2D(filters=1, kernel_size=1,
                                                activation='tanh', use_bias=True,
                                                **params_Conv2D,
                                                name="label_decoder_final_convolution_2"))

    def call(self, x, **kwargs):

        x, skip_x = self.encoder(x, **kwargs)

        y = self.mid_branch[0](x, **kwargs)

        for _branch in self.mid_branch[1:]:
            if self.add_instead_of_concat_after_neck:
                y = tf.add(y, _branch(x, **kwargs))
            else:
                y = tf.concat([y, _branch(x, **kwargs)], axis=channel_axis)

        x_image = self.decoder_image(y, skip_x=skip_x, **kwargs)

        for _lay_img in self.final_layers_image:
            x_image = _lay_img(x_image, **kwargs)

        x_label = self.decoder_label(y, skip_x=skip_x, **kwargs)

        for _lay_lbl in self.final_layers_label:
            x_label = _lay_lbl(x_label, **kwargs)

        return [x_image, x_label]
