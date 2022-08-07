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


def augmentation_function(max_image_size, input_image_size,
                          random_translation=True,
                          random_rotation=True,
                          random_reflection=True,
                          random_contrast=False):

    i_max_image_size = tf.cast(max_image_size, dtype=tf.dtypes.int32)
    i_input_image_size = tf.cast(input_image_size, dtype=tf.dtypes.int32)
    b_random_translation = tf.cast(random_translation, dtype=tf.dtypes.bool)
    b_random_rotation = tf.cast(random_rotation, dtype=tf.dtypes.bool)
    b_random_reflection = tf.cast(random_reflection, dtype=tf.dtypes.bool)
    b_random_contrast = tf.cast(random_contrast, dtype=tf.dtypes.bool)

    @tf.function
    def random_augment_pipeline(image, label, orig_size):

        tf_max_image_size = tf.ones(2, dtype=tf.dtypes.float32) * \
                            tf.cast(i_max_image_size, dtype=tf.dtypes.float32)

        tf_input_image_size = tf.ones(2, dtype=tf.dtypes.float32) * \
                              tf.cast(i_input_image_size, dtype=tf.dtypes.float32)

        tf_orig_size = tf.cast(orig_size, dtype=tf.dtypes.float32)

        corner = 0.5 * (tf_max_image_size - tf_orig_size)
        leg_room = tf_orig_size - tf_input_image_size

        if b_random_translation:
            offset_height = tf.cast(tf.math.round(
                corner[0] +
                tf.random.uniform(shape=[], minval=0.0, maxval=leg_room[0], dtype=tf.dtypes.float32)),
                dtype=tf.dtypes.int32)

            offset_width = tf.cast(tf.math.round(
                corner[1] +
                tf.random.uniform(shape=[], minval=0.0, maxval=leg_room[1], dtype=tf.dtypes.float32)),
                dtype=tf.dtypes.int32)
        else:
            offset_height = tf.cast(tf.math.round(corner[0]), dtype=tf.dtypes.int32)
            offset_width = tf.cast(tf.math.round(corner[1]), dtype=tf.dtypes.int32)

        image = tf.image.crop_to_bounding_box(image,
                                              offset_height, offset_width,
                                              i_input_image_size, i_input_image_size)

        label = tf.image.crop_to_bounding_box(label,
                                              offset_height, offset_width,
                                              i_input_image_size, i_input_image_size)

        if b_random_rotation:
            n_rot = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.dtypes.int32)

            image = tf.image.rot90(image, k=n_rot)
            label = tf.image.rot90(label, k=n_rot)

        if b_random_reflection:
            if tf.cast(tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.dtypes.int32), dtype=tf.dtypes.bool):
                image = tf.image.flip_left_right(image)
                label = tf.image.flip_left_right(label)

            if tf.cast(tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.dtypes.int32), dtype=tf.dtypes.bool):
                image = tf.image.flip_up_down(image)
                label = tf.image.flip_up_down(label)

        if b_random_contrast:
            image = tf.image.random_contrast(image, 0.5, 1.5)

        return image, label

    return random_augment_pipeline
