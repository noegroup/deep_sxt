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


@tf.function
def my_accuracy_function(label, prediction):

    prediction = tf.clip_by_value(0.5 * prediction + 0.5, 0.0, 1.0)
    label = tf.clip_by_value(0.5 * label + 0.5, 0.0, 1.0)

    pred_score = 1.0 - tf.norm(label - prediction, 1) / tf.norm(label, 1)

    return tf.clip_by_value(pred_score, 0.0, 1.0)
    