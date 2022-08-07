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
from deep_sxt.utilities.io import *
from .loss import my_accuracy_function
from tqdm.notebook import tqdm


class MyTrainer:
    def __init__(self, model, training_dataset, test_dataset,
                 optimizer=None, loss_function=None, accuracy_function=None, callback_class=None):

        self.model = model
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

        if loss_function is None:
            loss_function = tf.keras.losses.Huber(delta=0.1)

        if accuracy_function is None:
            accuracy_function = my_accuracy_function

        self.optimizer = optimizer
        self.loss_function = loss_function
        self.accuracy_function = accuracy_function

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')

        self.epoch = None
        self.train_loss_list, self.test_loss_list = None, None
        self.train_accuracy_list, self.test_accuracy_list = None, None

        self.reset_training_loop()

        self.callback_class = callback_class

    @tf.function
    def __train_step(self, images, labels):

        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.loss_function(labels, predictions)
            accuracy = self.accuracy_function(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy)

    @tf.function
    def __test_step(self, images, labels):
        predictions = self.model(images, training=False)

        loss = self.loss_function(labels, predictions)
        accuracy = self.accuracy_function(labels, predictions)

        self.test_loss(loss)
        self.test_accuracy(accuracy)

    def reset_training_loop(self):

        self.epoch = 0
        self.train_loss_list = []
        self.test_loss_list = []
        self.train_accuracy_list = []
        self.test_accuracy_list = []

    def one_epoch(self):

        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

        if self.callback_class is not None:
            self.callback_class.on_epoch_begin(self.epoch)

        for images, labels in tqdm(self.training_dataset):
            self.__train_step(images, labels)

        for test_images, test_labels in tqdm(self.test_dataset):
            self.__test_step(test_images, test_labels)

        if self.callback_class is not None:
            self.callback_class.on_epoch_end(self.epoch)

        self.epoch += 1

        template = 'Epoch {}, Loss: {:5.4f}, Accuracy: {:5.2f}%, Test Loss: {:5.4f}, Test Accuracy: {:5.2f}%'

        writeln(template.format(self.epoch,
                                self.train_loss.result(),
                                self.train_accuracy.result() * 100.0,
                                self.test_loss.result(),
                                self.test_accuracy.result() * 100.0))

        self.train_loss_list.append(self.train_loss.result())
        self.train_accuracy_list.append(self.train_accuracy.result())

        self.test_loss_list.append(self.test_loss.result())
        self.test_accuracy_list.append(self.test_accuracy.result())

