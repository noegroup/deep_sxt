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


import numpy as np
import os
import glob
import tifffile
import tensorflow as tf
from utilities.io import *
from tqdm.notebook import tqdm
import skimage.restoration as ski_restore


def label_preprocessing_function(_lbl):

    v_max = np.amax(_lbl)
    v_min = np.amin(_lbl)

    if np.abs(v_max - v_min) > 0.0:
        return (2.0 * (_lbl - v_min) / (v_max - v_min) - 1.0).astype(np.float32)
    else:
        return -np.ones_like(_lbl, dtype=np.float32)


def image_preprocessing_function(img, mu=None, sigma=None):

    if mu is None:
        mu = np.mean(img)

    if sigma is None:
        sigma = np.std(img)

    _img = (img - mu) / sigma

    return _img.astype(np.float32)


class ImageLoader:

    def __init__(self, dataset_folders=None,
                 test_split_fraction=0.2,
                 use_labels=True,
                 image_processing_function=None, label_processing_function=None):

        self.dataset_folders = []
        self.input_file_list = []
        self.use_labels = use_labels
        self.max_dim = None
        self.image_processing_function = lambda x: x
        self.label_processing_function = lambda x: x

        self.dataset_mean = None
        self.dataset_std = None

        if dataset_folders is not None:
            self.dataset_folders = dataset_folders

        if image_processing_function is not None:
            self.image_processing_function = image_processing_function

        if label_processing_function is not None:
            self.label_processing_function = label_processing_function

        if self.use_labels:

            for _f in self.dataset_folders:

                files = glob.glob(_f + "/*.labels.tif")

                for _file in files:

                    file_name = _file[:-11]

                    if os.path.isfile(file_name + ".tif"):
                        self.input_file_list.append(file_name)

                writeln(f"{len(files)} image files fetched from {_f} with corresponding labels!")
        else:

            for _f in self.dataset_folders:

                files = glob.glob(_f + "/*.tif")

                for _file in files:

                    file_name = _file[:-4]
                    self.input_file_list.append(file_name)

                writeln(f"{len(files)} image files fetched from {_f} without labels!")

        if len(self.input_file_list) > 0:

            self.__determine_image_dimensions(self.input_file_list)

            self.training_file_names, self.test_file_names = self.__split_train_test(test_split_fraction)

            self.trn_img_list, self.trn_lbl_list, self.trn_orig_size_list = self.__load_images(self.training_file_names)
            self.test_img_list, self.test_lbl_list, self.test_orig_size_list = self.__load_images(self.test_file_names)

    def __split_train_test(self, test_split_fraction):

        train_set_len = int(len(self.input_file_list) * (1.0 - test_split_fraction))

        ind = np.arange(len(self.input_file_list))

        sub = np.random.choice(ind, train_set_len, replace=False)

        ind_train = ind[sub]
        ind_test = np.delete(ind, sub)

        assert len(np.intersect1d(ind_train, ind_test)) == 0

        training_file_names = []
        test_file_names = []

        for _ind in ind_train:
            training_file_names.append(self.input_file_list[_ind])

        for _ind in ind_test:
            test_file_names.append(self.input_file_list[_ind])

        writeln(f"Number of images in the training set: {len(training_file_names)}")
        writeln(f"Number of images in the test set: {len(test_file_names)}")

        return training_file_names, test_file_names
    
    def __imread(self, file_name):
        
        img = tifffile.imread(file_name)
        
        if img.dtype == np.uint8:
            max_brightness_value = 255.0
        elif img.dtype == np.uint16:
            max_brightness_value = 65535.0
        else:
            raise ValueError("Unknown image format!")
        
        return img.astype(np.float32) / max_brightness_value
    
    def __determine_image_dimensions(self, file_names):

        write("Processing image information...")

        image_sum = image_sum2 = image_n = 0.0

        orig_size_list = []

        for _file_name in tqdm(file_names):

            img_file_name = _file_name + ".tif"
            img = self.__imread(img_file_name)

            image_n += np.prod(np.array(img.shape).astype(np.float32))
            image_sum += np.sum(img)
            image_sum2 += np.sum(img ** 2)

            if self.use_labels:
                lbl_file_name = _file_name + ".labels.tif"
                lbl = self.__imread(lbl_file_name)

                if img.shape != lbl.shape:
                    raise ValueError(f"Image dimensions ({img.shape}) and label dimensions ({lbl.shape}) don't match!")

            orig_size_list.append(img.shape)

        writeln("done!")

        self.max_dim = np.amax(np.array(orig_size_list))
        self.dataset_mean = (image_sum / image_n)
        self.dataset_std = np.sqrt((image_sum2 / image_n) - (image_sum / image_n) ** 2)

        writeln(f"Max. image dimensions = {self.max_dim}")
        writeln(f"Mean of all brightness values = {self.dataset_mean}")
        writeln(f"Standard deviation of all brightness values = {self.dataset_std}")

    def __load_images(self, file_names):

        img_list = []
        lbl_list = []
        orig_size_list = []

        for _file_name in tqdm(file_names):

            img_file_name = _file_name + ".tif"
            img = self.image_processing_function(self.__imread(img_file_name),
                                                 self.dataset_mean, self.dataset_std)
            img_size = img.shape

            p11 = (self.max_dim - img_size[0]) // 2
            p21 = self.max_dim - img_size[0] - p11

            p12 = (self.max_dim - img_size[1]) // 2
            p22 = self.max_dim - img_size[1] - p12

            img = np.pad(img, ((p11, p21), (p12, p22)), mode='mean')
            img_list.append(img.reshape((self.max_dim, self.max_dim, 1)).copy())

            if self.use_labels:
                lbl_file_name = _file_name + ".labels.tif"
                lbl = self.label_processing_function(self.__imread(lbl_file_name))
                lbl_size = lbl.shape

                assert img_size == lbl_size

                lbl = np.pad(lbl, ((p11, p21), (p12, p22)), constant_values=-1.0)
                lbl_list.append(lbl.reshape((self.max_dim, self.max_dim, 1)).copy())

            orig_size_list.append(np.array(img_size, dtype=np.int32))

        img_list, lbl_list, orig_size_list = np.array(img_list, dtype=np.float32),\
                                             np.array(lbl_list, dtype=np.float32),\
                                             np.array(orig_size_list, dtype=np.int32)

        return img_list, lbl_list, orig_size_list

    def save_to_file(self, file_name):

        write("Saving zipped datasets to file...")

        np.savez(file_name, self.trn_img_list, self.trn_lbl_list, self.trn_orig_size_list,
                 self.test_img_list, self.test_lbl_list, self.test_orig_size_list)

        writeln("done!")

    def load_from_file(self, file_name):

        write("Loading zipped datasets from file...")

        dat = np.load(file_name + ".npz")

        self.trn_img_list, self.trn_lbl_list, self.trn_orig_size_list,\
        self.test_img_list, self.test_lbl_list, self.test_orig_size_list = dat["arr_0"].astype(np.float32),\
                                                                           dat["arr_1"].astype(np.float32),\
                                                                           dat["arr_2"].astype(np.int32),\
                                                                           dat["arr_3"].astype(np.float32),\
                                                                           dat["arr_4"].astype(np.float32),\
                                                                           dat["arr_5"].astype(np.int32)

        self.max_dim = np.amax(self.trn_img_list.shape[1:3])

        writeln("done!")

        writeln(f"Max. image dimensions = {self.max_dim}")

    def get_dataset(self):

        write("Compiling tensorflow datasets...")

        trn_img_list = tf.data.Dataset.from_tensor_slices(self.trn_img_list)
        if self.use_labels:
            trn_lbl_list = tf.data.Dataset.from_tensor_slices(self.trn_lbl_list)
        else:
            trn_lbl_list = tf.data.Dataset.from_tensor_slices(self.trn_img_list)
        trn_size_list = tf.data.Dataset.from_tensor_slices(self.trn_orig_size_list)

        test_img_list = tf.data.Dataset.from_tensor_slices(self.test_img_list)
        if self.use_labels:
            test_lbl_list = tf.data.Dataset.from_tensor_slices(self.test_lbl_list)
        else:
            test_lbl_list = tf.data.Dataset.from_tensor_slices(self.test_img_list)
        test_size_list = tf.data.Dataset.from_tensor_slices(self.test_orig_size_list)

        writeln("done!")

        return tf.data.Dataset.zip((trn_img_list, trn_lbl_list, trn_size_list)),\
            tf.data.Dataset.zip((test_img_list, test_lbl_list, test_size_list))
