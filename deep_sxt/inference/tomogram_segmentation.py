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

import numpy as np
import tifffile
import h5py
import mrcfile


def load_tomogram_from_file(tomogram_file_path):
    """Loads tomogram image-stack from file.
    File type is deduced from file extension.

    Parameters:
        tomogram_file_path: string
            The path to tomogram file

    Returns:
        raw_tomogram: numpy.ndarray
            The stack of images loaded from file
    """

    file_extension = tomogram_file_path.split(".")[-1]

    print(f"File extension is determined as '{file_extension}'")

    try:
        if file_extension in ['tif', 'tiff']:
            print("loading the tomogram as TIFF...")
            raw_tomogram = tifffile.imread(tomogram_file_path).astype(np.float32)
        elif file_extension in ['rec', 'mrc', 'st']:
            print("loading tomogram with MRC...")
            f = mrcfile.open (tomogram_file_path, permissive=True)
            raw_tomogram = f.data.astype(np.float32)
        elif file_extension in ['h5']:
            print("loading tomogram as HDF5...")
            f = h5py.File(tomogram_file_path, mode='r')
            raw_tomogram = f["t0"]["channel0"][:, :, :].astype(np.float32)
        else:
            raise ValueError("Unknown tomogram filetype!")

        print(f"Tomogram loaded successfully from {tomogram_file_path}!")

    except Exception as inst:
        print(f"Tomogram load failed. Error:\n {inst}")
        raise

    v_max, v_min = np.amax(raw_tomogram), np.amin(raw_tomogram)

    if v_max > v_min:
        raw_tomogram = (raw_tomogram - v_min) / (v_max - v_min)

    print(f"Tomogram shape: {raw_tomogram.shape}")

    return raw_tomogram


def segment_slice(model, img, chunk_size=600, stride=400, use_rot=False):
    """Performs 2-dimensional segmentation on one slice of the tomogram using the deep model.

    Parameters:
        model :
            the deep model used for segmentation
        img : numpy.ndarray
            the slice image
        chunk_size : int
            size of image chunks (in pixels) to use in each invocation of the network
        stride : int
            strides taken over the image. Should be smaller than chunk_size
        use_rot : bool
            If True, each chunk of the image will also be fed to the network in 90-deg rotations
            and the result averaged over.

    Returns:
        output_lbl: numpy.ndarray
            Binary-mask segmented image with the same size as the input image
    """

    stride = min(stride, chunk_size)
    stride_x = stride
    stride_y = stride

    while chunk_size % 16 > 0:
        chunk_size -= 1
    while (img.shape[0] - chunk_size) % stride_x > 0:
        stride_x -= 1
    while (img.shape[1] - chunk_size) % stride_y > 0:
        stride_y -= 1

    output_lbl = np.zeros_like(img)
    n_output = np.zeros_like(img)

    d_ind_x = np.array([stride_x, stride_x], dtype=np.int32)
    d_ind_y = np.array([stride_y, stride_y], dtype=np.int32)

    ind_y = np.array([0, chunk_size], dtype=np.int32)

    while ind_y[1] <= img.shape[1]:

        ind_x = np.array([0, chunk_size], dtype=np.int32)

        while ind_x[1] <= img.shape[0]:

            if use_rot:
                for k in range(4):
                    chunk_img = np.rot90(img[ind_x[0]:ind_x[1], ind_y[0]: ind_y[1]], k=k)

                    chunk_output = model(chunk_img.reshape((1, *chunk_img.shape, 1)), training=False)

                    predicted_lbl = np.rot90(chunk_output[1].numpy()[0, :, :, 0].astype(np.float32), k=-k)
                    output_lbl[ind_x[0]:ind_x[1], ind_y[0]: ind_y[1]] += predicted_lbl
                    n_output[ind_x[0]:ind_x[1], ind_y[0]: ind_y[1]] += np.ones_like(chunk_img)
            else:
                chunk_img = img[ind_x[0]:ind_x[1], ind_y[0]: ind_y[1]]

                chunk_output = model(chunk_img.reshape((1, *chunk_img.shape, 1)), training=False)

                predicted_lbl = chunk_output[1].numpy()[0, :, :, 0].astype(np.float32)
                output_lbl[ind_x[0]:ind_x[1], ind_y[0]: ind_y[1]] += predicted_lbl
                n_output[ind_x[0]:ind_x[1], ind_y[0]: ind_y[1]] += np.ones_like(chunk_img)

            ind_x += d_ind_x

        ind_y += d_ind_y

    output_lbl /= n_output

    return output_lbl
