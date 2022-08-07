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
from scipy.signal.windows import gaussian
from skimage.measure import marching_cubes
import pymeshfix
import pyvista as pv
from tqdm.notebook import tqdm


def z_smooth (slices, smoothing_depth=2):
    """Performs smoothing on the segmented tomograms using a Gaussian filter

    Args:
        slices: numpy.ndarray
            3D int array containing the voxelized segmented tomogram
        smoothing_depth: int
            Slices in the range [-smoothing_depth, +smoothing_depth] are
            included in the Gaussian smoothing.

    Returns:
        : numpy.ndarray
            Smoothed-out slices with the same shape as the input stack

    """

    weight = gaussian(2 * smoothing_depth + 1, 1.0, True)

    smoothed_slices = []

    print("Smoothing tomogram...")

    for i in tqdm(range(slices.shape[0])):

        if i < smoothing_depth or i >= slices.shape[0] - smoothing_depth:
            smoothed_slices.append(slices[i, :, :].copy())
        else:
            interp_slice = np.zeros_like(slices[i, :, :])

            for j in range(len(weight)):
                interp_slice += weight[j] * slices[i - smoothing_depth + j, :, :]

            smoothed_slices.append((interp_slice > 0.0).astype(np.int32))

    return np.array(smoothed_slices, dtype=np.int32)


def surface_reconstruction(segmented_slices, flip_z=True):
    """Applies the Marching Cube algorithm to the voxelized tomogram

    Args:
        segmented_slices: numpy.ndarray
            3D int array containing the voxelized segmented tomogram
        flip_z: boolean
            Should flip the z-dimension before reconstruction

    Returns:
        : PyVista mesh
            Tessellated mesh of the reconstructed tomogram
    """

    seg_masks = np.transpose(segmented_slices, [1, 2, 0])

    if flip_z:
        seg_masks = np.flip(seg_masks, axis=2)

    print("Starting reconstruction...\n")

    verts, faces, normals, values = marching_cubes(seg_masks, method='lewiner',
                                                   level=None, allow_degenerate=False,
                                                   spacing=(1.0, 1.0, 1.0))

    print("Reconstruction successful!\n\n")
    print("Mesh statistics:")
    print("------------------")
    print(f"Number of vertices: {verts.shape[0]}")
    print(f"Number of faces: {faces.shape[0]}")
    print("\n")
    print("Fixing mesh...\n")

    mm = pymeshfix.MeshFix(verts, faces)

    print("Done!")

    return pv.wrap(mm.mesh)
