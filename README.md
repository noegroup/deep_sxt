# Deep-SXT
### deep learning-based segmentation and reconstruction of cellular cryo-soft X-ray tomograms

This repository holds the software for 2D segmentation and subsequent 3D reconstruction of cellular cryo soft x-ray tomorams.
The deep learning method is based on a specifically designed end-to-end convolutional architecture and has been trained on limited manual labels in a semi-supervised scheme. 

When you use the software, please cite the following preprint:

```
@article {DyhrSadeghi2022deepsxt,
	author = {Dyhr, Michael C. A. and Sadeghi, Mohsen and Moynova, Ralitsa and Knappe, Carolin and Kepsutlu, Burcu and
	 Werner, Stephan and Schneider, Gerd and McNally, James and Noe, Frank and Ewers, Helge},
	title = {3D-surface reconstruction of cellular cryo-soft X-ray microscopy tomograms using semi-supervised deep learning},
	elocation-id = {2022.05.16.492055},
	year = {2022},
	doi = {10.1101/2022.05.16.492055},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/05/16/2022.05.16.492055},
	eprint = {https://www.biorxiv.org/content/early/2022/05/16/2022.05.16.492055.full.pdf},
	journal = {bioRxiv}
}
```

The software is implemented in Python using a TensorFlow backend. To benefit from the GPU-accelerated deep learning provided by TensorFlow, you should run the software on a desktop PC equipped with a graphics card with an NVIDIA GPU.

Installation
---

We recommend using a conda environment for installing all the necessary components and running the notebooks. See <a href="https://docs.conda.io/en/latest/miniconda.html">https://docs.conda.io/en/latest/miniconda.html</a> for instructions on how to establish a local conda installation.

You can start by creating a dedicated conda environment. To do so, run the following command in terminal,

```
conda create -n your_environment_name python
```

Switch to the newly created environment,
 
```
conda activate your_environment_name
```

**Hint**: it is good practice to check if the ```pip``` instance is invoked from the correct environment by looking at the output of ```which pip``` and making sure that it points to the folder made for your new conda environment.

Having the conda environment up and running, you can install this software package easily.

First clone a local copy of the repository from **github**:

```
git clone https://github.com/noegroup/deep_sxt.git
```

Then navigate to the folder containing the cloned repository and install the software package,

```
pip install --upgrade .
```

If everything goes according to plan, you should not need to do anything else, and you can start using the software.

The most common problem would be with GPU drivers and their compatibility with TensorFlow. Make sure you have the latest NVIDIA drivers already installed on your machine. You can check driver version via the ```nvidia-smi``` command.


Application
---

The two Jupyter Notebook files

 - ```./notebooks/tomogram_2D_segmentation.ipynb```
 - ```./notebooks/tomogram_3D_reconstruction.ipynb```

contain the scripts necessary for processing tomograms with Deep-SXT.

These notebooks should be run in the given oder to first obtain the 2D segmentation of all the slices in a cryo-sxt tomogram, and then reconstruct the 3D surface representation using the 2D output. 

Using this software does not require training the deep network.
We have already obtained optimal network hyperparameters and have also fully trained the network using the semi-supervised approach described in the preprint.
Network parameters are already included in the repository in the ```./saved_models``` folder.
When you run the ```./notebooks/tomogram_2D_segmentation.ipynb``` notebook for the first time, a copy of network weights are automatically downloaded to the same folder.


Dependencies
---

 - **TensorFlow**
 - **Jupyter**
 - **numpy**
 - **matplotlib**
 - **tifffile**
 - **tqdm**

### Copyright

Copyright (c) 2022, Artificial Intelligence for the Sciences Group (AI4Science), Freie Universität Berlin, Germany.


