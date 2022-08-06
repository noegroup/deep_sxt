# Deep learning-based segmentation and reconstruction of cellular cryo-soft X-ray tomograms

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

We recommend using a conda environment for installing all the necessary components and running the notebooks. See <a href="https://docs.conda.io/en/latest/miniconda.html">https://docs.conda.io/en/latest/miniconda.html</a> for instructions on how to establish a local conda installation.

You can start by making a dedicated conda environment. To do so, run the following command in terminal,

```
conda create -n your_environment_name python
```

Switich to the newly-made environment,
 
```
conda activate your_environment_name
```

You can optionally check if the ```pip``` instance is invoked from the correct environment by looking at the output of ```which pip``` which should point to the folder made for your new conda environment.

Having the conda environment up and running, you can install this software package easily. First obtain a local copy from github,

```
git clone https://github.com/noegroup/deep_sxt.git
```

Then run the following install command inside the folder containing the downloaded code,

```
pip install --upgrade .
```

If everthing goes according to plan, you should not need to do anything else and you can start using the software. The most common problem would be with hardware compatibility and driver availability. Make sure you have the latest NVIDIA drivers already installed on your machine. You can check driver version via the ```nvidia-smi``` command.

The two Jupyter Notebook files ```./notebooks/tomogram_2D_segmentation.ipynb``` and ```./notebooks/tomogram_3D_reconstruction.ipynb``` should be run in tandem to first obtain the 2D segmentation of all the slices in a cryo-sxt tomogram, and subsequently reconstruct the 3D surface representation using the 2D output. 

We have already developed the suitable architecture and its corresponding hyperparameters and have fully trained the network using the semi-supervised approach described in the preprint. Architectural parameters are already included in the ```./saved_models``` folder. First run of the ```./notebooks/tomogram_2D_segmentation.ipynb``` should download a copy of network weights to the same folder.

The notebook contains the code necessary for setting up the deep neural network as well as the dataset manager that are used in the manuscript for producing the reported results.

In the current setup of the code, which does not repeat the training but uses the weights from an already trained version, the whole notebook should run within a minute time, and produce plots of prediction accuracies, false positive/negatives (similar to Figure 2c of the manuscript) as well as false-color output of micrographs with nanobarcodes highlighted (similar to Extended Data Figure 6 of the manuscript).

Current setup of the code loads hyperparameters and network weights from the ```./data directory```. The flags in the code can easily be changed to retrain and save the network.

This notebook depends on local installations of these libraries

Dependencies
------------
 - **PyTorch** with CUDA enabled for network training/prediction on the GPU. The code is compatible with PyTorch version > 1.9 with cudatoolkit version > 11.1. See <a href="https://pytorch.org/get-started/locally">https://pytorch.org/get-started/locally</a> for instructions.
 - **Jupyter** with Python version 3.7 or newer.
 - **numpy** for arrays and numerical computations
 - **scipy** for signal processing functions
 - **matplotlib** for plotting
 - **scikit-image** for image processing
 - **tifffile** for image IO
 - **tqdm** for progress tracking

### Copyright

Copyright (c) 2022, Artificial Intelligence for the Sciences Group (AI4Science), Freie Universität Berlin, Germany.


