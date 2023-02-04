# Deep-SXT
Deep learning-based segmentation and reconstruction of cellular cryo-soft X-ray tomograms

This repository contains the software (in the form of a Python package) for 2D segmentation and 3D reconstruction of
cellular cryo soft x-ray tomograms.

The software deploys a deep learning pipeline based on an end-to-end convolutional architecture. 
The deep network designed for this application has been trained on limited manual labels using a semi-supervised scheme.

The software relies on a TensorFlow backend. To benefit from the GPU-accelerated deep learning provided by TensorFlow, 
you should run the software on a desktop PC equipped with a graphics card with an NVIDIA GPU.

When you use the software, please cite the following preprint:

```
@article {DyhrSadeghi2022deepsxt,
	author = {Dyhr, Michael C. A. and Sadeghi, Mohsen and 
	Moynova, Ralitsa and Knappe, Carolin and Kepsutlu, Burcu and 
	Werner, Stephan and Schneider, Gerd and McNally, James and 
	Noe, Frank and Ewers, Helge},
	title = {3D-surface reconstruction of cellular cryo-soft X-ray microscopy 
	tomograms using semi-supervised deep learning},
	elocation-id = {2022.05.16.492055},
	year = {2022},
	doi = {10.1101/2022.05.16.492055},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/05/16/2022.05.16.492055},
	journal = {bioRxiv}
}
```



Installation
---

We recommend using a conda environment for installing all the necessary components and running the notebooks.
See https://docs.conda.io/en/latest/miniconda for instructions on how to set up a local ```miniconda``` installation.

With miniconda installed on your machine, you can create a dedicated conda environment for the software and all its dependencies.
To do so, run the following command in terminal (substitute ```your_environment_name``` with a name of your choosing),

```
conda create -n your_environment_name python
```

Switch to the newly created environment:

```
conda activate your_environment_name
```

**Hint**: Further steps of the installation uses ```pip``` package manager.
It is good practice to check beforehand if the ```pip``` instance is invoked from the correct environment by
looking at the output of ```which pip``` and making sure that it points to the folder made for your new conda environment.

Having the conda environment up and running, you can easily proceed with the software installation in two steps:

1. Clone a local copy of the repository from **GitHub**. Navigate to a folder you intend to download the software in and run the following command:

```
git clone https://github.com/noegroup/deep_sxt.git
```

2. Navigate to the folder containing the cloned repository (e.g. ```cd deep_sxt```) and install the software package,

```
pip install --upgrade .
```

If everything goes according to plan, you should not need to do anything else, and you can start using the software.

The most common problem would be with GPU drivers and their compatibility with TensorFlow.
Make sure you have the latest NVIDIA drivers already installed on your machine.
You can check your current driver version via the ```nvidia-smi``` command.
NVIDIA drivers can be downloaded from https://www.nvidia.com/download/index.aspx.


Application
---

The two Jupyter Notebook files

- ```./notebooks/tomogram_2D_segmentation.ipynb```
- ```./notebooks/tomogram_3D_reconstruction.ipynb```

contain the scripts necessary for processing tomograms with Deep-SXT.
To access these notebooks, navigate to the ```notebooks``` folder and run,

```
jupyter notebook
```

This should open a browser window with a tree view of the two notebooks.
These notebooks should normally be run in the given oder to first obtain the 2D segmentation of all the slices in a cryo-sxt tomogram,
and then reconstruct the 3D surface representation using the 2D output.

Deep network training
---

Using this software does not require tweaking/training the deep network.
We have already obtained optimal network hyperparameters and have also fully trained the network using the semi-supervised approach described in the preprint.
Network parameters are already included in the repository in the ```./network_params``` folder.
When you run the ```./notebooks/tomogram_2D_segmentation.ipynb``` notebook for the first time,
a copy of network weights are also automatically downloaded to the same folder.

In case you think your application can benefit from retraining the network on a new dataset, please use the following notebook:

```
./notebooks/training_semi_supervised_segmentation.ipynb
```

some information about the training procedure is given in the notebook. For more detailes, please contact the authors.

Dependencies
---

- TensorFlow
- Jupyter
- Numpy
- Matplotlib
- PyVista
- scikit-image
- tifffile
- mrcfile
- tqdm
- meshio
- pymeshfix
- ipyvtklink

### Copyright

Copyright (c) 2022-2023, Mohsen Sadeghi (mohsen.sadeghi@fu-berlin),
Artificial Intelligence for the Sciences Group (AI4Science),
Freie Universität Berlin, Germany.

