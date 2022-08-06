# 3D-surface reconstruction of cellular cryo-soft X-ray microscopy tomograms using semi-supervised deep learning

This repository holds the software for deep-learning-based segmentation and 3D reconstruction of cryo soft x-ray tomorams.

Please cite this preprint when you use the software,

```
@article {DyhrSadeghi2022deepsxt,
	author = {Dyhr, Michael C. A. and Sadeghi, Mohsen and Moynova, Ralitsa and Knappe, Carolin and Kepsutlu, Burcu and Werner, Stephan and Schneider, Gerd and McNally, James and Noe, Frank and Ewers, Helge},
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

We recommend making a conda environment for installing all the necessary components and running the notebook. See <a href="https://docs.conda.io/en/latest/miniconda.html">https://docs.conda.io/en/latest/miniconda.html</a> for instructions.

The two Jupyter Notebook files ```./tomogram_2D_segmentation.ipynb``` and ```./tomogram_3D_reconstruction.ipynb``` should be run in tandem to first obtain the 2D segmentation of all the slices in a tomogram, and subsequently reconstruct the 3D surface representation based on the 2D output. 

We have already developed a suitable architecture and have fully trained the network using the semi-supervised approach described in the preprint. Architectural parameters as well as network weights are saved in the ```./trained_models``` folder. 

The notebook contains the code necessary for setting up the deep neural network as well as the dataset manager that are used in the manuscript for producing the reported results.

In the current setup of the code, which does not repeat the training but uses the weights from an already trained version, the whole notebook should run within a minute time, and produce plots of prediction accuracies, false positive/negatives (similar to Figure 2c of the manuscript) as well as false-color output of micrographs with nanobarcodes highlighted (similar to Extended Data Figure 6 of the manuscript).

Current setup of the code loads hyperparameters and network weights from the ```./data directory```. The flags in the code can easily be changed to retrain and save the network.

This notebook depends on local installations of these libraries

 - **PyTorch** with CUDA enabled for network training/prediction on the GPU. The code is compatible with PyTorch version > 1.9 with cudatoolkit version > 11.1. See <a href="https://pytorch.org/get-started/locally">https://pytorch.org/get-started/locally</a> for instructions.
 - **Jupyter** with Python version 3.7 or newer.
 - **numpy** for arrays and numerical computations
 - **scipy** for signal processing functions
 - **matplotlib** for plotting
 - **scikit-image** for image processing
 - **tifffile** for image IO
 - **tqdm** for progress tracking

This notebook additionally relies on the included custom library ```./dnn_classifier``` for network and data handler components.

Running this code requires a desktop PC with a graphics card containing an NVIDIA GPU.

The local installation of all the required components should take < 10 min, if no problems with hardware compatibility and driver availability arise.




