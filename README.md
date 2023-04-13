# Agglomerative Clustering for Page Stream Segmentation

This repository contains the code and link to the data used in the paper "Using Deep Learned Vector Representations for Page Stream Segmentation by Agglomerative Clustering".

## Installation

To run the experiments presented in this research, we recommend using `Aconda` to install the dependencies required, and we have included a .yml file to easily install all the required dependencies. Although we recommend the usage of `conda` we have also included a `requirements.txt` file, so the required dependencies can also be installed using  `pip`.

Below are the steps for installing the required dependencies:

- Step 1: Clone /Download the repository to your local computer.

- Step 2 :Unzip the directory if needed, and change into the directory in the terminal, for example `cd path/to/folder/ `

- Step 3: If you are using conda, you can use the command below to create a new conda environment with all the requirements you need to run the code in this repository.

`conda env create -f environment.yml`


This will create an environment called 'PSS_SIGIR_env', which can be used to run the experiments in this notebook. Note that this environment comes with `jupyter lab` already installed, but without a link to the evironment, so you can't select it in Jupyter Lab yet.
To do this, first activate the environment:

`conda activate PSS_SIGIR_env`

Then run the following command:

`ipython kernel install --name "SIGIR_PSS_experiments_kernel" --user`

Now the kernel is linked to Jupyter Lab, and you can select it as the kernal to run the notebooks in Jupyter Lab. to start Jupyter Lab, make sure the environment is activated and type `jupyter lab` in the terminal.

If you want to install via `pip` please run the command below: 'pip install -r requirements.txt'


  
## Contents

This repository contains the code to run the experiments from the paper, with the following folders in it. <br>

**Experiments**
- This folder contains three notebooks. `Experiments.ipynb`, which contains the main experimental results, `DataExploration.ipynb` which contains some basic analysis on the dataset, and `ImageModelTraining.ipynb`, which contains the code for training the VGG16 model. <br>


**utils**
- utils contains two files, with the `metricutils.py` file implementing the metrics used in this research, and `utils.py` implementing various helper functions used for clustering.

**Data** 

The data folder contains the resources required to run the experiments in this research. The folder contains the following:
  - dataframes
    - Folder with the dataframes for train and test data containing the text and labels of the isntances in the dataset
  - images
    - Folder containing the images of the pages in the dataset in the format required by the VGG16 model
  - trained_VGG16_model
    - assets for the trained VGG16 model used to perform the predictions of the binary classification baseline.
  - finetuned_vectors.npy, pretrained_vectors.npy
    - Image vectors of the train and test portions of the dataset, extracted from the pretrained and finetuned VGG16 model
  - gold_standard.json
    - JSON file containing the binary label for each page in the dataset


## Downloading the dataset

The dataset used in this research is available through Zenodo (https://zenodo.org/record/7683111) as an anonymous data entry. The instructions for downloading the data are pretty straightforward you simply have to unzip the file and put it in the main folder of the repository, but we have also included a script in this repository that will allow you to automatically download the data as part of the set up of the repository. You can simply run this with `bash download.sh`
