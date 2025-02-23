# Anaconda setup
## Assumptions
This README file was written based on an environment with the following versions loaded:
* Windows 11 Home 23H2 OS Build 22631.4890
* Anaconda Navigator 2.6.4
* Visual Studio Code 1.97.2
## Setup
* Start Anaconda Navigator.
* From the Home panel, launch Visual Studio Code (VSC).
* Open the `audio-deepfake-detection` project in VSC.
* Start a terminal in VSC.
* Run:
  ```
  conda env create -f .\environment.yml
  ```
* After the environment has finished loading, run:
  ```
  conda activate audio-deepfake-detection
  ```
## Configure
### Warning
The project is currently configured to use Anaconda though originally it was created to use `google-colab`. Efforts have been made to continue to support `google-colab`, but it has not been tested.
### Options
* `active-data-source`: One of the immediate subkeys of `data-sources`. If no other changes are made to the configuration file, choose between `local-windows` or `google-colab`.
* `model.name`: Select name of machine language model file to create for the first part of the Jupyter Notebook, or the model file to load in the "Visualization" section of the Notebook.
* `data-sources`: One or more configurations for the data path of the source data. Each subkey under `data-sources` must be unique. Only one of these keys can be selected for `active-data-source`.
  * Currently two keys are needed under each `data-sources` subkey:
    * `vendor`: Key for any initialization needed to access a particular data source (ie: `google-colab` indicates that the program needs to call `drive.mount("/content/drive")`)
    * `dataPath`: Path to source data. Environment variables can be used.


## Update `environment.yml` and `requirements.txt` for commit to source control.
The following commands can be used to save changes to dependencies in the environment:
* Using Anaconda:
  ```
  conda env export -n audio-deepfake-detection-v5 > environment.yml
  ```
* Using PIP:
  ```
  pip freeze > requirements.txt
  ```