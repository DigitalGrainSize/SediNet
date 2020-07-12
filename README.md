# Upgrade in progress. please don't use until the next update when this message will disappear. Thanks



# SediNet: Build your own sediment descriptor

<!-- ______     ______     _____     __     __   __     ______     ______  -->
<!--/\  ___\   /\  ___\   /\  __-.  /\ \   /\ "-.\ \   /\  ___\   /\__  _\ -->
<!--\ \___  \  \ \  __\   \ \ \/\ \ \ \ \  \ \ \-.  \  \ \  __\   \/_/\ \/ -->
<!-- \/\_____\  \ \_____\  \ \____-  \ \_\  \ \_\\"\_\  \ \_____\    \ \_\ -->
<!--  \/_____/   \/_____/   \/____/   \/_/   \/_/ \/_/   \/_____/     \/_/ -->

By Dr Daniel Buscombe

daniel@mardascience.com

Deep learning framework for [optical granulometry](https://en.wikipedia.org/wiki/Optical_granulometry) (estimation of sedimentological variables from sediment imagery).

--------------------------------------------------------------------------------
## About SediNet
A configurable machine-learning framework for estimating either (or both) continuous and categorical variables from a photographic image of clastic sediment. It has wide potential application, even to subpixel imagery and complex mixtures, because the dimensions of the grains aren't being measured directly or indirectly, but using a mapping from image to requested output using a machine learning algorithm that you have to train using examples of your data.

For more details, please see the paper:

Buscombe, D. (2019). SediNet: a configurable deep learning model for mixed qualitative and quantitative optical granulometry. Earth Surface Processes and Landforms. https://onlinelibrary.wiley.com/doi/abs/10.1002/esp.4760

Free Earth ArXiv preprint [here](https://eartharxiv.org/fwsnp/)

This repository contains code and data to reproduce the above paper, as well as additional examples and jupyter notebooks that you can run on the cloud and use as examples to build your own Sedinet sediment descriptor

The algorithm implementation has changed since the paper, so the results are  slightly different but the concepts and data, and most of everything, have not changed.

SediNet can be configured and trained to estimate:
* up to nine numeric grain-size metrics in pixels from a single input image. Grain size is then recovered using the physical size of a pixel (note that sedinet doesn't help you estimate that). Appropriate metrics include mean, median or any other percentile
* equivalent sieve diameters directly from image features, without the need for area-to-mass conversion formulas and without even knowing the scale of one pixel. SediNet might be useful for other metrics such as sorting (standard deviation), skewness, etc. There could be multiple quantities that could be estimated from the imagery
* categorical variables such as grain shape, population, colour, etc

The motivating idea behind SediNet is community development of tools for information extraction from images of sediment. You can use SediNet "off-the-shelf", or other people's models, or configure it for your own purposes.

<!-- You can even choose to contribute imagery back to the project, so we can build bigger and better models collaboratively. If that sounds like something you would like to do, there is a [special repo](https://github.com/MARDAScience/SediNet-Contrib) for you wonderful people -->

Within this package there are several examples of different ways it can be configured for estimating categorical variables and various numbers of continuous variables

You can use the models in this repository for your purposes (and you might find them useful because they have been trained on large numbers of images). If that doesn't work for you, you can train SediNet for your own purposes even on small datasets



### How SediNet works

Sedinet is a [deep learning](https://en.wikipedia.org/wiki/Deep_learning) model, which is a type of machine learning model that uses very large neural networks to automatically extract features from data to make predictions. For imagery, network layers typically use convolutions therefore the models are called [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) or CNNs for short.

CNNs have multiple processing layers (called [convolutional layers](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/) or blocks) and nonlinear transformations (that include [batch normalization](https://en.wikipedia.org/wiki/Batch_normalization), [activation](https://en.wikipedia.org/wiki/Activation_function), and [dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks))), with the outputs from each layer passed as inputs to the next. The model architecture is summarised below:

![Fig3-sedinet_fig_ann2_v3](https://user-images.githubusercontent.com/3596509/61979684-59a79700-afa9-11e9-9605-4f893784f65b.png)


<!-- --------------------------------------------------------------------------------
## Run in your browser!

The following links will open jupyter notebooks in Google Colab, which is a free cloud computing service

### Categorical

##### Use SediNet to estimate sediment population

[Open this link](https://colab.research.google.com/drive/1M-hX5oBS2K0hof4oVG15oN_vNSi9AD2E)

##### Use SediNet to estimate sediment shape

[Open this link](https://colab.research.google.com/drive/1mSxuJzto6QReAddGZ6A0fZ_g1o9qLljN)

which is equivalent to the following respective commands from the command line using an installed SediNet (see below):

```
python sedinet_predict.py -c config/config_pop.json
```

and

```
python sedinet_predict.py -c config/config_shape.json
``` -->

<!-- #### Continuous

##### Sediment grain size prediction (sieve size) on a small population of beach sands

[Open this link](https://colab.research.google.com/drive/1CFkE4meWHQ7ylmWSN01BO8qJu2fcpvcS)

##### Sediment grain size prediction (9 percentiles of the cumulative distribution) on a small population of beach sands

[Open this link](https://colab.research.google.com/drive/1GYZUVkLLQQhJygwsrkR11dDWAlLCy5aJ)

##### Sediment grain size prediction (9 percentiles of the cumulative distribution) on a large 400 image dataset

[Open this link](https://colab.research.google.com/drive/11sm53rkS-dYjanBPVAAUM7VnjEBWH59v)


which is equivalent to the following respective commands from the command line using an installed SediNet (see below):

```
python sedinet_predict.py -c config/config_sievedsand_sieve.json
```

```
python sedinet_predict.py -c config/config_sievedsand_9prcs.json
```

and

```
python sedinet_predict.py -c config/config_9percentiles.json
``` -->


<!-- #### Other Continuous Examples

##### Sediment grain size prediction (9 percentiles of the cumulative distribution) on generic sands

[Open this link](https://colab.research.google.com/drive/1kQoWmyUOOQFYNebTi6t9VZP5HRkgjxLa)

##### Sediment grain size prediction (9 percentiles of the cumulative distribution) on generic gravels

[Open this link](https://colab.research.google.com/drive/1VHciGSyp4wsgo8w6mSbxlP_A228rScQ_)

##### Sediment grain size prediction (sieve size plus 4 percentiles of the cumulative distribution) on a small population of beach sands

[Open this link](https://colab.research.google.com/drive/1oXtYZJ3niDm3XOeKG1xikOZSjqIBVbQD)

which is equivalent to the following respective commands from the command line using an installed SediNet (see below):

```
python sedinet_predict.py -c config/config_sand.json
```

```
python sedinet_predict.py -c config/config_gravel.json
```

and

```
python sedinet_predict.py -c config/config_sievedsand_sieve_plus.json
``` -->

--------------------------------------------------------------------------------
## Install and run on your computer
You must have python 3, pip for python 3, git and conda. On Windows I recommend the latest [Anaconda](https://www.anaconda.com/distribution/) release.

Windows:
```
git clone --depth 1 https://github.com/MARDAScience/SediNet.git
```

Linux/Mac:
```
git clone --depth 1 git@github.com:MARDAScience/SediNet.git
```

Anaconda/miniconda:

If you want to use your GPU for computations with tensorflow, edit the `conda_env/sedinet.yml` replacing `tensorflow` with `tensorflow-gpu`

(if you are a regular or long-term conda user, perhaps this is a good time to ```conda clean --packages``` and ```conda update -n base conda```?)

```
conda env create -f conda_env/sedinet.yml
```

```
conda activate sedinet
```

(Later, when you're done ... ```conda deactivate sedinet```)


#### Install and instructions for developers
If you wish to contribute to the development of this project (yes please!) it is better that you first fork this repository to your own github, then work on changes, and submit a pull request. Before submitting, please test your code changes by running a full set of tests in `predict_all.sh`, then verifying they all executed without error.

You can also contribute imagery this way, but if you do so, also please provide a dataset (csv file) that goes along with the imagery, a file that describes the data with your name and contact details, (and you should also thank yourself in this README!)


--------------------------------------------------------------------------------
## Fork this repo and run on Google Cloud Platform (GCP)

First, follow instructions [here](https://tudip.com/blog-post/run-jupyter-notebook-on-google-cloud-platform/) for how to set up an instance to run in GCP. Make sure to set a static IP address, as per the instructions, and make a note of that because you'll need it later

Then open a shell into the VM and set it up to

```
ssh-keygen -t rsa -b 4096 -C "yourname@youremail.com"

eval "$(ssh-agent -s)"

ssh-add ~/.ssh/id_rsa

cat ~/.ssh/id_rsa.pub
```

Then copy the key into your github profile keys. For more information about how to do that, see [here](https://help.github.com/en/articles/adding-a-new-ssh-key-to-your-github-account). xclip likely won't work, but you can simply copy (Ctrl-C) the text printed to screen


You will be cloning your fork of the main repo, so replace ```YOURUSERNAME``` in the below code to clone the repo and set up a conda environment to run in

```
git clone --depth 1 git@github.com:YOURUSERNAME/SediNet.git
cd SediNet

pip install --upgrade pip

conda env create -f conda_env/sedinet.yml

source activate sedinet
```

Now you can run sedinet on the cloud.

To run the jupyter notebooks, run the following command to run the jupyter notebook server

```
python -m ipykernel install --user
jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000
```

The jupyterlab server will be displayed at

```
http://IP:8888
```

where ```IP``` is the static IP of the VM that you noted earlier.


<!-- --------------------------------------------------------------------------------
## Run the interactive applications in your web browser

```
bokeh serve RunSediNet.ipynb
```

which should open a web application to run through your browser at http://localhost:60339/ -->

--------------------------------------------------------------------------------
## Replicate the paper results

Note that you will see different results than in the paper because
* the implementation has changed, with more research, with a different loss function, image dimensions, and batch sizes
* training and testing files are randomly selected with a randomness that can't fully be controlled with a seed

--------------------------------------------------------------------------------
### Predict grain size/shape/population from a set of images


The ipynb files in the ```notebooks``` directory are the same jupyter notebooks as in the Colab notebook links above. You can run them by

```
conda activate sedinet
python -m ipykernel install --user
jupyter notebook
```

then in your browser you should be able to navigate to the notebooks you wish to execute

The following instructions are for running the provided python scripts on your computer

--------------------------------------------------------------------------------
#### Continuous

##### Sediment grain size prediction (sieve size) on a small population of beach sands
```
python sedinet_predict.py -c config/config_sievedsand_sieve_predict.json -w grain_size_sieved_sands/res/color/sievesand_sieve_siso_batch8_sieve__checkpoint.hdf5
```

<!-- ![sievesand_sieve_siso_batch8_sieve__checkpoint_skill](https://user-images.githubusercontent.com/53406404/73790480-ca501d00-475d-11ea-8fca-60b6fdc5efa1.png) -->


##### Sediment grain size prediction (9 percentiles of the cumulative distribution) on a small population of beach sands

```
python sedinet_predict.py -c config/config_sievedsand_sieve_plus_predict.json -w grain_size_sieved_sands/res/grey/sievesand_sieve_plus_simo_batch8_P16_P25_P50_P75_P84_sieve__checkpoint.hdf5
```

<!-- ![sand_generic_9prcs_simo_batch8_P10_P16_P25_P5_P50_P75_P84_P90_P95__checkpoint_skill](https://user-images.githubusercontent.com/53406404/73790362-970d8e00-475d-11ea-8a8d-aae8504de15c.png) -->


##### Sediment grain size prediction (9 percentiles of the cumulative distribution) on a large 400 image dataset

```
python sedinet_predict.py -c config/config_9percentiles_predict.json -w grain_size_global/res/grey/global_9prcs_simo_batch8_P10_P16_P25_P5_P50_P75_P84_P90_P95__checkpoint.hdf5
```

<!-- ![global_9prcs_simo_batch8_P10_P16_P25_P5_P50_P75_P84_P90_P95__checkpoint_skill](https://user-images.githubusercontent.com/53406404/73790415-a8569a80-475d-11ea-96fa-a31d207d25ad.png) -->

--------------------------------------------------------------------------------
#### Categorical

##### Use SediNet to estimate sediment population

```
python sedinet_predict.py -c config/config_pop.json -w grain_population/res/color/pop_model_checkpoint.hdf5
```

<!-- ![pop_model_checkpoint_cmT](https://user-images.githubusercontent.com/53406404/73790144-39794180-475d-11ea-8236-ecc7edf5bece.png) -->

##### Use SediNet to estimate sediment shape

```
python sedinet_predict.py -c config/config_shape_predict.json -w grain_shape/res/color/shape_model_checkpoint.hdf5
```

<!-- ![shape_model_checkpoint_cmT](https://user-images.githubusercontent.com/53406404/73790124-2f574300-475d-11ea-9584-dcf4723cd1db.png) -->



--------------------------------------------------------------------------------
## Other Examples

##### Sediment grain size prediction (9 percentiles of the cumulative distribution) on generic sands

```
python sedinet_predict.py -c config/config_sand_predict.json -w grain_size_sand_generic/res/grey/sand_generic_9prcs_simo_batch8_P10_P16_P25_P5_P50_P75_P84_P90_P95__checkpoint.hdf5
```

<!-- ![sand_generic_9prcs512_batch8_xy-base16_predict](https://user-images.githubusercontent.com/3596509/62002419-6e268500-b0b8-11e9-8c1a-83fc54e9d66a.png) -->


##### Sediment grain size prediction (9 percentiles of the cumulative distribution) on generic gravels

```
python sedinet_predict.py -c config/config_gravel_predict.json -w grain_size_gravel_generic/res/grey/gravel_generic_9prcs_simo_batch8_P10_P16_P25_P5_P50_P75_P84_P90_P95__checkpoint.hdf5
```

<!-- ![gravel_generic_9prcs512_batch8_xy-base16_predict](https://user-images.githubusercontent.com/3596509/62002214-46352280-b0b4-11e9-84fc-65e66116386b.png) -->


##### Sediment grain size prediction (sieve size plus 4 percentiles of the cumulative distribution) on a small population of beach sands

```
python sedinet_predict.py -c config/config_sievedsand_sieve_plus.json
```

<!-- ![sievesand_sieve_plus_simo_batch8_P16_P25_P50_P75_P84_sieve__checkpoint_skill](https://user-images.githubusercontent.com/53406404/73790479-ca501d00-475d-11ea-817b-1c7745108aab.png) -->


##### Sediment grain size prediction (7 percentiles of the cumulative distribution) on a large population of mixed sand and gravel beach sediment (collected by Sarah Jorger, NAU)

```
python sedinet_predict.py -c config/config_mattole.json
```

<!-- ![mattole_simo_batch8_p10_p16_p25_p50_p75_p84_p90__checkpoint_skill](https://user-images.githubusercontent.com/53406404/73790700-22871f00-475e-11ea-83ce-37ad72745171.png) -->



--------------------------------------------------------------------------------
## Train the models yourself

#### Continuous

##### Train SediNet for sediment grain size prediction (9 percentiles of the cumulative distribution) on a small population of beach sands

```
python sedinet_train.py -c config/config_sievedsand_sieve_plus.json
```

<!-- ![sievesand_9prcs512_batch8_xy-base26_log](https://user-images.githubusercontent.com/3596509/62001390-40374580-b0a4-11e9-8803-1aabce95dab9.png) -->

##### Train SediNet for sediment mid sieve size on a small population of beach sands

```
python sedinet_train.py -c config/config_sievedsand_sieve.json
```
<!--
![sievesand_sieve512_batch8_xy-base22_log](https://user-images.githubusercontent.com/3596509/62001432-5bef1b80-b0a5-11e9-9a74-c613b1ad85d5.png) -->

##### Train SediNet for sediment grain size prediction (9 percentiles of the cumulative distribution) on a large population of 400 images

```
python sedinet_train.py -c config/config_9percentiles.json
```

<!-- ![global_9prcs512_batch8_xy-base24_log](https://user-images.githubusercontent.com/3596509/62001561-64952100-b0a8-11e9-973b-b496f4e1dfee.png) -->

#### Categorical

##### Train SediNet for sediment population prediction

```
python sedinet_train.py -c config/config_pop.json
```

<!-- ![pop_base22_model_checkpoint_cmT](https://user-images.githubusercontent.com/3596509/62001568-8ee6de80-b0a8-11e9-8e13-634a614e979b.png)

![pop_base22_model_checkpoint_cm](https://user-images.githubusercontent.com/3596509/62001571-927a6580-b0a8-11e9-89e1-10b88b760ceb.png) -->

##### Train SediNet for sediment shape prediction

```
python sedinet_train.py -c config/config_shape.json
```

<!-- ![shape_base20_model_checkpoint_cmT](https://user-images.githubusercontent.com/3596509/62001821-d885f800-b0ad-11e9-9082-dca57913f8f8.png)

![shape_base20_model_checkpoint_cm](https://user-images.githubusercontent.com/3596509/62001822-da4fbb80-b0ad-11e9-846e-aa4d7cbd1256.png) -->


### Other examples

##### Train SediNet for sediment grain size prediction (sieve size plus 4 percentiles of the cumulative distribution) on a small population of beach sands

```
python sedinet_train.py -c config/config_sievedsand_sieve_plus.json
```

<!-- ![sievesand_sieve_plus512_batch8_xy-base18_log](https://user-images.githubusercontent.com/3596509/62001639-817e2400-b0a9-11e9-920e-9b729873a41c.png) -->

##### Train SediNet for sediment grain size prediction (9 percentiles of the cumulative distribution) on gravel images

```
python sedinet_train.py -c config/config_gravel.json
```

<!-- ![gravel_generic_9prcs_simo_batch8_P10_P16_P25_P5_P50_P75_P84_P90_P95__checkpoint_skill](https://user-images.githubusercontent.com/53406404/73790586-f10e5380-475d-11ea-8f47-fb9b6d518eaa.png) -->


##### Train SediNet for sediment grain size prediction (9 percentiles of the cumulative distribution) on sand images

```
python sedinet_train.py -c config/config_sand.json
```

<!-- ![sand_generic_9prcs512_batch8_xy-base16_log](https://user-images.githubusercontent.com/3596509/62001865-b80a6d80-b0ae-11e9-8dcd-0c3c3030c366.png) -->

--------------------------------------------------------------------------------
## More details about inputs and using this tool on your own data

### The config file

A typical SediNet model configuration for predicting categorical variables is:

* train_csvfile: csv file containing image file names and corresponding categorical variable for training (e.g. "dataset_population_train.csv")
* test_csvfile: csv file containing image file names and corresponding categorical variable for testing (e.g. "dataset_population_test.csv")
* var: name of column in csvfile to estimate (e.g. "pop")
* numclass: number of classes within var (e.g. 6),
* res_folder: subdirectory name that contains csvfile (e.g. "grain_population")
* dropout: proportion of neurons to randomly drop before fully connected layer (e.g. 0.2)

A typical SediNet model configuration for predicting continuous variables is:

* train_csvfile: csv file containing image file names and corresponding continuous variable for training (e.g. "data_set_400images_train.csv")
* test_csvfile: csv file containing image file names and corresponding continuous variable for testing (e.g. "data_set_400images_test.csv")
* res_folder: subdirectory name that contains csvfile (e.g. "grain_size_global")
* name: prefix of file names for outputs (e.g. "global_9prcs")
* variables: in the form "variable": "variable" (up to 9)
* dropout: proportion of neurons to randomly drop before fully connected layer (e.g. 0.5)
* greyscale: true = use greyscale version of the image, or false (use color version)

### The defaults.py file

Contains values for defaults that you may change. They are listed in order of likelihood that you might change them:

```

# set to False if you wish to use cpu (not recommended)
USE_GPU = True  ##False

# size of image in pixels. keep this consistent in training and application
IM_HEIGHT = 600 # suggestd: 512 -- 1024
IM_WIDTH = IM_HEIGHT

# max. number of training epics
NUM_EPOCHS = 10 #100

# number of images to feed the network per step in epoch
BATCH_SIZE =  [2,4,6] #suggested: 4 --16

# if True, use a smaller (shallower) network architecture
SHALLOW = False ##False=larger network

# optimizer (gradient descent solver) good alternative == 'adam'
OPT = 'rmsprop'

## loss function for continuous models (2 choices)
CONT_LOSS = 'pinball'
## CONT_LOSS = 'mse'

## loss function for continuous models (2 choices)
CAT_LOSS = 'focal'
#CAT_LOSS = 'categorical_crossentropy'

# a tolerance for the training. Do not change until you've researched its effects
MIN_DELTA = 0.0001

# minimum learning rate (lambda in the manuscript)
MIN_LR = 1e-5 #1e-5 -- 1e-2

# the factor applied to the learning rate when the appropriate triggers are made
FACTOR = 0.8

# training stops early if the number of successive epochs with no validation loss exceeds this number
STOP_PATIENCE = 15

# base number of conv2d filters in categorical models
BASE_CAT = 30

# base number of conv2d filters in continuous models
BASE_CONT = 30

# number of Dense units for categorical prediction
CAT_DENSE_UNITS = 128

# number of Dense units for continuous prediction
CONT_DENSE_UNITS = 1024

```

### How to use on your own data

SediNet is very configurable. You can specify many variables in the config file, from the size of the imagery to use, to the number of models to ensemble and their respective batch sizes.

#### Train your own SediNet for continuous variable prediction

The SediNet training function ```train_sedinet_continuous.py``` is set up to predict arbitrary numbers of continuous variables. All your specific information (what data set to use, what to predict, etc) is contained in the config file and called the same way as above. For example:

```
python train_sedinet_continuous.py -c config/config_custom_4prcs.json
```

where ```config/config_custom_4prcs.json``` has ben put together by you in the config folder like the following example that would estimate the mean grain size and 4 arbitrary percentiles:

```
{
  "train_csvfile" : "your_train_dataset.csv",
  "test_csvfile" : "your_test_dataset.csv",
  "mean" : "mean",
  "P20": "P20",
  "P40": "P46",
  "P60": "P60",
  "P80": "P80",
  "res_folder": "my_custom_model",
  "name"  : "custom_4prcs",
  "dropout": 0.5,
  "greyscale": true  
}
```

* The program will still expect your images to reside inside the 'images' folder

* You must label the file names in your csv file the same way as in the examples, i.e. "images/yourfilename.ext" and that column must be labeled 'files'


#### Train your own SediNet for categorical prediction

Put together a config file in the config folder (called, say ```config_custom_colour.json```) and populate it like this example:

```
{
  "csvfile" : "dataset_colour.csv",
  "var"     : "colour",
  "numclass" : 6,
  "res_folder": "grain_colour",
  "name": "grain_colour",
  "dropout": 0.5,
  "greyscale': false
}
```

* Note that categories in the csvfile should be numeric integers increasing from zero


#### Using your model

Just the same way as the examples. For categorical ...

```
python sedinet_predict.py -c config/config_custom_colour.json
```

and continuous ...

```
python sedinet_predict.py -c config/config_custom_4prcs.json
```

--------------------------------------------------------------------------------

## Other things

### If you have an issue, comment or suggestion ...
Please use the 'issues' tab so everyone can see the question and answer. Please do not email me directly. Thanks

### Contribute your data!
Please see the [SediNet-Contrib repo](https://github.com/MARDAScience/SediNet-Contrib)

### Please cite
If you find this useful for your research please cite this paper:

> Buscombe, D. (2019). SediNet: a configurable deep learning model for mixed qualitative and quantitative optical granulometry. Earth Surface Processes and Landforms

### Acknowledgements
Thanks to the following individuals for donating imagery:
* Rob Holman (Oregon State University)
* Dave Rubin (University of California Santa Cruz)
* Jon Warrick (US Geological Survey)
* Brian Romans (Virginia Tech)
* Christopher Heuberk (Freie Universitat Berlin)
* Sarah Joerger, Mike Smith (Northern Arizona University)

### Benchmark test results
Using the following settings ...

```
USE_GPU = True  ##False
IM_HEIGHT = 600
IM_WIDTH = IM_HEIGHT
NUM_EPOCHS = 100
BATCH_SIZE =  [4,6,8]
SHALLOW = False
OPT = 'rmsprop'
CONT_LOSS = 'pinball'
CAT_LOSS = 'focal'
MIN_DELTA = 0.0001
MIN_LR = 1e-5
FACTOR = 0.8
STOP_PATIENCE = 15
BASE_CAT = 30
BASE_CONT = 30
CAT_DENSE_UNITS = 128
CONT_DENSE_UNITS = 1024
```


#### Global (400 images), 9 percentiles

* Mean percent error for D50 (train / test)

| Batch/Image size| 768   | 1024  |
| ------ | ------ | ------|
| 6,8,12      | X      |X      |
| 2,4,6      | 36 / 30      |X      |
| 4,6,8      | 34 / 25      |X      |



#### Pesacdero Sand Sieve Sizes

* Mean percent error for sieve size (sediment that lands on a sieve of mesh size) (train / test)

| Batch/Image size| 768   | 1024  |
| ------ | ------ | ------|
| 4      | X      |X      |
| 2,4,6      | X      |X      |
| 4,6,8      | X      |X      |



#### Grain shape

* Mean accuracy for grain shape (train / test)

| Batch/Image size| 768   | 1024  |
| ------ | ------ | ------|
| 4      | X      |X      |
| 2,4,6      | X     |X      |
| 4,6,8      | 0.95 / 0.68      |X      |

#### Grain population

* Mean accuracy for grain population (train / test)

| Batch/Image size| 768   | 1024  |
| ------ | ------ | ------|
| 4      | X      |X      |
| 2,4,6      | X     |X      |
| 4,6,8      | X     |X      |


### Release notes

> Release v1.0 (Sep 30 2019): initial submission of SediNet paper to journal

[![DOI](https://zenodo.org/badge/199072106.svg)](https://zenodo.org/badge/latestdoi/199072106)


> Release v1.1 (Nov 5 2019): upgrade from keras with Tensorflow 1.X backend to Tensorflow 2.0 native keras. Enforce TF==2.0 in conda yml file

> Release v1.2 (Feb 4 2020): major upgrade with the following improvements:
1) Additional dataset (Mattole); a mixed sand-gravel beach data set collected in summer 2019 by Sarah Joerger as part of her MS in Geology at Northern Arizona University
2) Robust continuous variable scaling; all response variables now get scaled using scikit-learn's RobustScaler, which removes the median and scales the data according to the quantile range. See https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html. . This information is contained in the scaler.pkl file. Categorical model training is unchanged
3) Removed user requirement in previous versions for specifying base model size; previously, the overall size of the model was controlled by a variable ```base```, specifying the minimum number of filters in the first convolutional block. Now, that number is fixed (30) and the user has an option to make the model more or less shallow, which has the effect of adding an additional convolutional block. This is controlled using ```shallow = True/False``` in ```defaults.py```.  Applies to both categorical and continuous variables
4) Removed bias in continuous model predictions; the final step is to fit a linear model to (x,y-predicted) to remove any small biases left in predictions. This information is contained in the bias.pkl file. Categorical model training is unchanged
5) 1024x1024 image size used by default (larger than previous). For color imagery on smaller GPUs, you may have to compromise between batch size and image size. I did the latter, using 768x768x3 pixels
6) The user now creates two csv files -- one for training files and associated values, and one for testing files and their associated variables
7) Sedinet no longer creates (uses) ensembles of models (model predictions), which speeds up model training and execution. Applies to both categorical and continuous variables
8) Set things up so multiple inputs can be used to predict outputs (i.e. imagery plus an attribute) -- however this is not yet and may not ever be implemented
9) Default now uses rmsprop optimizer (instead of adam)
10) Color as well as greyscale imagery (user optional)
11) Simpler workflow of ```train``` followed by ```predict``` - no longer any separate scripts for continous and categorical variables. Adding ```numclasses``` to the config file tells the project to use a categorical variable workflow
12) Use of GPU is controlled by ```use_gpu``` = True\False in the ```defaults.py``` script

> Update 26 June 2020:
1) updated yml file and tested env on linux (pop!OS, ubuntu) and windows (10)
2) updated README
3) switched from deprecated `from sklearn.externals import joblib` to `joblib`
4) added a folder of updated jupyter notebooks

> Release v1.3
> Update July 2020:
1) fixed generator error by adding exception and print statement to `get_data_generator_Nvars_siso_simo`, `get_data_generator_Nvars_miso_mimo`, and `get_data_generator_1image`
2) new optimised `defaults.py` values (image size = 768, batch_size = 4, shallow=True)
3) added `BASE_CAT` and `BASE_CONT` options to `defaults.py`
4) added image size, model depth, to output file names
5) added `CAT_DENSE_UNITS` and `CONT_DENSE_UNITS` options to `defaults.py`
6) added `CONT_LOSS` and `CAT_LOSS` options to `defaults.py`, with defaults from `tensorflow_addons` (conda env yml updated). loss for categorical models can now be `focal` (default) or `categorical_crossentropy`. Models for continuous variables can now be `pinball` (default)
7) all global variables are now capitalized, for readability/etc
8) general tidy up of code for readability
9) fixed bug in categorical model training (expected ndarray, not list, for batch labels)
10) fixed bug in categorical model plotting
11) added LICENSE
12) dynamically grow the memory used on the GPU
13) new benchmarking section of the readme tabulating results with default settings
14) now can take multiple batch sizes and build an ensemble model. This generally results in higher accuracy but more models = more model training time



x
