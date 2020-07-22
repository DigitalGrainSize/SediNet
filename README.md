# Warning, good things are coming. Major upgrade in progress. Listen to announcements on twitter or when this message disappears

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

You can use the models in this repository for your purposes (and you might find them useful because they have been trained on large numbers of images). If that doesn't work for you, you can train SediNet for your own purposes even on small datasets.

The examples have been curated with the following hardware specification in mind: 16 GB RAM, and Nvidia GPU with 11 GB of DDR4 or DDR6 memory (e.g. RTX 2080 Ti). If you have access to larger GPU memory, you can use larger imagery and larger batch sizes and you should achieve better accuracy.


### How SediNet works

Sedinet is a [deep learning](https://en.wikipedia.org/wiki/Deep_learning) model, which is a type of machine learning model that uses very large neural networks to automatically extract features from data to make predictions. For imagery, network layers typically use convolutions therefore the models are called [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) or CNNs for short.

CNNs have multiple processing layers (called [convolutional layers](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/) or blocks) and nonlinear transformations (that include [batch normalization](https://en.wikipedia.org/wiki/Batch_normalization), [activation](https://en.wikipedia.org/wiki/Activation_function), and [dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks))), with the outputs from each layer passed as inputs to the next. The model architecture is summarised below:

![Fig3-sedinet_fig_ann2_v3](https://user-images.githubusercontent.com/3596509/61979684-59a79700-afa9-11e9-9605-4f893784f65b.png)

SediNet is very configurable, and is designed primarily to be a research tool. There are two in-built model sizes (shallow and false), and numerous options for how to train and treat the data. For example, data inputs can optionally be scaled. Various image sizes can be used.  A single batch size may be chosen, or a model might be constructed using multiple batch sizes. Therefore it might take some experimentation to achieve optimal results for a particular dataset. Hopefully, this toolbox makes such experimentation straightforward. It isn't always obvious what combinations of settings to use, so be prepared to construct models using a variety of settings, then using the model with the best validation scores.


<!-- --------------------------------------------------------------------------------
## Run in your browser!

The following links will open jupyter notebooks in Google Colab, which is a free cloud computing service

### Categorical

##### Use SediNet to estimate sediment population

[Open this link]()

##### Use SediNet to estimate sediment shape

[Open this link]()

-->

<!-- #### Continuous

##### Sediment grain size prediction (sieve size) on a small population of beach sands

[Open this link]()

##### Sediment grain size prediction (9 percentiles of the cumulative distribution) on a small population of beach sands

[Open this link]()

##### Sediment grain size prediction (9 percentiles of the cumulative distribution) on a large 400 image dataset

[Open this link]()

-->

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

If you do NOT want to use your GPU for computations with tensorflow, edit the `conda_env/sedinet.yml` replacing `tensorflow-gpu` with `tensorflow`. This is NOT recommended for training models, only using them for prediction.

(if you are a regular or long-term conda user, perhaps this is a good time to ```conda clean --packages``` and ```conda update -n base conda```?)

```
conda env create -f conda_env/sedinet.yml
```

```
conda activate sedinet
```

(Later, when you're done ... ```conda deactivate sedinet```)

--------------------------------------------------------------------------------
## Train and use the provided example models yourself

The following examples have been selected to demonstrate the range of options you can choose when optimizing a SediNet model for a particular dataset. It therefore serves as a guide, rather than a gallery of best possible model outcomes. I encourage you to experiment with a few sets of options before deciding on a final optimal configuration and defaults file. Sometimes, using multiple batch sizes can be advantageous.

#### Continuous

##### Train SediNet for sediment grain size prediction (4 percentiles of the cumulative distribution plus sieve size) on a small population of beach sands

```
python sedinet_train.py -c config/config_sievedsand_sieve_plus.json
```

Subsequently predict using:
```
python sedinet_predict.py -c config/config_sievedsand_sieve_plus.json -w grain_size_sieved_sands/res_sieve_plus/sievesand_sieve_plus_simo_batch8_im512_512_6vars_pinball_aug_scale.hdf5
```

The above model has been trained with a single batch size of 8, with 768x768 pixel imagery, augmentation, and scaling

##### Train SediNet for sediment mid sieve size on a small population of beach sands

```
python sedinet_train.py -c config/config_sievedsand_sieve.json
```

Subsequently predict using:
```
python sedinet_predict.py -c config/config_sievedsand_sieve.json -w grain_size_sieved_sands/res_sieve/sievesand_sieve_siso_batch7_im512_512_1vars_pinball_aug_scale.hdf5
```

The above model has been trained with a single batch size of 8, with 768x768 pixel imagery, augmentation, and scaling


##### Train SediNet for sediment grain size prediction (9 percentiles of the cumulative distribution) on a large population of 400 images

```
python sedinet_train.py -c config/config_9percentiles.json
```

Subsequently predict using:
```
python sedinet_predict.py -c config/config_9percentiles.json -1 grain_size_global/res/global_9prcs_simo_batch12_im768_768_9vars_pinball_noaug.hdf5 -2 grain_size_global/res/global_9prcs_simo_batch13_im768_768_9vars_pinball_noaug.hdf5 -3 grain_size_global/res/global_9prcs_simo_batch14_im768_768_9vars_pinball_noaug.hdf5
```

The above model has been trained with multiple batch size of 12, 13 and 14, with 768x768 pixel imagery, no augmentation, and no variable scaling

#### Categorical

##### Train SediNet for sediment population prediction

```
python sedinet_train.py -c config/config_pop.json
```

Subsequently predict using:

```
python sedinet_predict.py -c config/config_pop.json -1 -2 -3
```
The above model has been trained with multiple batch size of 4, 6 and 8, with 768x768 pixel imagery, no augmentation, and no variable scaling (by default for categorical variables)


##### Train SediNet for sediment shape prediction

```
python sedinet_train.py -c config/config_shape.json
```

Subsequently predict using:

```
python sedinet_predict.py -c config/config_shape.json -1 grain_shape/res/grain_shape_siso_batch6_im768_768_shape_focal_noaug.hdf5 -2 grain_shape/res/grain_shape_siso_batch8_im768_768_shape_focal_noaug.hdf5 -3 grain_shape/res/grain_shape_siso_batch10_im768_768_shape_focal_noaug.hdf5
```
The above model has been trained with multiple batch size of 6, 8 and 10, with 768x768 pixel imagery, no augmentation, and no variable scaling (by default for categorical variables)

### Other examples


##### Train SediNet for sediment grain size prediction (9 percentiles of the cumulative distribution) on gravel images

```
python sedinet_train.py -c config/config_gravel.json
```

Subsequently predict using:

```
python sedinet_predict.py -c config/config_gravel.json -w grain_size_gravel_generic/res/gravel_generic_9prcs_simo_batch6_im768_768_9vars_pinball_aug.hdf5
```


##### Train SediNet for sediment grain size prediction (9 percentiles of the cumulative distribution) on sand images

```
python sedinet_train.py -c config/config_sand.json
```

Subsequently predict using:
```
python sedinet_predict.py -c config/config_sand.json -w grain_size_sand_generic/res_9prcs/sand_generic_9prcs_simo_batch12_im768_768_9vars_pinball_noaug_scale.hdf5
```


##### Train SediNet for sediment grain size prediction (3 percentiles of the cumulative distribution) on sand images

```
python sedinet_train.py -c config/config_sand_3prcs.json
```

Subsequently predict using:
```
python sedinet_predict.py -c config/config_sand_3prcs.json -w grain_size_sand_generic/res_3prcs/sand_generic_3prcs_simo_batch12_im768_768_3vars_pinball_noaug_scale.hdf5
```


##### Train SediNet for estimating mean size and sorting from images of mixed sand and gravel

```
python sedinet_train.py -c config/mattole.json
```

Subsequently predict using:

```
python sedinet_predict.py -c config/config_mattole.json -w mattole/res/mattole_simo_batch7_im512_512_2vars_pinball_aug.hdf5
```

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
* scale: if true, use scikit-learn's robust scaler to scale all variables. Otherwise, False for no scaling


### The defaults.py file

Contains values for defaults that you may change. They are listed in order of likelihood that you might change them:

```

# size of image in pixels. keep this consistent in training and application
# suggestd: 512 -- 1024 (larger = larger GPU required)
# integer
IM_HEIGHT = 768 #1024
IM_WIDTH = IM_HEIGHT #1024 #IM_HEIGHT

# number of images to feed the network per step in epoch #suggested: 4 --16
# integer
#BATCH_SIZE = 7

#use an ensemble of batch sizes like this
BATCH_SIZE = [4,6,8]

# if True, use a smaller (shallower) network architecture
##True or False ##False=larger network
SHALLOW = False #True

## if True, carry out data augmentation. 2 x number of images used in training
##True or False
DO_AUG = False #True

# maximum learning rate ##1e-1 -- 1e-4
MAX_LR = 1e-3

# max. number of training epics (20 -100)
# integer
NUM_EPOCHS = 100

## loss function for continuous models (2 choices)
CONT_LOSS = 'pinball'
#CONT_LOSS = 'mse'

## loss function for categorical (disrete) models (2 choices)
CAT_LOSS = 'focal'
#CAT_LOSS = 'categorical_crossentropy'

# optimizer (gradient descent solver) good alternative == 'rmsprop'
OPT = 'rmsprop' #'adam'

# base number of conv2d filters in categorical models
# integer
BASE_CAT = 30

# base number of conv2d filters in continuous models
# integer
BASE_CONT = 30

# number of Dense units for continuous prediction
# integer
CONT_DENSE_UNITS = 1024 #512

# number of Dense units for categorical prediction
# integer
CAT_DENSE_UNITS = 128

```

### Filename convention

For continuously distributed variables, file names are constructed according to the following convention

```
name "_" mode "_batch" batch_size "_im" IM_HEIGHT "_shallow_" varstring "_" CONT_LOSS "_aug_scale.hdf5"
```
if imagery is not augmented, `aug` in the above is replaced with `noaug`. If variables are not scaled, `_scale` is missing from the end

For categorical variables, we use

```
name "_" mode "_batch" batch_size "_im" IM_HEIGHT "_shallow_" varstring "_" CAT_LOSS "_aug.hdf5"
```

if imagery is not augmented, `aug` in the above is replaced with `noaug`. Categorical variables are never scaled

--------------------------------------------------------------------------------
## How to use on your own data

SediNet is very configurable. You can specify many variables in the config file, from the size of the imagery to use, to the number of models to ensemble and their respective batch sizes.

### Train your own SediNet for continuous variable prediction

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
  "scale": false

}
```

* The program will still expect your images to reside inside the 'images' folder

* You must label the file names in your csv file the same way as in the examples, i.e. "images/yourfilename.ext" and that column must be labeled 'files'


### Train your own SediNet for categorical prediction

Put together a config file in the config folder (called, say ```config_custom_colour.json```) and populate it like this example:

```
{
  "csvfile" : "dataset_colour.csv",
  "var"     : "colour",
  "numclass" : 6,
  "res_folder": "grain_colour",
  "name": "grain_colour",
  "dropout": 0.5,
}
```

Notes:

* Categorical variables are not scaled, therefore "scale" is ignored, if present
* Categories in the csvfile should be numeric integers increasing from zero

<!--
#### Using your model

Just the same way as the examples. For categorical ...

```
python sedinet_predict.py -c config/config_custom_colour.json
```

and continuous ...

```
python sedinet_predict.py -c config/config_custom_4prcs.json
``` -->


--------------------------------------------------------------------------------

## Release notes

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
12) Use of GPU is controlled by ```use_gpu``` = True \ False in the ```defaults.py``` script

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
12) now can take multiple batch sizes and build an ensemble model. This generally results in higher accuracy but more models = more model training time
13) response variables can be scaled using a robust scaler, or not. Use `scale=True` in a config file to use scaling
14) now checks for estimating weights path in root and `res_folder` directory and, if present, uses it. This can be used to add batch size combinations sequentially
15) optionally, training imagery is now augmented if `DO_AUG=True` (in the defaults file). This doubles the training set, by augmenting each image (random horizontal shift, followed by a vertical flip)
16) file names shorter (number of variables enumerated rather than each listed)
17) improved/rewritten README
18) more consistent and descriptive filenaming convention
19) simpler structure: `train` only does training (no prediction). Use `predict` to get train and test sets evaluation. This also allows defaulting to CPU for prediction, to avoid OOM errors that are more likely using GPU for prediction
20) no separate config file for prediction. One config file for both training and prediction
21) fixed many bugs, including one that was using 3-band greyscale imagery (doh!)
22) uses an exponentially decreasing learning rate scheduler rather than adaptive (because validation loss can be erratic)
23) uses depthwise separable 2d convolutions rather than trad 2d convs. see [here](https://keras.io/api/layers/convolution_layers/separable_convolution2d/)
24) variables in `defaults.py` based on consideration of accuracy across many datasets, both included and not included as pat of the SediNet package
25) categorical models also have a shallow and false option
26) `predict_all.sh` is a fully worked example of using the framework to predict on all continuous datasets

> The most important changes area
* depthwise separable convolution layers
* exponentially decreasing learning rate scheduler
* pinball loss for continuous variables
* focal loss and "shallow=False" for categorical variables

--------------------------------------------------------------------------------
## Other things

### Replicate the paper results

Note that you will see different results than in the paper because
* the implementation has changed, with more research, with a different loss function, image dimensions, and batch sizes
* training and testing files are randomly selected with a randomness that can't fully be controlled with a seed

### Known bugs

After long training periods, especially with multiple batch sizes, the `train` script gets killed at the end when it tries to use the model(s) in prediction mode. It is unclear why this happens. However, if you run the script again, with the same everything, this time it will skip the model training (assuming the `hdf5` files are still in the root directory or in `res_folder` - you'll see a `Loading weights that already exist:` message) and use the model weights to predict.

### If you have an issue, comment or suggestion ...
Please use the 'issues' tab so everyone can see the question and answer. Please do not email me directly. Thanks

<!-- ### Contribute your data!
Please see the [SediNet-Contrib repo](https://github.com/MARDAScience/SediNet-Contrib) -->

### Please cite
If you find this useful for your research please cite this paper:

> Buscombe, D. (2019). SediNet: a configurable deep learning model for mixed qualitative and quantitative optical granulometry. Earth Surface Processes and Landforms 45 (3), 638-651. https://onlinelibrary.wiley.com/doi/abs/10.1002/esp.4760

### Acknowledgements
Thanks to the following individuals for donating imagery:
* Rob Holman (Oregon State University)
* Dave Rubin (University of California Santa Cruz)
* Jon Warrick (US Geological Survey)
* Brian Romans (Virginia Tech)
* Christopher Heuberk (Freie Universitat Berlin)
* Sarah Joerger, Mike Smith (Northern Arizona University)

### Avoiding OOM (out of memory) errors

In order of trial:

1. use a smaller batch size (`BATCH_SIZE` in `defaults.py`)
2. turn augmentation off (`DO_AUG = False` in `defaults.py`)
3. use smaller imagery (`IM_WIDTH` and `IM_HEIGHT` in `defaults.py`)
4. use a bigger GPU


--------------------------------------------------------------------------------
## Notes for developers

### Organization
SediNet is organized as follows:

1. Model training

  * when `sedinet_train.py` is called, it first sets an operating system environmental variable that controls the use or otherwise of the GPU. It uses GPU 0 if use_GPU=True, otherwise GPU -1 (shorthand for CPU)
  * it imports everything in `sedinet_infer` which import everything in `sedinet_models`, so on for `sedinet_utils`, and finally `imports`
  * `imports` sets global variables and reads the `defaults.py`
  * then `sedinet_train.py` reads the specified (at the command line) `config` file, organizes the config variables, and finally calls `run_training_siso_simo` from `sedinet_infer`
  * `run_training_siso_simo` runs either `make_cat_sedinet` to make a categorical model, or `make_sedinet_siso_simo` to make a continuous model  (both called from `sedinet_models`)
  * then `train_sedinet_siso_simo` for continuous model training, or `train_sedinet_cat` for categorical (both called from `sedinet_infer`)
  * finally it calls `predict_test_train_siso_simo` or `predict_test_train_cat` for cont/cat variables (both called from `sedinet_utils`) and `tidy` moves the files into the `res_folder`, specified in the `config` file

2. Model prediction

  * Given a provided `config` file, `csv_file` and `weights_file`, use the model defined in the config file, load the weights, and estimate the variables on the images listed in the csv file
  * When `sedinet_predict.py` is called, it first sets an operating system environmental variable that controls the use or otherwise of the GPU. It uses GPU 0 if use_GPU=True, otherwise GPU -1 (shorthand for CPU)
  * it imports everything in `sedinet_eval` which import everything in `sedinet_models`, so on for `sedinet_utils`, and finally `imports`
  * `imports` sets global variables and reads the `defaults.py`
  * then `sedinet_predict.py` reads the specified (at the command line) `config` file, and weights file, organizes the config variables, and finally calls `estimate_siso_simo` or `estimate_categorical` from `sedinet_eval`
  * `estimate_siso_simo` runs `make_sedinet_siso_simo` to make a continuous model. `estimate_categorical` runs `make_cat_sedinet` to make a categorical model  (both called from `sedinet_models`)
  * then `predict_test_train_siso_simo` for continuous model training, or `predict_test_train_cat` for categorical (both called from `sedinet_utils`) and finally `tidy` moves the files into the `res_folder`, specified in the `config` file


### Contribute
If you wish to contribute to the development of this project (yes please!) it is better that you first fork this repository to your own github, then work on changes, and submit a pull request. Before submitting, please test your code changes by running a full set of tests in `predict_all.sh`, then verifying they all executed without error.

You can also contribute imagery this way, but if you do so, also please provide a dataset (csv file) that goes along with the imagery, a file that describes the data with your name and contact details, (and you should also thank yourself in this README!)

### Fork this repo and run on Google Cloud Platform (GCP)

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


--------------------------------------------------------------------------------
## Future plans

* predict on folder of sample images script
*  change batch generators into a better keras ones that will allow augmentation etc e.g [this](https://www.kaggle.com/amyjang/tensorflow-cnn-data-augmentation-prostate-cancer) or [this](https://www.kaggle.com/amyjang/tensorflow-pneumonia-classification-on-x-rays)
* multiple input, using pyDGS output perhaps? (unsupervised prior / covariate )
* k-folds cross-val for training
* transfer learning  
* aggregate over other hyperparameters besides batch size, such as loss function
