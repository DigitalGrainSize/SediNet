# SediNet

<!-- ______     ______     _____     __     __   __     ______     ______  -->
<!--/\  ___\   /\  ___\   /\  __-.  /\ \   /\ "-.\ \   /\  ___\   /\__  _\ -->
<!--\ \___  \  \ \  __\   \ \ \/\ \ \ \ \  \ \ \-.  \  \ \  __\   \/_/\ \/ -->
<!-- \/\_____\  \ \_____\  \ \____-  \ \_\  \ \_\\"\_\  \ \_____\    \ \_\ -->
<!--  \/_____/   \/_____/   \/____/   \/_/   \/_/ \/_/   \/_____/     \/_/ -->

By Dr Daniel Buscombe
daniel.buscombe@nau.edu

Deep learning framework for optical granulometry (estimation of sedimentological variables from sediment imagery)

Accompanies the paper:

Buscombe, D. (2019, in review). SediNet: a configurable deep learning model for mixed qualitative and quantitative optical granulometry. Submitted to Earth Surface Processes and Landforms

### About SediNet

The idea behind SediNet is that you configure it for your own purposes, so there are several examples of different ways it can be configured for estimating categorical variables and various numbers of continuous variables 

However, you might also find any of these models useful for your purposes because they have been trained on large numbers of images


![Fig3-sedinet_fig_ann2_v3](https://user-images.githubusercontent.com/3596509/61979684-59a79700-afa9-11e9-9605-4f893784f65b.png)


### Install
You must have python 3, pip for python 3, git and conda. On Windows I recommend the latest Anaconda release. On Linux, git should come as standard and miniconda would be the way to go. Personally, I don't use conda but system builds (deb, yum, apt) within a virtual environment, but either way a VM of some description to contain SediNet would be a good idea. Mac users: ?

Windows:
```
git clone --depth 1 https://github.com/dbuscombe-usgs/SediNet.git
```

Linux/Mac:
```
git clone --depth 1 git@github.com:dbuscombe-usgs/SediNet.git
```

Anaconda/miniconda:

(if you are a regular or long-term conda user, perhaps this is a good time to ```conda clean --packages``` and ```conda update -n base conda```?)

```
conda env create -f conda_env/sedinet.yml
```

```
conda activate sedinet
```

(Later, when you're done ... ```conda deactivate sedinet```)


### If you have an issue, comment or suggestion ...
Please use the 'issues' tab so everyone can see the question and answer. Please do not email me directly. Thanks


### Replicate the paper results

(Note that you may get slightly different results than in the paper because training and testing files are randomly selected with a randomness that can't fully be controlled with a seed)

#### Continuous

##### Train SediNet for sediment grain size prediction (9 percentiles of the cumulative distribution) on a small population of beach sands

```
python train_sedinet_continuous.py -c config_sievedsand_9prcs.json
```

![sievesand_9prcs512_batch8_xy-base26_log](https://user-images.githubusercontent.com/3596509/62001390-40374580-b0a4-11e9-8803-1aabce95dab9.png)

##### Train SediNet for sediment mid sieve size on a small population of beach sands

```
python train_sedinet_continuous.py -c config_sievedsand_sieve.json
```

![sievesand_sieve512_batch8_xy-base22_log](https://user-images.githubusercontent.com/3596509/62001432-5bef1b80-b0a5-11e9-9a74-c613b1ad85d5.png)

##### Train SediNet for sediment grain size prediction (9 percentiles of the cumulative distribution) on a large population of 400 images

```
python train_sedinet_continuous.py -c config_9percentiles.json
```

![global_9prcs512_batch8_xy-base24_log](https://user-images.githubusercontent.com/3596509/62001561-64952100-b0a8-11e9-973b-b496f4e1dfee.png)

#### Categorical

##### Train SediNet for sediment population prediction

```
python train_sedinet_categorical.py -c config_pop.json
```

![pop_base22_model_checkpoint_cmT](https://user-images.githubusercontent.com/3596509/62001568-8ee6de80-b0a8-11e9-8e13-634a614e979b.png)

![pop_base22_model_checkpoint_cm](https://user-images.githubusercontent.com/3596509/62001571-927a6580-b0a8-11e9-89e1-10b88b760ceb.png)

##### Train SediNet for sediment shape prediction

```
python train_sedinet_categorical.py -c config_shape.json
```

![shape_base20_model_checkpoint_cmT](https://user-images.githubusercontent.com/3596509/62001821-d885f800-b0ad-11e9-9082-dca57913f8f8.png)

![shape_base20_model_checkpoint_cm](https://user-images.githubusercontent.com/3596509/62001822-da4fbb80-b0ad-11e9-846e-aa4d7cbd1256.png)


### Other examples

##### Train SediNet for sediment grain size prediction (sieve size plus 4 percentiles of the cumulative distribution) on a small population of beach sands

```
python train_sedinet_continuous.py -c config_sievedsand_sieve_plus.json
```

![sievesand_sieve_plus512_batch8_xy-base18_log](https://user-images.githubusercontent.com/3596509/62001639-817e2400-b0a9-11e9-920e-9b729873a41c.png)

##### Train SediNet for sediment grain size prediction (9 percentiles of the cumulative distribution) on gravel images

```
python train_sedinet_continuous.py -c config_gravel.json
```

![gravel_generic_9prcs512_batch8_xy-base16_log](https://user-images.githubusercontent.com/3596509/62001702-00c02780-b0ab-11e9-93d5-6d586e960b03.png)

##### Train SediNet for sediment grain size prediction (9 percentiles of the cumulative distribution) on sand images

```
python train_sedinet_continuous.py -c config_sand.json
```

![sand_generic_9prcs512_batch8_xy-base16_log](https://user-images.githubusercontent.com/3596509/62001865-b80a6d80-b0ae-11e9-8dcd-0c3c3030c366.png)

### The config file

A typical SediNet model configuration for predicting categorical variables is:

* base: base or minimum number of convolution filters in each SediNet block (e.g. 20)
* N: minimum number of examples per class (e.g. 5)
* csvfile: csv file containing image file names and corresponding categorical variable (e.g. "dataset_population.csv")
* var: name of column in csvfile to estimate (e.g. "pop") 
* numclass: number of classes within var (e.g. 6),
* res_folder: subdirectory name that contains csvfile (e.g. "grain_population")
* dropout: proportion of neurons to randomly drop before fully connected layer (e.g. 0.2)

A typical SediNet model configuration for predicting continuous variables is:

* base: base or minimum number of convolution filters in each SediNet block (e.g. 24)
* csvfile: csv file containing image file names and corresponding continuous variable (e.g. "data_set_400images.csv")
* res_folder: subdirectory name that contains csvfile (e.g. "grain_size_global")
* name: prefix of file names for outputs (e.g. "global_9prcs")
* variables: in the form "variable": "variable" (up to 9) 
* dropout: proportion of neurons to randomly drop before fully connected layer (e.g. 0.5)
* add_bn: true = add bathc normalization before fully connected layer (recommended) or false (do not add batch normalization)

### The defaults.py file

Contains values for defaults that you may change. They are listed in order of likelihood that you might change them:

TRAIN_TEST_SPLIT = 0.5  (the train/test split e.g. 0.5 = 50% train, 0.3 = 30% train, etc)

IM_HEIGHT = 512 (size of image in pixels. keep this consistent in training and application)

IM_WIDTH = IM_HEIGHT

num_epochs = 100 (max. number of training epics)

batch_size = 8 (number of images to feed the network per step in epoch)
 
valid_batch_size = batch_size

epsilon = 0.0001 (a tolerance for the training. Do not change until you've researched its effects)

min_lr = 0.0001 (minimum learning rate. lambda in the manuscript)

factor = 0.8 (the factor applied to the learning rate when the appropriate triggers are made - see paper)


### How to use on your own data

#### Train your own SediNet for continuous variable prediction

The SediNet training function ```train_sedinet_continuous.py``` is set up to predict arbitrary numbers of continuous variables. All your specific information (what data set to use, what to predict, etc) is contained in the config file and called the same way as above. For example:

```
python train_sedinet_continuous.py -c config_custom.json
```

where ```config_custom.json``` has ben put together by you in the config folder like the following example that would estimate the mean grain size and 4 arbitrary percentiles:

```
{
  "base" : 32,
  "csvfile" : "your_dataset.csv",
  "mean" : "mean",
  "P20": "P20", 
  "P40": "P46", 
  "P60": "P60", 
  "P80": "P80", 
  "res_folder": "my_custom_model",
  "name"  : "custom_4prcs"
}
```

* The program will still expect your images to reside inside the 'images' folder

* You must label the file names in your csv file the same way as in the examples, i.e. "images/yourfilename.ext" and that column must be labeled 'files'


#### Train your own SediNet for categorical prediction

Put together a config file in the config folder and populate it with the following fields:

* base:
* N:
* csvfile:
* var:
* numclass:
* res_folder: 


Example:

```
{
  "base" : 20,
  "N"  : 5,
  "csvfile" : "dataset_colour.csv",
  "var"     : "colour", 
  "numclass" : 6,
  "res_folder": "grain_colour"
}
```

* Note that categories in the csvfile should be numeric integers increasing from zero


### Contribute your data!
If you have data (images and corresponding labels or grain size information) you would like to contribute, please submit a pull request or email me!


### Acknowledgements
Thanks to the following individuals for donating imagery:
* Rob Holman (Oregon State University) 
* Dave Rubin (University of California Santa Cruz)
* Jon Warrick (US Geological Survey)
* Brian Romans (Virginia Tech)
* Christopher Heuberk (Freie Universitat Berlin) 

