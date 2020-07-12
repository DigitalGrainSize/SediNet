
## Written by Daniel Buscombe,
## MARDA Science
## daniel@mardascience.com

##> Release v1.3 (July 2020)

###===================================================
# import libraries
import sys, getopt, json, os
from numpy import any as npany
from sedinet_eval import *

#==============================================================
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:c:w:")
    except getopt.GetoptError:
        print('python sedinet_predict.py -w weightsfile.hdf5 -c configfile.json')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python sedinet_predict.py -w mattole/mattole_simo_batch8_p10_p16_p25_p50_p75_p84_p90__checkpoint.hdf5 -c config/config_mattole_predict.json')
            sys.exit()
        elif opt in ("-c"):
            configfile = arg
        elif opt in ("-w"):
            weights_path = arg

    #weights_path = 'mattole/res/color/mattole_simo_batch8_p10_p16_p25_p50_p75_p84_p90__checkpoint.hdf5'
    #configfile = '/home/filfy/Desktop/mattole_predict/config_mattole_predict_samples.json'

    if not os.path.isfile(os.getcwd()+os.sep+weights_path):
        if not os.path.isfile(weights_path):
           print("Weights path does not exist ... exiting")
           sys.exit()

    try:
       # load the user configs
       with open(os.getcwd()+os.sep+configfile) as f:
          config = json.load(f)
    except:
       # load the user configs
       with open(configfile) as f:
          config = json.load(f)

    ###===================================================
    ## user defined variables: proportion of data to use for training (a.k.a. the "train/test split")
    csvfile = config["csvfile"] #csvfile containing image names and class values
    res_folder = config["res_folder"] #folder containing csv file and that will contain model outputs
    name = config["name"] #name prefix for output files
    greyscale = config["greyscale"] #convert imagery to greyscale or not
    dropout = config["dropout"]

    try:
       numclass = config['numclass']
    except:
       numclass = 0

    vars = [k for k in config.keys() if not npany([k.startswith('csvfile'), k.startswith('dropout'), k.startswith('base'), k.startswith('res_folder'), k.startswith('train_csvfile'), k.startswith('test_csvfile'), k.startswith('name'), k.startswith('greyscale'), k.startswith('aux_in'), k.startswith('N'), k.startswith('numclass')])]

    vars = sorted(vars)

    auxin = [k for k in config.keys() if k.startswith('aux_in')]

    if len(auxin) > 0:
       auxin = config[auxin[0]]   ##at least for now, just one 'auxilliary' (numerical/categorical) input in addition to imagery
       if len(vars) ==1:
          mode = 'miso'
       elif len(vars) >1:
          mode = 'mimo'
    else:
       if len(vars) ==1:
          mode = 'siso'
       elif len(vars) >1:
          mode = 'simo'

    print("Mode: %s" % (mode))
    ###==================================================

    csvfile = res_folder+os.sep+csvfile

    if (mode=='siso' or mode=='simo'):
       if numclass>0:
          estimate_categorical(vars, csvfile, res_folder, dropout, numclass, greyscale, weights_path)
       else:
          estimate_siso_simo(vars, csvfile, greyscale, weights_path, dropout)

    if (mode=='miso' or mode=='mimo'):
       estimate_miso_mimo(vars, csvfile, greyscale, auxin, weights_path, dropout)
