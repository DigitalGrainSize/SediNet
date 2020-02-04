
## Written by Daniel Buscombe,
## MARDA Science
## daniel@mardascience.com

##> Release v1.2 (Feb 4 2020)

###===================================================
# import libraries
import sys, getopt, json, os
from numpy import any as npany
from sedinet_infer import *    

#==============================================================	
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:c:")
    except getopt.GetoptError:
        print('python sedinet_train.py -c configfile.json')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python sedinet_train.py -c config/config_9percentiles.json')
            sys.exit()
        elif opt in ("-c"):
            configfile = arg
    
    #configfile = 'config/config_mattole.json'  

    # load the user configs
    with open(os.getcwd()+os.sep+configfile) as f:
        config = json.load(f)     
         
    ###===================================================
    ## user defined variables: proportion of data to use for training (a.k.a. the "train/test split")
    train_csvfile = config["train_csvfile"] #csvfile containing image names and class values
    test_csvfile = config["test_csvfile"] #csvfile containing image names and class values
    res_folder = config["res_folder"] #folder containing csv file and that will contain model outputs
    name = config["name"] #name prefix for output files
    greyscale = config["greyscale"] #convert imagery to greyscale or not
    dropout = config["dropout"] #dropout factor
    
    try:
       numclass = config['numclass']
    except:
       numclass = 0
                        
    #output variables            
    vars = [k for k in config.keys() if not npany([k.startswith('base'), k.startswith('res_folder'), k.startswith('train_csvfile'), k.startswith('test_csvfile'), k.startswith('name'), k.startswith('greyscale'), k.startswith('aux_in'), k.startswith('dropout'), k.startswith('N'), k.startswith('numclass')])]
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

    train_csvfile = res_folder+os.sep+train_csvfile
    test_csvfile = res_folder+os.sep+test_csvfile
    
    #with strategy.scope():
    #    run_continuous_training(vars, csvfile, base, name, res_folder, dropout, add_bn)
    if (mode=='siso' or mode=='simo'):
       run_training_siso_simo(vars, train_csvfile, test_csvfile, name, res_folder, mode, greyscale, dropout, numclass) 

    if (mode=='miso' or mode=='mimo'):
       run_training_miso_mimo(vars, train_csvfile, test_csvfile, name, res_folder, mode, greyscale, auxin, dropout, numclass) 


      

