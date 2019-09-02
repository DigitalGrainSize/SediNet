
# ______     ______     _____     __     __   __     ______     ______  
#/\  ___\   /\  ___\   /\  __-.  /\ \   /\ "-.\ \   /\  ___\   /\__  _\ 
#\ \___  \  \ \  __\   \ \ \/\ \ \ \ \  \ \ \-.  \  \ \  __\   \/_/\ \/ 
# \/\_____\  \ \_____\  \ \____-  \ \_\  \ \_\\"\_\  \ \_____\    \ \_\ 
#  \/_____/   \/_____/   \/____/   \/_/   \/_/ \/_/   \/_____/     \/_/ 
#
# By Dr Daniel Buscombe,
# daniel@mardascience.com

###===================================================
# import libraries
from sedinet_models import *
import sys, getopt, json
     
#==============================================================	
if __name__ == '__main__':

    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:c:")
    except getopt.GetoptError:
        print('python sedinet_predict_categorical.py -c configfile.json')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python sedinet_predict_categorical.py -c config_9percentiles.json')
            sys.exit()
        elif opt in ("-c"):
            configfile = arg

    # load the user configs
    with open(os.getcwd()+os.sep+'config'+os.sep+configfile) as f:    
        config = json.load(f)     
         
    ###===================================================
    ## user defined variables: proportion of data to use for training (a.k.a. the "train/test split")
    base    = int(config["base"]) #minimum number of convolutions in a sedinet convolutional block
    csvfile = config["csvfile"] #csvfile containing image names and class values
    res_folder = config["res_folder"] #folder containing csv file and that will contain model outputs
    var = config["var"] # column name in the csv you wish to estimate
    numclass = config["numclass"] # number of classes
    dropout = float(config["dropout"]) 
                            
    ###==================================================
    ID_MAP = dict(zip(np.arange(numclass), [str(k) for k in range(numclass)]))
    csvfile = res_folder+os.sep+csvfile

    estimate_categorical(var, csvfile, base, ID_MAP, res_folder, dropout)


      

