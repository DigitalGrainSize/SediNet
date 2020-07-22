###> Release v1.3 (July 2020)

## Follow the instructions and then uncomment the code line and run one by one

# in defaults, BATCH_SIZE=6, DO_AUG=True. In config file, scale=false
# python sedinet_predict.py -c config/config_gravel.json -w grain_size_gravel_generic/res/gravel_generic_9prcs_simo_batch6_im768_768_9vars_pinball_aug.hdf5
#
# # in defaults, BATCH_SIZE=7, DO_AUG=True. In config file, scale=true
# python sedinet_predict.py -c config/config_sievedsand_sieve.json -w grain_size_sieved_sands/res_sieve/sievesand_sieve_siso_batch7_im512_512_1vars_pinball_aug_scale.hdf5
#
# # in defaults, BATCH_SIZE=7, DO_AUG=True. In config file, scale=false
# python sedinet_predict.py -c config/config_mattole.json -w mattole/res/mattole_simo_batch7_im512_512_2vars_pinball_aug.hdf5
#
# # in defaults, BATCH_SIZE=8, DO_AUG=True. In config file, scale=true
# python sedinet_predict.py -c config/config_sievedsand_sieve_plus.json -w grain_size_sieved_sands/res_sieve_plus/sievesand_sieve_plus_simo_batch8_im512_512_6vars_pinball_aug_scale.hdf5
#
# # in defaults, BATCH_SIZE=12, DO_AUG=False. In config file, scale=true
# python sedinet_predict.py -c config/config_sand.json -w grain_size_sand_generic/res_9prcs/sand_generic_9prcs_simo_batch12_im768_768_9vars_pinball_noaug_scale.hdf5
#
# # in defaults, BATCH_SIZE=12, DO_AUG=False. In config file, scale=true
# python sedinet_predict.py -c config/config_sand_3prcs.json -w grain_size_sand_generic/res_3prcs/sand_generic_3prcs_simo_batch12_im768_768_3vars_pinball_noaug_scale.hdf5
#
# # in defaults, BATCH_SIZE=[12,13,14], DO_AUG=False. In config file, scale=false
# python sedinet_predict.py -c config/config_9percentiles.json -1 grain_size_global/res/global_9prcs_simo_batch12_im768_768_9vars_pinball_noaug.hdf5 -2 grain_size_global/res/global_9prcs_simo_batch13_im768_768_9vars_pinball_noaug.hdf5 -3 grain_size_global/res/global_9prcs_simo_batch14_im768_768_9vars_pinball_noaug.hdf5
