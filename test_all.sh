##> Release v1.3 (July 2020)

## train / all cases

### continuous
python sedinet_train.py  -c config/config_9percentiles.json
python sedinet_train.py -c config/config_gravel.json
python sedinet_train.py -c config/config_sand.json
python sedinet_train.py -c config/config_sievedsand_sieve.json
python sedinet_train.py -c config/config_sievedsand_sieve_plus.json
python sedinet_train.py -c config/config_sand_3prcs.json
python sedinet_train.py  -c config/config_mattole.json

### categorical
python sedinet_train.py -c config/config_pop.json
python sedinet_train.py -c config/config_shape.json


### test / predict on all samples

### continuous
python sedinet_predict.py -c config/config_9percentiles_predict.json -w grain_size_global/res/grey/global_9prcs_simo_batch8_P10_P16_P25_P5_P50_P75_P84_P90_P95__checkpoint.hdf5
python sedinet_predict.py -c config/config_gravel_predict.json -w grain_size_gravel_generic/res/grey/gravel_generic_9prcs_simo_batch8_P10_P16_P25_P5_P50_P75_P84_P90_P95__checkpoint.hdf5
python sedinet_predict.py -c config/config_sand_predict.json -w grain_size_sand_generic/res/grey/sand_generic_9prcs_simo_batch8_P10_P16_P25_P5_P50_P75_P84_P90_P95__checkpoint.hdf5
python sedinet_predict.py -c config/config_sievedsand_sieve_predict.json -w grain_size_sieved_sands/res/grey/sievesand_sieve_siso_batch8_sieve__checkpoint.hdf5
python sedinet_predict.py -c config/config_sievedsand_sieve_plus_predict.json -w grain_size_sieved_sands/res/grey/sievesand_sieve_plus_simo_batch8_P16_P25_P50_P75_P84_sieve__checkpoint.hdf5
python sedinet_predict.py -c config/config_sand_3prcs_predict.json -w grain_size_sand_generic/res/grey/sand_generic_3prcs_simo_batch8_P10_P50_P90__checkpoint.hdf5
python sedinet_predict.py -c config/config_mattole_predict.json -w mattole/res/grey/mattole_simo_batch8_p10_p16_p25_p50_p75_p84_p90__checkpoint.hdf5

### categorical
python sedinet_predict.py -c config/config_pop_predict.json -w grain_population/res/color/pop_model_checkpoint.hdf5
python sedinet_predict.py -c config/config_shape_predict.json -w grain_shape/res/color/shape_model_checkpoint.hdf5


### predict on "unseen" samples

python sedinet_predict.py -c config/config_9percentiles_predict_samples.json -w grain_size_global/res/grey/global_9prcs_simo_batch8_P10_P16_P25_P5_P50_P75_P84_P90_P95__checkpoint.hdf5
