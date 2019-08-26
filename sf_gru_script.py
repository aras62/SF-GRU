from os import environ

from sf_gru import SFGRU
from pie_data import PIE
from jaad_data import JAAD


data_opts = {'fstride': 1, # Sampling resolution from sequences
             'sample_type': 'all',  # 'all', 'beh'. Only for JAAD dataset
             'subset': 'default',  # 'default', 'high_visibility', 'all'. Only for JAAD dataset
             'height_rng': [0, float('inf')],  # Height restriction on pedestrians
             'squarify_ratio': 0,  # width/height ration between [0,1]. 0 keeps the original ratio.
             'data_split_type': 'random',  # 'kfold', 'random', 'default'
             'seq_type': 'crossing',  # Sequence type. For this code only use 'crossing'
             'min_track_size': 75,  # Minimum size below which sequences are discarded
             'random_params': {'ratios': None,
                               'val_data': True,
                               'regen_data': True},
             'kfold_params': {'num_folds': 5, 'fold': 1}}


# Specify the type of dataset. Note that JAAD does not have vehicle speed information.
# Instead vehicle actions will be used
dataset = 'pie'
if dataset == 'pie':
    imdb = PIE(data_path=environ.copy()['PIE_PATH'])
elif dataset == 'jaad':
    imdb = JAAD(data_path=environ.copy()['JAAD_PATH'])

#NOTE: Poses should be in 'data/data/<dataset>/poses'
train_test = 1  # 0 train, 1 train-test, 2 test

# Path to the folder where the model are saved. ONly use if only test function is used
saved_files_path = ''

#  For more information see sf_gru.py:get_data()
model_opts = {'obs_input_type': ['local_box', 'local_context', 'pose', 'box', 'speed'],
              'enlarge_ratio': 1.5,
              'pred_target_type': ['crossing'],
              'obs_length': 15,
              'time_to_event': 60,
              'dataset': dataset,
              'normalize_boxes': True}
method_class = SFGRU(num_hidden_units=256,
                     global_pooling='avg',
                     regularizer_val=0.0001)



if train_test < 2:
    beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)

    saved_files_path = method_class.train(beh_seq_train,
                                          epochs=60,
                                          batch_size=32,
                                          model_opts=model_opts,
                                          lr=0.000005)
if train_test > 0:
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
    acc , auc, f1, precision, recall = method_class.test(beh_seq_test,
                                                         saved_files_path)
