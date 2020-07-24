import os
from sf_gru import SFGRU
from pie_data import PIE

data_opts ={'fstride': 1,
            'subset': 'default',
            'data_split_type': 'random',  # kfold, random, default
            'seq_type': 'crossing',
            'min_track_size': 75} ## for obs length of 15 frames + 60 frames tte. This should be adjusted for different setup
imdb = PIE(data_path=os.environ.copy()['PIE_PATH']) # change with the path to the dataset

model_opts = {'obs_input_type': ['local_box', 'local_context', 'pose', 'box', 'speed'],
              'enlarge_ratio': 1.5,
              'pred_target_type': ['crossing'],
              'obs_length': 15,  # Determines min track size
              'time_to_event': 60, # Determines min track size
              'dataset': 'pie',
              'normalize_boxes': True}

method_class = SFGRU()
beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
saved_files_path = method_class.train(beh_seq_train, model_opts=model_opts)
beh_seq_train = imdb.generate_data_trajectory_sequence('test', **data_opts)
acc, auc, f1, precision, recall = method_class.test(beh_seq_train, saved_files_path)
