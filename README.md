# SF-GRU

<p align="center">
<img src="sf_gru_diagram.png" alt="sf-gru" align="middle" width="600"/>
</p>

This is the python implementation for paper **A. Rasouli, I. Kotseruba, and J. K. Tsotsos, "Pedestrian Action Anticipation using Contextual Feature Fusion in Stacked RNNs", BMVC 2019.**


### Table of contents
* [Dependencies](#dependencies)
* [Datasets](#datasets)
* [Train](#train)
* [Test](#test)
* [Citation](#citation)
* [Authors](#authors)
* [License](#license)


<a name="dependencies"></a>
## Dependencies
The interface is written and tested using python 3.5. The interface also requires
the following external libraries:<br/>
* tensorflow (tested with 1.9 and 1.14)
* keras (tested with 2.1 and 2.2)
* scikit-learn
* numpy
* pillow

<a name="datasets"></a>
## Datasets
The code is trained and tested with [JAAD](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) and [PIE](http://data.nvision2.eecs.yorku.ca/PIE_dataset/) datasets.


<a name="train"></a>
## Train
A sample training script is provided below:

```
from sf_gru import SFGRU
from pie_data import PIE

data_opts = { 'seq_type': 'crossing',
              'data_split_type': 'random',
               ... }
imdb = PIE(data_path=environ.copy()['PIE_PATH'])

model_opts = {'obs_input_type': ['local_box', 'local_context', 'pose', 'box', 'speed'],
              ...}

method_class = SFGRU()
beh_seq_train = imdb.generate_data_trajectory_sequence('train', **data_opts)
saved_files_path = method_class.train()

```
`from pie_data import PIE` imports the data interface. Download the interface from the corresponding annotation repository.<br/>
`data_opts = { 'seq_type': 'crossing', ... }` specifies the data generation parameters form the dataset. Make sure that `seq_type` is set to `'crossing'`.  Refer to `generate_data_trajectory_sequence()` method in corresponding interface for more information. <br/>
`model_opts = {...}` specifies how the training data should be prepared for the model. Refer to `sf_gru.py:get_data()` for more
information on how to set the parameters. <br/>
`method_class = SFGRU()` instantiates an object of type SFGRU. <br/>
`imdb.generate_data_trajectory_sequence()` generate data sequences from the dataset interface.<br/>
`method_class.train()` trains the model and returns the path to the folder where model and data processing
parameters are saved.

A sample of training code can be found in `sf_gru_script.py`. All the default parameters in the script replicate the conditions in which the model was trained for the paper. Note that since `'random'` split data is used, the model may yield different performance at test time.


<a name="test"></a>
## Test
A sample test script is provided below
```
from sf_gru import SFGRU
from pie_data import PIE

data_opts = { 'seq_type': 'crossing',
              'data_split_type': 'random',
               ... }
imdb = PIE(data_path=environ.copy()['PIE_PATH'])

method_class = SFGRU()
beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
saved_files_path = <path_to_model_folde>
acc , auc, f1, precision, recall = method_class.test(beh_seq_test, saved_files_path)
```
The procedure is similar to train with the exception that there is no need to specify `model_opts` as they Are
saved in the model folder at train time.<br/>
In the case only test is run without training, `saved_files_path` should be specified. It should be the path to the folder where model and training parameters are saved. Note that if test follows train, the path is returned by `train()` function.<br/>
`method_class.test()` test the performance of the model and return the results using the following 5 metrics `acc` (accuracy) , `auc` (area under curve), `f1`, `precision` and `recall`. A sample of training code can be found in `sf_gru_script.py`.


<a name="citation"></a>
# Citation
If you use our dataset, please cite:
```
@inproceedings{rasouli2017they,
  title={Pedestrian Action Anticipation using Contextual Feature Fusion in Stacked RNNs},
  author={Rasouli, Amir and Kotseruba, Iuliia and Tsotsos, John K},
  booktitle={BMVC},
  year={2019}
}

```
<a name="authors"></a>
## Authors

* **[Amir Rasouli](http://www.cse.yorku.ca/~aras/index.html)**
* **[Iuliia Kotseruba](http://www.cse.yorku.ca/~yulia_k/)**

Please send email to aras@eecs.yorku.ca or yulia_k@eecs.yorku.ca if there are any problems with downloading or using the data.

<a name="license"></a>
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
