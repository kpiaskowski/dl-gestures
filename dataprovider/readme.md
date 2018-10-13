###Dataproviders
Scripts located in this folder contain fast data providers for neural networks. Each dataset is 
associated with one specific provider. Currently, 2 datasets are supported:
- Jester
- Chalearn Isolated.

#### TFRecords and data generation
To increase the speed of data loading, a TensorFlow TFRecord format is used to store data. 
Unfortunately, this the process of converting regular data to TFRecords must be done offline,
before the actual training. So, after the dataset is downloaded, a script associated with this 
dataset must be used to generate TFRecords (regular data can be deleted after the generation is finished).

The generation scripts assume the fixed directory structure (see below).
They can be run from command line:

- Jester: `python jester.py --data_dir=... --csv_dir=... --tfrecords_path=...`
    - --data_dir - path to the data where parent folder, where Jester folders with png images are stored
    - --csv_dir - path to the data where parent folder, where Jester csv files are stored
    - --tfrecords_path - a path where TFRecords will be stored
    - Jester dataset should consist of two main directories: `data` when all sequences are stored (each one in its own subfolder) and `csv`, where
      csv files describing dataset are stored. 
      
- Chalearn Isolated: `python chalearn_isolated.py --root_dir=... --tfrecords_path=...`
    - --root_dir - path to directory where "train" folder and "train_list.txt" file are stored
    - --tfrecords_path - a path where TFRecords will be stored
    - Chalearn Isolated dataset contains two main items: `train` folder, containing data (unfortunately, 
      no val and test data with labels is provided for Chalearn) and `train_list.txt` file, describing labels. 
      
#### How to use dataproviders
The example usage of dataproviders is shown in the `main` function of each generation script (the code in __main__ function is commented - just uncomment it). The scipts can be run either from IDE or
from command line (in order to do so, just run them without specyfing any argument). Please note, that `data_dir` variable must be correctly specified.
For for training, the procedure is identical. 