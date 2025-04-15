# ENCM 509 - Final Project Group 32
| Student name | UCID |
| -------------|----:|
| Rian Opperman | 30118288 |
|Mohammed Alshoura | 30126200|


## Folder structure

All files relevant to the project can be found in the `./project` directory, they are split as follows:

- `autheticator`: contains the pythons script to find new recodings in the `data_collecter` directory and runs a trained model to predict the user responsible for the recording. The `auth.py` runs using the same environment that trains the models, it needs to be pointed to a model in the `model` directory to work properly.

- `data`: contains the dataset made to train the model one includes `genuine/RO`, `genuine/MA`, `imposter`, and `unused_imposter`, which is made to hold extra imposter recordings that we felt were diluting the number of genuines.

- `data_collector`: contains `LeapMotion_Recorder2.py` and associated files to run the python2 data collection script in order to add data to the dataset, or to run new recordings against the `authenticatior` script.

- `model_builder` contains the notebooks to train, test the models. It also contains some extra visualization like the confusion matrix for example


## Requirements

Based on the instruction found in Lab7:


### For running the model_builder and autheticator scripts run the following commands:
``` bash
conda create -n leapmotionNN python=3.9
conda activate leapmotionNN
conda install -c conda-forge keras tensorflow opencv pandas scikit-learn matplotlib notebook
```

Additionally run the following as a final check:
```pip install -r requirements.txt```

Now you should be able to run the notebooks in `model_builder` and run the `auth.py` script as follows

``` bash
python ./autheticator/auth.py
```

### For running the data_collector script

- install the leapmotion driver as specified in the lab7 manual.
- run the following:
``` bash
conda create -n py2 python=2.7
conda activate py2
conda install numpy pandas
```

Now you should be able to run the `data_colletor` script as follows:
``` bash
python LeapMotion Recorder2 .py
```