import os
import time
from typing import Optional

import pandas as pd
import numpy as np

#import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join, dirname, abspath
from tensorflow.keras.metrics import Precision, Recall
from warnings import simplefilter
from tensorflow.keras.models import load_model
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# the length of records (if shorter, we need to add some zero rows)
NUMBER_TIMESTEPS = 150
# the number of features (from the data)
NUMBER_FEATURES = 203
# the number of classes/gestures
NUMBER_OUTPUTS = 3
# you can encode more than 1 but for this example we have binary output (circle/swipe)

# choose data we need
columns = ['handPalmPosition_X','handPalmPosition_Y','handPalmPosition_Z',
          'pitch', 'roll', 'yaw', 'GenuineRO', 'GenuineMA', 'imposter',
          'wristPosition_X', 'wristPosition_Y','wristPosition_Z',
          'elbowPosition_X', 'elbowPosition_Y', 'elbowPosition_Z', 'handType']

finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    
for finger in finger_names:
    columns.append(finger + 'Length')
    columns.append(finger + 'Width')

for finger in finger_names:
    for bone in bone_names:
        columns.append(finger + bone + 'Start_X')
        columns.append(finger + bone + 'Start_Y')
        columns.append(finger + bone + 'Start_Z')
        columns.append(finger + bone + 'End_X')
        columns.append(finger + bone + 'End_Y')
        columns.append(finger + bone + 'End_Z')
        columns.append(finger + bone + 'Direction_X') 
        columns.append(finger + bone + 'Direction_Y') 
        columns.append(finger + bone + 'Direction_Z') 

columns_to_expand = [
    'handPalmPosition', 'wristPosition', 'elbowPosition',
    'ThumbMetacarpalStart', 'ThumbMetacarpalEnd', 'ThumbMetacarpalDirection',
    'ThumbProximalStart', 'ThumbProximalEnd', 'ThumbProximalDirection',
    'ThumbIntermediateStart', 'ThumbIntermediateEnd', 'ThumbIntermediateDirection',
    'ThumbDistalStart', 'ThumbDistalEnd', 'ThumbDistalDirection',
    'IndexMetacarpalStart', 'IndexMetacarpalEnd', 'IndexMetacarpalDirection',
    'IndexProximalStart', 'IndexProximalEnd', 'IndexProximalDirection',
    'IndexIntermediateStart', 'IndexIntermediateEnd', 'IndexIntermediateDirection',
    'IndexDistalStart', 'IndexDistalEnd', 'IndexDistalDirection',
    'MiddleMetacarpalStart', 'MiddleMetacarpalEnd', 'MiddleMetacarpalDirection',
    'MiddleProximalStart', 'MiddleProximalEnd', 'MiddleProximalDirection',
    'MiddleIntermediateStart', 'MiddleIntermediateEnd', 'MiddleIntermediateDirection',
    'MiddleDistalStart', 'MiddleDistalEnd', 'MiddleDistalDirection',
    'RingMetacarpalStart', 'RingMetacarpalEnd', 'RingMetacarpalDirection',
    'RingProximalStart', 'RingProximalEnd', 'RingProximalDirection',
    'RingIntermediateStart', 'RingIntermediateEnd', 'RingIntermediateDirection',
    'RingDistalStart', 'RingDistalEnd', 'RingDistalDirection',
    'PinkyMetacarpalStart', 'PinkyMetacarpalEnd', 'PinkyMetacarpalDirection',
    'PinkyProximalStart', 'PinkyProximalEnd', 'PinkyProximalDirection',
    'PinkyIntermediateStart', 'PinkyIntermediateEnd', 'PinkyIntermediateDirection',
    'PinkyDistalStart', 'PinkyDistalEnd', 'PinkyDistalDirection'
]

class_names = ['RO', 'MA', 'Imposter']

def split_and_expand(df, column_name_prefix):
    columns = [f'{column_name_prefix}_{i}' for i in ['X', 'Y', 'Z']]
    df[columns] = df[column_name_prefix].str[1:-1].str.split(',', expand=True)
    df.drop(columns=[column_name_prefix], inplace=True)

def process_data(datafiles, use_prerecorded=False, columns_to_expand=None, path='./'):
    x = []
    num = 1  # Counter for saving files (if needed)
    
    for sample in datafiles:
        relative_path = path + '/' + f'/{sample}' if use_prerecorded else path + '/' + f'/{sample}'
        tmp = pd.read_csv(relative_path, usecols=columns if use_prerecorded else None)
        
        # trying something
        # Simple handType mapping: strings to integers
        if 'handType' in tmp.columns:
            tmp['handType'] = tmp['handType'].replace({
                'Right hand': 1,
                'Left hand': 2
            }).fillna(0).astype(float)

        
        if not use_prerecorded and columns_to_expand:
            for col in columns_to_expand:
                split_and_expand(tmp, col)
        
        tmp.fillna(0, inplace=True)
        tmp.replace('', 0, inplace=True)
        # tmp = tmp.astype(float)
        
        # print(f'{relative_path}\nsize raw = {tmp.shape}')
        tmp = tmp.reindex(range(NUMBER_TIMESTEPS), fill_value=0) if tmp.shape[0] < NUMBER_TIMESTEPS else tmp.head(NUMBER_TIMESTEPS)
        # print(f'size normalized = {tmp.shape}')
        
        tmp_x = tmp.drop(columns=['GenuineRO', 'GenuineMA', 'imposter'], errors='ignore')
        tmp_x.drop(columns=[col for col in tmp_x.columns if col not in columns], inplace=True)
        
        if use_prerecorded:
            path_str = f'save/{num}.csv'
            # tmp_x.to_csv(path_str)
            num += 1
        
        x.append(tmp_x)
    return x


def watch(directory: str = "../data_collector", filename: str = "data1.csv", interval: int = 1) -> Optional[str]:
    while True:
        files = os.scandir(directory)
        for entry in files:
            # print(f"{entry.name}: is_file? {entry.is_file()}")
            if entry.is_file() and entry.name == filename:
                data = None
                data = process_data([filename], columns_to_expand=columns_to_expand, path=directory)
                os.remove(os.fsdecode(entry.path))
                return data
        time.sleep(interval)
        

def main():
    tf.get_logger().setLevel('ERROR')
    model = load_model("../model/gesture_model_bidirectional.h5")
    while True:
        data = watch()
        if data is None:
            continue

        print("Found a new file!")
        x = np.array(data, dtype=float).reshape((1, NUMBER_TIMESTEPS, NUMBER_FEATURES))
        y = model.predict(x, verbose=0)
        y = np.argmax(y, axis=1)
        print(f"Newest recording is {class_names[y[0]]}")


if __name__ == "__main__":
    main()
