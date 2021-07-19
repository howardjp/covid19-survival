import bz2
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from globals import logger


def get_data_by_patient(patient_idx, covid19_data):
    observation_trn_id = []
    for patient_id in sorted(patient_idx):
        observation_trn_id.extend([*covid19_data['info'][patient_id].keys()])

    time_index = covid19_data['time_index']
    time_index_df = pd.DataFrame({'time_index': time_index})

    x_array = []
    y_array = []

    for observation_id in sorted(observation_trn_id):
        logger.trace(f'Reading information for observation {observation_id}')
        duration = covid19_data['outcome'][observation_id]['time']
        event = covid19_data['outcome'][observation_id]['outcome']
        y_array.append([duration, event])
        x = pd.DataFrame(covid19_data['data'][observation_id]).fillna(value=np.nan)
        x = pd.merge_ordered(time_index_df, x, left_on='time_index', right_on=0, fill_method=None)
        x = x.drop(['time_index', 0], axis=1)
        x = x.to_numpy()
        x_mask = (~torch.isnan(torch.Tensor(x))).cumsum(dim=0).cpu()
        x = pd.concat([pd.DataFrame(time_index), pd.DataFrame(x), pd.DataFrame(x_mask.numpy())], axis=1).to_numpy()
        x_array.append(x)

    x_array = torch.Tensor(x_array)
    y_array = torch.Tensor(y_array)

    logger.debug(f'x_trn_array size: {x_array.size()}')
    logger.debug(f'y_trn_array size: {y_array.size()}')

    return x_array, y_array


def get_data(file_name, n = None):
    logger.trace(f'Opening file {file_name}')
    with bz2.open(file_name, 'rt', encoding="utf-8") as f:
        covid19_data = json.load(f)

    patient_ids_idx = covid19_data['info'].keys()
    if n is None:
        logger.info(f"Selecting {n} patients at random as the patient dataset for training, validation, and testing")
        patient_ids_idx, _ = train_test_split([*patient_ids_idx], test_size=n)
    patient_trn_idx, patient_tst_idx = train_test_split([*patient_ids_idx], test_size=0.20)
    patient_trn_idx, patient_val_idx = train_test_split(patient_trn_idx, test_size=0.20)

    logger.debug(f'Collecting training observation data...')
    x_trn_array, y_trn_array = get_data_by_patient(patient_trn_idx, covid19_data)

    logger.debug(f'Collecting testing observation data...')
    x_tst_array, y_tst_array = get_data_by_patient(patient_tst_idx, covid19_data)

    logger.debug(f'Collecting validation observation data...')
    x_val_array, y_val_array = get_data_by_patient(patient_val_idx, covid19_data)

    id_list = dict()
    id_list["training_patient_ids"] = patient_trn_idx
    id_list["testing_patient_ids"] = patient_tst_idx
    id_list["validation_patient_ids"] = patient_val_idx

    return (x_trn_array, y_trn_array), (x_tst_array, y_tst_array), (x_val_array, y_val_array), id_list
