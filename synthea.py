import bz2
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import torchcde

from globals import logger


def interp_data(file_name, interpolation="cubic"):
    logger.trace(f'Opening file {file_name}')
    with bz2.open(file_name, 'rt', encoding="utf-8") as f:
        covid19_data = json.load(f)

    interpolation_data = dict()
    interpolation_data['xdata'] = dict()
    interpolation_data['ydata'] = dict()

    time_index = covid19_data['time_index']
    time_index_df = pd.DataFrame({'time_index': time_index})
    patient_idx = covid19_data['info'].keys()
    interpolation_data['patient_list'] = list(patient_idx)

    for patient_id in sorted(patient_idx):
        x_array = []
        y_array = []

        logger.debug(f'Reading information for patient {patient_id}')

        observation_idx = covid19_data['info'][patient_id].keys()
        for observation_id in sorted(observation_idx):
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

        logger.trace(f'x_array size: {x_array.size()}')
        logger.trace(f'y_array size: {y_array.size()}')

        if interpolation == "linear":
            x_array = torchcde.linear_interpolation_coeffs(x_array)
        else:
            x_array = torchcde.natural_cubic_coeffs(x_array)

        logger.trace(f'post-interpolation x_array size: {x_array.size()}')
        logger.trace(f'post-interpolation y_array size: {y_array.size()}')

        interpolation_data['xdata'][patient_id] = x_array.numpy().tolist()
        interpolation_data['ydata'][patient_id] = y_array.numpy().tolist()

    return interpolation_data


def get_data_by_patient(patient_idx, covid19_data):
    i = 0

    for patient_id in sorted(patient_idx):
        i += 1
        logger.trace(f'Reading information for patient {patient_id}')

        x_array_ptx = np.array(covid19_data['xdata'][patient_id])
        y_array_ptx = np.array(covid19_data['ydata'][patient_id])

        logger.trace(f'readback x_array size: {x_array_ptx.shape}')
        logger.trace(f'readback y_array size: {y_array_ptx.shape}')

        if i == 1:
            x_array = x_array_ptx
            y_array = y_array_ptx
        else:
            x_array = np.concatenate([x_array_ptx, x_array], axis = 0)
            y_array = np.concatenate([y_array_ptx, y_array], axis = 0)

    x_array = torch.Tensor(x_array)
    y_array = torch.Tensor(y_array)
    logger.trace(f'combined x_array size: {x_array.shape}')
    logger.trace(f'combined y_array size: {y_array.shape}')

    return x_array, y_array


def load_data(file_name, n=None):
    logger.trace(f'Opening file {file_name}')
    with bz2.open(file_name, 'rt', encoding="utf-8") as f:
        covid19_data = json.load(f)

    patient_ids_idx = covid19_data['patient_list']
    if n is not None:
        logger.info(f"Selecting {n} patients at random as the patient dataset for training, validation, and testing")
        patient_ids_idx, _ = train_test_split(patient_ids_idx, train_size=n)
    patient_trn_idx, patient_tst_idx = train_test_split(patient_ids_idx, test_size=0.20)
    patient_trn_idx, patient_val_idx = train_test_split(patient_trn_idx, test_size=0.20)

    logger.debug(f'Collecting training observation data...')
    x_trn_array, y_trn_array = get_data_by_patient(patient_trn_idx, covid19_data)

    logger.debug(f'Collecting validation observation data...')
    x_val_array, y_val_array = get_data_by_patient(patient_val_idx, covid19_data)

    logger.debug(f'Collecting testing observation data...')
    x_tst_array, y_tst_array = get_data_by_patient(patient_tst_idx, covid19_data)

    id_list = dict()
    id_list["training_patient_ids"] = patient_trn_idx
    id_list["testing_patient_ids"] = patient_tst_idx
    id_list["validation_patient_ids"] = patient_val_idx

    return (x_trn_array, y_trn_array), (x_val_array, y_val_array), (x_tst_array, y_tst_array), id_list
