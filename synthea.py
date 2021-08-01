import cbor
import pathlib

import torch
from sklearn.model_selection import train_test_split

from globals import logger


def get_data_by_patient(patient_idx, index_data, data_dir):
    i = 0

    x_array = torch.Tensor()
    y_array = torch.Tensor()

    for patient_id in sorted(patient_idx):
        i += 1
        logger.debug(f'Reading information for patient {patient_id}')

        with open(pathlib.Path(data_dir) / (patient_id + ".cbor"), 'rb') as f:
            patient_data = cbor.load(f)

        x_array_ptx = torch.Tensor(patient_data)
        y_array_ptx = torch.Tensor(index_data[patient_id])

        logger.trace(f"x_array_ptx.size = {x_array_ptx.shape}, y_array_ptx.size = {y_array_ptx.shape}")
        x1, _, _ = x_array_ptx.size()
        if x1 > 1:
            x_array = torch.cat((x_array, x_array_ptx), dim=0)
            y_array = torch.cat((y_array, y_array_ptx), dim=0)
        else:
            logger.debug(f"Skipping patient {patient_id}")

    return x_array, y_array


def load_data(data_dir, n=None):
    index_path = pathlib.Path(data_dir) / "index.cbor"
    logger.trace(f'Opening index file {index_path}')
    with open(index_path, 'rb') as f:
        index_data = cbor.load(f)

    patient_ids_idx = list(index_data.keys())
    if n is not None:
        logger.info(f"Selecting {n} patients at random as the patient dataset for training, validation, and testing")
        patient_ids_idx, _ = train_test_split(patient_ids_idx, train_size=n)
    patient_trn_idx, patient_tst_idx = train_test_split(patient_ids_idx, test_size=0.20)
    patient_trn_idx, patient_val_idx = train_test_split(patient_trn_idx, test_size=0.20)

    logger.debug(f'Collecting training observation data...')
    x_trn_array, y_trn_array = get_data_by_patient(patient_trn_idx, index_data, data_dir)

    logger.debug(f'Collecting validation observation data...')
    x_val_array, y_val_array = get_data_by_patient(patient_val_idx, index_data, data_dir)

    logger.debug(f'Collecting testing observation data...')
    x_tst_array, y_tst_array = get_data_by_patient(patient_tst_idx, index_data, data_dir)

    id_list = dict()
    id_list["training_patient_ids"] = patient_trn_idx
    id_list["testing_patient_ids"] = patient_tst_idx
    id_list["validation_patient_ids"] = patient_val_idx

    return (x_trn_array, y_trn_array), (x_val_array, y_val_array), (x_tst_array, y_tst_array), id_list
