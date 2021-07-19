import bz2
import json

import numpy as np
import pandas as pd
import torch
import torchcde
import torchtuples as tt

import c19ode

from sklearn.model_selection import train_test_split

from pycox.models import CoxCC, CoxPH, PCHazard


def get_data_by_patient(logger, patient_idx, covid19_data):
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


def get_data(logger, file_name):
    logger.trace(f'Opening file {file_name}')
    with bz2.open(file_name, 'rt', encoding="utf-8") as f:
        covid19_data = json.load(f)

    patient_ids_idx = covid19_data['info'].keys()
    patient_trn_idx, patient_tst_idx = train_test_split([*patient_ids_idx], test_size=0.20)
    patient_trn_idx, patient_val_idx = train_test_split(patient_trn_idx, test_size=0.20)

    logger.debug(f'Collecting training observation data...')
    x_trn_array, y_trn_array = get_data_by_patient(logger, patient_trn_idx, covid19_data)

    logger.debug(f'Collecting testing observation data...')
    x_tst_array, y_tst_array = get_data_by_patient(logger, patient_tst_idx, covid19_data)

    logger.debug(f'Collecting validation observation data...')
    x_val_array, y_val_array = get_data_by_patient(logger, patient_val_idx, covid19_data)

    id_list = dict()
    id_list["training_patient_ids"] = patient_trn_idx
    id_list["testing_patient_ids"] = patient_tst_idx
    id_list["validation_patient_ids"] = patient_val_idx

    return (x_trn_array, y_trn_array), (x_tst_array, y_tst_array), (x_val_array, y_val_array), id_list


def make_model(logger, file_name, output_name, model_type="coxcc", batch_size=256, max_epochs=10,
               interpolation="cubic", verbose=False, device="cuda"):
    if device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    logger.info(f'Using device {device} for PyTorch processing')

    trn_array, tst_array, val_array, id_list = get_data(logger, file_name)
    x_trn_array, y_trn_array = trn_array
    x_tst_array, y_tst_array = tst_array
    x_val_array, y_val_array = val_array
    x1, x2, x3 = x_trn_array.size()

    logger.debug(f'Starting initial interpolation using method {interpolation}')
    if interpolation == "linear":
        logger.info(f'Starting interpolation on training dataset')
        x_trn_array = torchcde.linear_interpolation_coeffs(x_trn_array)
        logger.info(f'Starting interpolation on validation dataset')
        x_val_array = torchcde.linear_interpolation_coeffs(x_val_array)
        logger.info(f'Starting interpolation on testing dataset')
        x_tst_array = torchcde.linear_interpolation_coeffs(x_tst_array)
    else:
        logger.info(f'Starting interpolation on training dataset')
        x_trn_array = torchcde.natural_cubic_coeffs(x_trn_array)
        logger.info(f'Starting interpolation on validation dataset')
        x_val_array = torchcde.natural_cubic_coeffs(x_val_array)
        logger.info(f'Starting interpolation on testing dataset')
        x_tst_array = torchcde.natural_cubic_coeffs(x_tst_array)
    logger.debug(f'Completed initial interpolation')

    out_features = 1
    compute_baseline_hazards = False
    get_target = lambda df: (df[:, 0], df[:, 1])
    if model_type == 'pchazard':
        num_durations = 10
        labtrans = PCHazard.label_transform(num_durations)
        y_trn_array = labtrans.fit_transform(*get_target(y_trn_array))
        y_val_array = labtrans.transform(*get_target(y_val_array))
        out_features = labtrans.out_features
    else:
        y_trn_array = get_target(y_trn_array)
        y_val_array = get_target(y_val_array)
        compute_baseline_hazards = True

    val = tt.tuplefy(x_val_array, y_val_array)
    val = val.repeat(10).cat()

    # We don't need to transform the test labels
    durations_test, events_test = get_target(y_tst_array)

    logger.trace(f'durations_test.shape = {durations_test.shape}')
    logger.trace(f'events_test.shape = {events_test.shape}')

    net = c19ode.NeuralCDE(input_channels=x3, hidden_channels=64, output_channels=out_features,
                           interpolation=interpolation)

    if model_type == 'pchazard':
        model = PCHazard(net, tt.optim.Adam, duration_index=labtrans.cuts)
        lr_tolerance = 8
    elif model_type == 'coxph':
        model = CoxPH(net, tt.optim.Adam)
        lr_tolerance = 10
    else:
        model = CoxCC(net, tt.optim.Adam)
        lr_tolerance = 2

    logger.debug(f'Starting search for optimal learning rate...')
    lr_finder = model.lr_finder(x_trn_array, y_trn_array, batch_size, tolerance=lr_tolerance)
    best_lr = lr_finder.get_best_lr()
    if best_lr > 0.01:
        logger.info(f"Best learning rate found is {best_lr}, using 0.01")
        best_lr = 0.01
    else:
        logger.info(f"Best learning rate found is {best_lr}")
    model.optimizer.set_lr(best_lr)

    callbacks = [tt.cb.EarlyStopping()]
    logger.debug("Starting model fit")
    log = model.fit(x_trn_array, y_trn_array, batch_size, max_epochs, callbacks, verbose, val_data=val)
    logger.info(f"Minimum validation loss: {log.to_pandas().val_loss.min()}")
    logger.info(f"Loss in batches: {model.score_in_batches(val)}")
    output_name = str(output_name)
    model_info_file_name = output_name + ".json.bz2"

    if compute_baseline_hazards:
        model.compute_baseline_hazards()
    logger.debug(f"Writing model information to {model_info_file_name}")
    model_info = dict()
    model_info["method"] = model_type
    model_info["maximum_epochs"] = max_epochs
    model_info["interpolation_type"] = interpolation
    model_info["patient_ids"] = id_list
    model_info["training_history"] = json.loads(log.to_pandas().to_json())
    model_info["predictions"] = json.loads(model.predict_surv_df(x_tst_array).to_json())
    model_info["true_values"] = json.dumps(y_tst_array.tolist())
    with bz2.open(model_info_file_name, 'wt', encoding="utf-8") as f:
        json.dump(model_info, f)

    if compute_baseline_hazards:
        logger.debug(f"Writing model to {output_name}.pt and {output_name}_blh.pickle")
    else:
        logger.debug(f"Writing model to {output_name}.pt")
    model.save_net(output_name)

    logger.info("Done.")
    return
