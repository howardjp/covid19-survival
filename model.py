import bz2
import json

import torch
import torchcde
import torchtuples as tt
from pycox.models import CoxCC, PCHazard, LogisticHazard
import torch

import c19ode
from globals import logger


def make_interp(trn_array, val_array, tst_array, interpolation="cubic"):
    x_trn_array, y_trn_array = trn_array
    x_tst_array, y_tst_array = tst_array
    x_val_array, y_val_array = val_array

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

    return (x_trn_array, y_trn_array), (x_val_array, y_val_array), (x_tst_array, y_tst_array),


def run_model(trn_array, val_array, model_type="coxcc", batch_size=256, max_epochs=10,
              interpolation="cubic", verbose=False, device = "cpu"):
    x_trn_array, y_trn_array = trn_array
    x_val_array, y_val_array = val_array

    x1, x2, x3 = x_trn_array.shape

    if device == "cuda":
        logger.debug(f'Converting default tensor type to FloatTensor')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    logger.debug(f'Adjusting labels, as appropriate')
    out_features = 1
    get_target = lambda df: (df[:, 0], df[:, 1])
    num_durations = 10
    if model_type == 'pchazard':
        label_transform = PCHazard.label_transform(num_durations)
        y_trn_array = label_transform.fit_transform(*get_target(y_trn_array))
        y_val_array = label_transform.transform(*get_target(y_val_array))
        out_features = label_transform.out_features
    elif model_type == "logistic":
        label_transform = LogisticHazard.label_transform(num_durations)
        y_trn_array = label_transform.fit_transform(*get_target(y_trn_array))
        y_val_array = label_transform.transform(*get_target(y_val_array))
        out_features = label_transform.out_features
    else:
        y_trn_array = get_target(y_trn_array)
        y_val_array = get_target(y_val_array)

    if interpolation == "linear":
        input_channel_count = x3
    else:
        input_channel_count = int(x3/4)

    if opts == "cuda":
        logger.debug(f'Moving data to the GPU')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        x_trn_array = x_trn_array.cuda()
        y_trn_array = y_trn_array.cuda()
        x_val_array = x_val_array.cuda()
        y_val_array = y_val_array.cuda()

    val = tt.tuplefy(x_val_array, y_val_array)
    val = val.repeat(10).cat()

    net = c19ode.NeuralCDE(input_channels=input_channel_count, hidden_channels=64, output_channels=out_features, interpolation=interpolation)

    if model_type == 'pchazard':
        model = PCHazard(net, tt.optim.Adam, duration_index=label_transform.cuts)
        learning_rate_tolerance = 8
    elif model_type == 'logistic':
        model = LogisticHazard(net, tt.optim.Adam, duration_index=label_transform.cuts)
        learning_rate_tolerance = 8
    else:
        model = CoxCC(net, tt.optim.Adam)
        learning_rate_tolerance = 2

    logger.debug(f'Size of training data, x = {x_trn_array.shape} y = ({y_trn_array[0].shape}, {y_trn_array[1].shape})')
    logger.debug(f'Size of validation data, x = {x_val_array.shape} y = ({y_val_array[0].shape}, {y_val_array[1].shape})')

    logger.debug(f'Starting search for optimal learning rate...')
    learning_rate_finder = model.lr_finder(x_trn_array, y_trn_array, batch_size, tolerance=learning_rate_tolerance, shuffle=False)
    best_lr = learning_rate_finder.get_best_lr()
    if best_lr > 0.01:
        logger.info(f"Best learning rate found is {best_lr}, using 0.01")
        best_lr = 0.01
    else:
        logger.info(f"Best learning rate found is {best_lr}")
    model.optimizer.set_lr(best_lr)

    callbacks = [tt.cb.EarlyStopping()]
    logger.debug("Starting model fit")
    log = model.fit(x_trn_array, y_trn_array, batch_size, max_epochs, callbacks, verbose, val_data=val, shuffle=False)
    logger.info(f"Minimum validation loss: {log.to_pandas().val_loss.min()}")
    logger.info(f"Loss in batches: {model.score_in_batches(val)}")

    return (model, log)


def test_model(model, log, tst_array, id_list, output_name, model_type="coxcc"):
    x_tst_array, y_tst_array = tst_array
    compute_baseline_hazards = False

    output_name = str(output_name)
    model_info_file_name = output_name + ".json.bz2"

    durations_tst = y_tst[:,0].numpy()
    events_tst = y_tst[:,1].numpy()

    if model_type == "coxcc":
        compute_baseline_hazards = True
        model.compute_baseline_hazards()

    logger.debug(f"Running test evaluations")
    if model_type == "logistic":
        surv = model.interpolate(10).predict_surv_df(x_tst_array)
    else:
        surv = model.predict_surv_df(x_tst_array)

    logger.debug(f"Collecting evaluations information")
    time_grid = numpy.linspace(durations_tst.min(), durations_tst.max(), 100)
    ev = EvalSurv(surv, durations_tst, events_tst, censor_surv='km')

    eval = dict()
    eval['antolini'] = ev.concordance_td('antolini')
    eval['antoliniadj'] = ev.concordance_td('adj_antolini')
    eval['ibs'] = ev.integrated_brier_score(time_grid)
    eval['inbll'] = ev.integrated_nbll(time_grid)
    eval['bs'] = ev.brier_score(time_grid).to_numpy().tolist()
    eval['nbll'] = ev.nbll(time_grid).to_numpy().tolist()

    logger.info(f"Writing model information to {model_info_file_name}")
    model_info = dict()
    model_info["method"] = model_type
    model_info["patient_ids"] = id_list
    model_info["training_history"] = json.loads(log.to_pandas().to_json())
    model_info["predictions"] = json.loads(surv.to_json())
    model_info["true_values"] = json.dumps(y_tst_array.tolist())
    model_info['eval'] = eval

    with bz2.open(model_info_file_name, 'wt', encoding="utf-8") as f:
        json.dump(model_info, f)

    if compute_baseline_hazards:
        logger.info(f"Writing model to {output_name}.pt and {output_name}_blh.pickle")
    else:
        output_name = output_name + ".model"
        logger.info(f"Writing model to {output_name}")
    model.save_net(output_name)
