import bz2
import json

import numpy
import torch
import torch.nn as nn
import torchcde
import torchtuples as tt

from pycox.evaluation import EvalSurv
from pycox.models import CoxCC, PCHazard, LogisticHazard, CoxPH, MTLR, PMF

import c19ode, loss, sdt
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


def run_model(trn_array, val_array, model_type="coxcc", network_type="neuralcde", batch_size=256, max_epochs=10, num_durations = 10,
              interpolation="cubic", verbose=False, device="cpu", backend="torchdiffeq", lr = None, optim = "adam", odesolver="rk4"):
    x_trn_array, y_trn_array = trn_array
    x_val_array, y_val_array = val_array

    x1, x2, x3 = x_trn_array.shape

    if device == "cuda":
        logger.debug(f'Converting default tensor type to FloatTensor')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        use_cuda = False

    logger.debug("creating test tensor")
    _ = torch.Tensor([0, 1, 0, 1])
    logger.debug(f'Adjusting labels, as appropriate')
    out_features = 1
    get_target = lambda df: (df[:, 0], df[:, 1])
    if model_type == 'pchazard':
        label_transform = PCHazard.label_transform(num_durations)
    elif model_type == "logistic" or model_type == "sdt":
        label_transform = LogisticHazard.label_transform(num_durations)
    elif model_type == "mtlr":
        label_transform = MTLR.label_transform(num_durations)

    if model_type == "coxcc":
        y_trn_array = get_target(y_trn_array)
        y_val_array = get_target(y_val_array)
    else:
        y_trn_array = label_transform.fit_transform(*get_target(y_trn_array))
        y_val_array = label_transform.transform(*get_target(y_val_array))
        out_features = label_transform.out_features

    if interpolation == "linear":
        input_channel_count = x3
    else:
        input_channel_count = int(x3 / 4)

    val = tt.tuplefy(x_val_array, y_val_array)

    if network_type == 'neuralcde':
        logger.debug(f'Creating neural CDE net')
        net = c19ode.NeuralCDE(input_channels=input_channel_count, hidden_channels=64, output_channels=out_features,
                               interpolation=interpolation, backend=backend, odesolver=odesolver)
    elif network_type == "vanillamlp":
        logger.debug("Creating vanilla MLP")
        num_nodes = [32, 32]
        batch_norm = True
        dropout = None
        net = tt.practical.MLPVanilla(input_channel_count, num_nodes, out_features, batch_norm, dropout)

    learning_rate_tolerance = 4

    if optim == 'rms':
        optimizer = tt.optim.RMSprop
    elif optim == 'sgd':
        optimizer = tt.optim.SGD
    elif optim == 'adamwr':
        optimizer = tt.optim.AdamWR
    elif optim == 'adamw':
        optimizer = tt.optim.AdamW
    else:
        optimizer = tt.optim.Adam

    if model_type == 'pchazard':
        # model = PCHazard(net, tt.optim.Adam, loss=loss.BrierLoss(), duration_index=label_transform.cuts)
        model = PCHazard(net, optimizer, duration_index=label_transform.cuts)
    elif model_type == 'logistic':
        model = LogisticHazard(net, optimizer, duration_index=label_transform.cuts)
    elif model_type == "mtlr":
        model = MTLR(net, optimizer, duration_index=label_transform.cuts)
    elif model_type == "sdt":
        net = c19ode.NeuralCDE(input_channels=input_channel_count, hidden_channels=192,
                               output_channels=input_channel_count, interpolation=interpolation, backend=backend,
                               use_tanh=False)
        sdt_net = nn.Sequential(net,
                                sdt.SDT(input_dim=input_channel_count, output_dim=out_features, use_cuda=use_cuda))
        model = PMF(sdt_net, optimizer, duration_index=label_transform.cuts)
    else:
        model = CoxPH(net, optimizer)
        learning_rate_tolerance = 2

    logger.debug(f'Size of training data, x = {x_trn_array.shape} y = ({y_trn_array[0].shape}, {y_trn_array[1].shape})')
    logger.debug(
        f'Size of validation data, x = {x_val_array.shape} y = ({y_val_array[0].shape}, {y_val_array[1].shape})')

    if device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if lr is not None:
        best_lr = lr
    else:
        logger.trace(f'Starting search for optimal learning rate...')
        learning_rate_finder = model.lr_finder(x_trn_array, y_trn_array, batch_size, tolerance=learning_rate_tolerance, shuffle=False)
        best_lr = learning_rate_finder.get_best_lr()
        if best_lr > 0.01:
            logger.debug(f"Best learning rate found is {best_lr}, using 0.01")
            best_lr = 0.01
        else:
            logger.debug(f"Best learning rate found is {best_lr}")
    logger.info(f"Setting learning rate to {best_lr}")
    model.optimizer.set_lr(best_lr)

    callbacks = [tt.cb.EarlyStopping()]
    logger.debug("Starting model fit")
    log = model.fit(x_trn_array, y_trn_array, batch_size, max_epochs, callbacks, verbose, val_data=val, shuffle=False)
    logger.info(f"Minimum validation loss: {log.to_pandas().val_loss.min()}")
    logger.info(f"Loss in batches: {model.score_in_batches(val)}")

    return (model, log)


def test_model(model, log, trn_array, tst_array, id_list, output_name, model_type="coxcc"):
    x_trn_array, y_trn_array = trn_array
    x_tst_array, y_tst_array = tst_array
    compute_baseline_hazards = False

    output_name = str(output_name)
    model_info_file_name = output_name + ".json.bz2"

    durations_tst = y_tst_array[:, 0]
    events_tst = y_tst_array[:, 1]

    durations_trn = y_trn_array[:, 0]
    events_trn = y_trn_array[:, 1]

    if model_type == "coxcc":
        compute_baseline_hazards = True
        model.compute_baseline_hazards(x_trn_array, (durations_trn, events_trn), sample=100)

    logger.debug(f"Running test evaluations")
    if model_type == "logistic":
        surv = model.interpolate(10).predict_surv_df(x_tst_array)
    elif model_type == "mtlr":
        surv = model.interpolate(10).predict_surv_df(x_tst_array)
    else:
        surv = model.predict_surv_df(x_tst_array)

    logger.trace(f"durations_tst.shape: {durations_tst.shape}")
    logger.trace(f"events_tst.shape: {events_tst.shape}")
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

    logger.debug(f"Antolini concordance: {eval['antolini']}")
    logger.debug(f"Adjusted Antolini concordance: {eval['antoliniadj']}")
    logger.debug(f"Integrated Brier Score: {eval['ibs']}")
    logger.debug(f"Integrated NBLL: {eval['inbll']}")

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
