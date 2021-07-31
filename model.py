import bz2
import json

import torchcde
import torchtuples as tt
from pycox.models import CoxCC, CoxPH, PCHazard

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
              interpolation="cubic", verbose=False):
    x_trn_array, y_trn_array = trn_array
    x_val_array, y_val_array = val_array

    x1, x2, x3 = x_trn_array.shape

    out_features = 1
    get_target = lambda df: (df[:, 0], df[:, 1])
    if model_type == 'pchazard':
        num_durations = 10
        label_transform = PCHazard.label_transform(num_durations)
        y_trn_array = label_transform.fit_transform(*get_target(y_trn_array))
        y_val_array = label_transform.transform(*get_target(y_val_array))
        out_features = label_transform.out_features
    else:
        y_trn_array = get_target(y_trn_array)
        y_val_array = get_target(y_val_array)

    val = tt.tuplefy(x_val_array, y_val_array)
    val = val.repeat(10).cat()

    net = c19ode.NeuralCDE(input_channels=int(x3/4), hidden_channels=64, output_channels=out_features, interpolation=interpolation)

    if model_type == 'pchazard':
        model = PCHazard(net, tt.optim.Adam, duration_index=label_transform.cuts)
        learning_rate_tolerance = 8
    elif model_type == 'coxph':
        model = CoxPH(net, tt.optim.Adam)
        learning_rate_tolerance = 10
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

    if model_type != "pchazard":
        compute_baseline_hazards = True
        model.compute_baseline_hazards()
    logger.debug(f"Writing model information to {model_info_file_name}")
    model_info = dict()
    model_info["method"] = model_type
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
