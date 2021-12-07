import argparse
import bz2
import json
import pathlib
import sys
import tempfile
import time

import torch
import torchcde

import model
import synthea
from globals import logger, log_level_map, app_name, info_banner


def main():
    """Console script for c19cox."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', metavar="device", type=str, default=None,
                        choices=['cpu', 'cuda', 'opengl', 'opencl'], help='set the processing device for PyTorch')
    parser.add_argument('--data', metavar="data", type=str, default="abbrev.json.bz2", help="dataset to use")
    parser.add_argument('--name', metavar="name", type=str,
                        default=pathlib.Path(tempfile.mktemp(suffix='', prefix='run-', dir=".")),
                        help="use name as output file base")
    parser.add_argument('--n', metavar="count", type=int, default=None, help='use count random patients for modeling, defaults to all available')
    parser.add_argument('--type', metavar="type", type=str, default="coxcc", choices=['coxcc', 'pchazard', 'logistic', "mtlr", "sdt"],
                        help='set the model type')
    parser.add_argument('--solver', metavar="type", type=str, default="torchdiffeq", choices=['torchdiffeq', 'torchsde'],
                        help='set the model type')
    parser.add_argument('--optim', metavar="optim", type=str, default="adam", choices=['adam', "adamw", "adamwr", "rms", "sgd"], help='set the optimizer')
    parser.add_argument('--odesolver', metavar="odesolver", type=str, default="rk4", choices=["dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun", "euler", "midpoint", "rk4", "explicit_adams", "implicit_adams", "fixed_adams", "scipy_solver"], help='set the ODE solver')
    parser.add_argument('--interp', metavar="interp", type=str, default="cubic", choices=['cubic', 'linear'],
                        help='set the interpolation type')
    parser.add_argument('--loglevel', metavar="level", type=str, default="info",
                        choices=['trace', 'debug', 'info', 'warn', 'error', 'critical', 'off'],
                        help='set the logging level to the console')
    parser.add_argument('--maxepochs', metavar="count", type=int, default=200,
                        help='set the maximum number of training epochs')
    parser.add_argument('--batchsize', metavar="size", type=int, default=32, help='set the batch size for training')
    parser.add_argument('--seed', metavar="seed", type=int, default=int(time.time()), help='set the random seed')
    parser.add_argument('--durations', metavar="durations", type=int, default=10, help='set the potential durations count')
    parser.add_argument('--lr', metavar="rate", type=float, default=None, help='set the learning rate')
    parser.add_argument('--dataprep', action='store_true', default=False, help='run primary interpolation and save results')
    parser.add_argument('--verbose', action='store_true', default=False, help='set verbose output from the trainer')
    parser.add_argument('--version', action='store_true', default=False, help='display version information and quit')
    opts = vars(parser.parse_args())

    python_version = sys.version.replace('\n', '')
    version_string = f'Python {python_version}, {torch.__name__} {torch.__version__}, {torchcde.__name__} {torchcde.__version__}'
    logger.info(f'{info_banner}, {app_name}')
    logger.info(version_string)
    if opts["version"]:
        sys.exit()

    logger.set_level(log_level_map[opts["loglevel"]])
    logger.debug(f'Console logging level {opts["loglevel"]}')

    if opts["dataprep"]:
        logger.trace(f"Reading data and starting interpolation")
        interp_data = synthea.interp_data(opts['data'], interpolation=opts["interp"])
        model_info_file_name = opts["name"] + ".json.bz2"
        logger.info(f"Writing interpolated data to {model_info_file_name}")
        with bz2.open(model_info_file_name, 'wt', encoding="utf-8") as f:
            json.dump(interp_data, f)
        sys.exit()

    if opts["device"] is None:
        if torch.cuda.is_available():
            opts["device"] = "cuda"
        else:
            opts["device"] = "cpu"

    torch.set_default_tensor_type('torch.FloatTensor')
    logger.info(f'Using device {opts["device"]} for PyTorch processing')

    logger.info(f'Setting the PyTorch seed to {opts["seed"]}')
    torch.manual_seed(opts["seed"])

    trn_array, val_array, tst_array, id_list = synthea.load_data(opts['data'], opts['n'])
    trn_array = (trn_array[0].cpu().numpy(), trn_array[1].cpu().numpy())
    val_array = (val_array[0].cpu().numpy(), val_array[1].cpu().numpy())
    tst_array = (tst_array[0].cpu().numpy(), tst_array[1].cpu().numpy())

    logger.info(f'Using model type {opts["type"]}')
    logger.info(f'Using solver type {opts["solver"]}')
    logger.info(f'Starting time series classification with maximum epochs of {opts["maxepochs"]}')

    (cde_model, log) = model.run_model(trn_array, val_array, model_type=opts["type"], num_durations=opts["durations"],
                    batch_size=opts["batchsize"], max_epochs=opts["maxepochs"], verbose=opts["verbose"],
                    interpolation=opts["interp"], device = opts["device"], backend=opts["solver"], lr = opts["lr"],
                                       optim=opts["optim"], odesolver=opts["odesolver"])

    model.test_model(cde_model, log, trn_array, tst_array, id_list, output_name=opts["name"], model_type=opts["type"])

    logger.info("Done.")

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
