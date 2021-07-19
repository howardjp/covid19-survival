import argparse
import os
import pathlib
import sys
import tempfile
import time

import spdlog
import torch
import torchcde

import synthea

infoBanner = "James Howard's ACM Thesis Project"
appName = os.path.basename(__file__)

console = spdlog.ConsoleLogger(appName)
log_level_map = {"trace": spdlog.LogLevel.TRACE, "debug": spdlog.LogLevel.DEBUG, "info": spdlog.LogLevel.INFO,
                 "warn": spdlog.LogLevel.WARN, "err": spdlog.LogLevel.ERR, "critical": spdlog.LogLevel.CRITICAL,
                 "off": spdlog.LogLevel.OFF}


def main():
    """Console script for c19cox."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', metavar="device", type=str, default=None,
                        choices=['cpu', 'cuda', 'opengl', 'opencl'], help='set the processing device for PyTorch')
    parser.add_argument('--data', metavar="data", type=str, default="abbrev.json.bz2", help="dataset to use")
    parser.add_argument('--name', metavar="name", type=str,
                        default=pathlib.Path(tempfile.mktemp(suffix='', prefix='run-', dir=".")),
                        help="use name as output file base")
    parser.add_argument('--type', metavar="type", type=str, default="coxcc", choices=['coxcc', 'coxph', 'pchazard'],
                        help='set the model type')
    parser.add_argument('--interp', metavar="interp", type=str, default="cubic", choices=['cubic', 'linear'],
                        help='set the interpolation type')
    parser.add_argument('--loglevel', metavar="level", type=str, default="info",
                        choices=['trace', 'debug', 'info', 'warn', 'error', 'critical', 'off'],
                        help='set the logging level to the console')
    parser.add_argument('--maxepochs', metavar="count", type=int, default=200,
                        help='set the maximum number of training epochs')
    parser.add_argument('--batchsize', metavar="size", type=int, default=32, help='set the batch size for training')
    parser.add_argument('--seed', metavar="seed", type=int, default=int(time.time()), help='set the random seed')
    parser.add_argument('--verbose', action='store_true', default=False, help='set verbose output from the trainer')
    parser.add_argument('--version', action='store_true', default=False, help='display version information and quit')
    opts = vars(parser.parse_args())

    python_version = sys.version.replace('\n', '')
    version_string = f'Python {python_version}, {torch.__name__} {torch.__version__}, {torchcde.__name__} {torchcde.__version__}'
    console.info(f'{infoBanner}, {appName}')
    console.info(version_string)
    if opts["version"]:
        sys.exit()

    console.set_level(log_level_map[opts["loglevel"]])
    console.debug(f'Console logging level {opts["loglevel"]}')

    if opts["device"] is None:
        if torch.cuda.is_available():
            opts["device"] = "cuda"
        else:
            opts["device"] = "cpu"

    console.info(f'Setting the PyTorch seed to {opts["seed"]}')
    torch.manual_seed(opts["seed"])

    console.info(f'Using model type {opts["type"]}')
    console.info(f'Starting time series classification with maximum epochs of {opts["maxepochs"]}')
    synthea.make_model(console, file_name=opts["data"], output_name=opts["name"], model_type=opts["type"],
                       batch_size=opts["batchsize"], max_epochs=opts["maxepochs"], verbose=opts["verbose"],
                       interpolation=opts["interp"], device=opts["device"])


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
