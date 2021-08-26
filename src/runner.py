"""
Script to for training and evaluation.
"""
import argparse
import importlib
import json
import logging
import os
import random
import shutil
from argparse import Namespace
import horovod.torch as hvd
import numpy as np
import torch
from filelock import FileLock

random.seed(1334)
np.random.seed(1334)

logging.basicConfig(level=logging.INFO)


def get_args_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="actions")

    parser_train = subparsers.add_parser("train", add_help=False)
    parser_train.add_argument("--experiment_dpath", type=str, required=False, default=None,
                              help="Directory where training artifacts will be stored.")
    parser_train.add_argument("--dataset", type=str, required=True,
                              help="Fully qualified package name of the Dataset class.")
    parser_train.add_argument("--model", type=str, required=True,
                              help="Fully qualified package name of the Model class.")
    parser_train.add_argument("--notes", type=str, required=False, default="NA", help="Notes for experiment")
    parser_train.add_argument("--epochs", type=int, required=True, default=None)
    parser_train.add_argument("--use_horovod", action='store_true', default=False)
    parser_train.set_defaults(which="train")

    parser_classify_users = subparsers.add_parser("use_model", add_help=False)
    parser_classify_users.add_argument("--experiment_dpath", type=str, required=False, default=None,
                                       help="Directory where training artifacts will be stored.")
    parser_classify_users.add_argument("--method", type=str, required=False, default=None,
                                       help="Directory where training artifacts will be stored.")
    parser_classify_users.set_defaults(which="use_model")

    return parser


def get_model_class(cmd, copy=True):
    """
    Get the model class.
    :param cmd: Namespace object with parsed command line arguments.
    :return: Model class
    """
    model_module_str = ".".join(cmd.model.split(".")[:-1])
    model_class_str = cmd.model.split(".")[-1]

    model_module = importlib.import_module(model_module_str)
    if copy:
        module_fpath = model_module.__file__
        shutil.copy(module_fpath, cmd.experiment_dpath)
    model_class = getattr(model_module, model_class_str)

    return model_class


def get_dataset_class(cmd, copy=True):
    """
    Get the Dataset class.
    :param cmd: Namespace object with parsed command line arguments.
    :return: Dataset class
    """
    dataset_module_str = ".".join(cmd.dataset.split(".")[:-1])
    dataset_class_str = cmd.dataset.split(".")[-1]

    dataset_module = importlib.import_module(dataset_module_str)

    if copy:
        dataset_module_fpath = dataset_module.__file__
        shutil.copy(dataset_module_fpath, cmd.experiment_dpath)
    dataset_class = getattr(dataset_module, dataset_class_str)

    return dataset_class


if __name__ == '__main__':
    parser = get_args_parser()
    cmd, other_args = parser.parse_known_args()

    if cmd.which == "train":
        dataset_class = get_dataset_class(cmd)
        if cmd.use_horovod:
            hvd.init()
            torch.manual_seed(1334)
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(1334)
            torch.set_num_threads(1)

            with FileLock(os.path.expanduser('~/.horovod_lock')):
                dataset = dataset_class(cmd, other_args)
        else:
            dataset = dataset_class(cmd, other_args)

        model_class = get_model_class(cmd)
        model = model_class(cmd, other_args)

        with open(os.path.join(cmd.experiment_dpath, "config.json"), "w") as writer:
            json.dump(vars(cmd), writer)

        model.train(dataset)

    elif cmd.which == "use_model":
        with open(os.path.join(cmd.experiment_dpath, "config.json")) as reader:
            experiment_configuration = Namespace(**json.load(reader))
            experiment_configuration.which = cmd.which
            experiment_configuration.method = cmd.method

        model_class = get_model_class(experiment_configuration, False)
        model = model_class(experiment_configuration, other_args)

        dataset_class = get_dataset_class(experiment_configuration, False)
        dataset = dataset_class(experiment_configuration, other_args)
        method = getattr(dataset_class, cmd.method)
        method(dataset, model.keras_model)
