import argparse
import yaml

import os
import random
import logging

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from omegaconf import OmegaConf

from src.utils.export_model import ModelExport
from src.utils.tensorboard import TensorBoardLoggerWithMetrics
from src.utils.model_factory import ModelFactory
from src.utils.options import BaseOptions
from src.utils.versioning import get_git_diff

from hydra.experimental import compose, initialize
from sklearn.model_selection import ParameterGrid
from src.utils.checkpointing import set_latest_checkpoint
from src.data.sequence_module import AlternateSequenceDataModule

# register models
import src.models

# This is to avoid using too many CPUs as per
# https://discuss.pytorch.org/t/cpu-usage-far-too-high-and-training-inefficient/57228
# Somehow, model.dataset.num_workers: 1 does not help
torch.set_num_threads(2)


def run(cfg: BaseOptions):
    # resolve variable interpolation
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)

    logging.info("Configuration:\n%s" % OmegaConf.to_yaml(cfg))

    if not torch.cuda.is_available():
        logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        logging.info("!!! CUDA is NOT available !!!")
        logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Create data module
    # Note: we need to manually process the data module in order to get a valid skeleton
    # We don't call setup() as this breaks ddp_spawn training for some reason
    # Calling setup() is not required here as we only need access to the skeleton data, achieved with get_skeleton()
    dm = instantiate(cfg.dataset)
    dm.prepare_data()

    # create model
    model = None
    if isinstance(dm, AlternateSequenceDataModule):
        model = ModelFactory.instantiate(cfg)
    else:
        try:
            model = ModelFactory.instantiate(cfg, skeleton=dm.get_skeleton())
        except AttributeError:
            model = ModelFactory.instantiate(cfg)

    # setup logging
    metrics = model.get_metrics()
    tb_logger = TensorBoardLoggerWithMetrics(save_dir=cfg.logging.path,
                                             name=cfg.logging.name,
                                             version=cfg.logging.version,
                                             metrics=metrics)
    all_loggers = [tb_logger]

    # setup callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(dirpath=tb_logger.log_dir + '/checkpoints', save_last=cfg.logging.checkpoint.last,
                                     save_top_k=cfg.logging.checkpoint.top_k, monitor=cfg.logging.checkpoint.monitor,
                                     mode=cfg.logging.checkpoint.mode, every_n_epochs=cfg.logging.checkpoint.every_n_epochs))
    if cfg.logging.export_period > 0:
        callbacks.append(ModelExport(dirpath=tb_logger.log_dir + '/exports', filename=cfg.logging.export_name, period=cfg.logging.export_period))
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # Set random seem to guarantee proper working of distributed training
    # Note: we do it just before instantiating the trainer to guarantee nothing else will break it
    rnd_seed = cfg.seed
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)
    logging.info("Random Seed: %i" % rnd_seed)

    # Save Git diff for reproducibility
    current_log_path = os.path.normpath(os.getcwd() + "/./" + tb_logger.log_dir)
    logging.info("Logging saved to: %s" % current_log_path)
    if not os.path.exists(current_log_path):
        os.makedirs(current_log_path)
    git_diff = get_git_diff()
    if git_diff != "":
        with open(current_log_path + "/git_diff.patch", 'w') as f:
            f.write(git_diff)

    # training
    trainer = pl.Trainer(logger=all_loggers, callbacks=callbacks, **cfg.trainer)
    trainer.fit(model, datamodule=dm)

    # export
    model.export(tb_logger.log_dir + '/model.onnx')

    # test
    metrics = model.evaluate()
    if isinstance(dm, AlternateSequenceDataModule):
        print("=================== ANIDANCE BENCHMARK =====================")
        print("L2P@5", "L2P@15", "L2P@30" )
        print("{:.3f}".format(metrics["L2P@5"]), "{:.4f}".format(metrics["L2P@15"]), "{:.4f}".format(metrics["L2P@30"]))
        print("============================================================")
    else:
        print("=================== LAFAN BENCHMARK ========================")
        print("L2Q@5", "L2Q@15", "L2Q@30", "L2P@5", "L2P@15", "L2P@30", "NPSS@5","NPSS@15", "NPSS@30", )
        print("{:.3f}".format(metrics["L2Q@5"]), "{:.4f}".format(metrics["L2Q@15"]), "{:.4f}".format(metrics["L2Q@30"]), \
            "{:.3f}".format(metrics["L2P@5"]), "{:.4f}".format(metrics["L2P@15"]), "{:.4f}".format(metrics["L2P@30"]), \
            "{:.4f}".format(metrics["NPSS@5"]), "{:.5f}".format(metrics["NPSS@15"]), "{:.5f}".format(metrics["NPSS@30"]))
        print("============================================================")
    

def main(filepath: str, overrides: list = []):
    with open(filepath) as f:
        experiment_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    config_path = "src/configs"
    initialize(config_path=config_path)

    base_config = experiment_cfg["base_config"]
    experiment_params = experiment_cfg["parameters"]
    for k in experiment_params:
        if not isinstance(experiment_params[k], list):
            experiment_params[k] = [experiment_params[k]]

    param_grid = ParameterGrid(experiment_params)
    for param_set in param_grid:
        param_overrides = []

        for k in param_set:
            param_overrides.append(k + "=" + str(param_set[k]))

        # add global overrides last
        param_overrides += overrides

        cfg = compose(base_config + ".yaml", overrides=param_overrides)

        set_latest_checkpoint(cfg)

        run(cfg.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False, description="Experiment")
    parser.add_argument('--config', type=str, help='Path to the experiment configuration file', required=True)
    parser.add_argument("overrides", nargs="*",
                        help="Any key=value arguments to override config values (use dots for.nested=overrides)", )
    args = parser.parse_args()

    main(filepath=args.config, overrides=args.overrides)
