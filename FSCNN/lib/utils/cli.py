from pytorch_lightning.utilities import cli as CLI
import pytorch_lightning as pl
from lib.utils.log import log_system_info
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
)
import logging
from time import time, sleep

import torch


logger = logging.getLogger(__name__)


class CustomCli(CLI.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--name", default="debug", help="name of the condition")
        parser.add_argument("--root_dir", default="output")
        parser.add_argument(
            "--split_ind",
            type=int,
            default=-1,
            help="index of the run, used to name the run directory if set to value >= 0",
        )
        parser.add_argument(
            "--data_dir",
            default="../data/annotated/rsna_bone_age/",
            type=str,
            help="legacy, dir for testing using tta",
        )
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults(
            {
                "early_stopping.monitor": "val_mad",
                "early_stopping.patience": 50,
                "early_stopping.mode": "min",
                "lr_monitor.logging_interval": "epoch",
            }
        )

    def setup_callbacks(self) -> None:
        """
        handle logging and checkpointing
        """
        output_path = self.config["root_dir"]
        version = (
            f"split_{self.config['split_ind']}"
            if self.config["split_ind"] >= 0
            else None
        )

        tb_logger = pl.loggers.tensorboard.TensorBoardLogger(
            save_dir=output_path,
            name=self.config["name"],
            log_graph=True,
            default_hp_metric=False,
            version=version,
        )
        logger.info(f"Output directory set to {tb_logger.log_dir}")
        ckp_callback = pl.callbacks.ModelCheckpoint(
            monitor=self.config.early_stopping.monitor,
            dirpath=tb_logger.log_dir + "/ckp/",
            filename="best_model",
            save_top_k=1,
            mode="min",
            save_last=True,
            auto_insert_metric_name=False,
            verbose=True,
        )
        self.trainer.callbacks.append(ckp_callback)
        self.trainer.callbacks.append(pl.callbacks.DeviceStatsMonitor())
        self.trainer.logger = tb_logger
        for cb in self.trainer.callbacks:
            if isinstance(cb, pl.utilities.cli.SaveConfigCallback):
                cb.overwrite = True
        self.trainer.callbacks.append(pl.callbacks.ModelSummary(max_depth=3))

    def get_model_weights(self) -> str:
        for cb in self.trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                logger.info("Using best model")
                return cb.best_model_path

    def log_info(self) -> None:
        log_system_info(logger)

    def log_train_stats(self) -> None:
        logger.info(f"===== Training finished ======")
        logger.info(
            f"Training time : {(time() - self.model.start_time) / 60:.2f}min for {self.model.global_step} steps of training"
        )
        logger.info(
            f"Training speed: {(self.model.global_step / (time() - self.model.start_time)):.2f} steps/second"
        )
