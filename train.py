from __future__ import annotations

import os
import random

import dotenv
import hydra
import lightning.pytorch as pl
import numpy as np
import omegaconf
import torch
import wandb

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# MixedPrecisionPlugin
from lightning.pytorch.plugins import MixedPrecisionPlugin
from lightning.pytorch.profilers import PyTorchProfiler
from loguru import logger
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity

from fdlsar import utils
from fdlsar.callbacks.ssl_online import SSLOnlineEvaluator
from fdlsar.callbacks.wandb_callbacks import LogDataSum, LogImagePredictions

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

import sys

# ----------------------------------------------------------------------------------
# suggested invocation (use your own configs folder
#
#    ./scripts/gitandrun.py python train.py --config-path configs/s1grd_veg
#
# ----------------------------------------------------------------------------------


@hydra.main(
    version_base="1.1",
    config_path="configs/default",
    config_name="train.yaml",
)
def main(config: DictConfig):
    # setting separate seeds for data and model
    if "model_seed" in config:
        model_seed = config.model_seed
    else:
        model_seed = 42

    if "data_seed" in config:
        data_seed = config.data_seed
    else:
        data_seed = 42

    logger.info(f"training with model seed {model_seed}")
    logger.info(f"training with data seed {data_seed}")

    random.seed(data_seed)
    np.random.seed(data_seed)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)

    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    # get tags
    tags = config.tags if "tags" in config else []
    if isinstance(tags, str):
        tags = tags.split()

    experiment_name = config.experiment_name

    wandb_logger = WandbLogger(
        name=experiment_name,
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
        tags=tags,
    )

    # log data sum if requested in config
    if "log_data_sum" in config:
        dataloader = hydra.utils.instantiate(config.dataloader)
        dataloader.prepare_data()
        input = list(config.dataloader.datasets_spec.keys())[
            0
        ]  # picking the first key as example
        data_sum = 0
        for batch in dataloader.train_dataloader():
            data_sum += batch[input].sum()
        logger.info("-------------")
        logger.info(f"dataset sum : {data_sum}")
        logger.info("-------------")

    log_cmd_wandb = config.log_cmd_wandb if "log_cmd_wandb" in config else False
    logger.info(log_cmd_wandb)
    if log_cmd_wandb:
        cmd = " ".join(sys.argv)
        logger.info(f"Command executed: {cmd}")

    logger.info("-------------")
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info("-------------")

    log_config_as = config.log_config_as if "log_config_as" in config else "yaml"
    if log_config_as == "yaml":
        yaml_str = omegaconf.OmegaConf.to_yaml(config)
        logger.debug(f"Hydra-config: {yaml_str}")
    else:
        logger.debug(f"Hydra-config: {config}")

    if "load_checkpoint" in config.keys():
        hr = utils.find_hydra_run_path(
            outputs_dir=config.load_checkpoint.outputs_dir,
            wandb_runid=config.load_checkpoint.wandb_runid,
        )
        logger.info("hydra run path for previous model", hr)
        model = utils.load_ckpt_from_hydra_run(hr)
    else:
        logger.info("instantiating model")
        model = hydra.utils.instantiate(config.model)
        if isinstance(model, tuple):
            logger.info("selecting first model in tuple")
            model = model[0]

    logger.info(f"full model checksum  {utils.print_checksum_of_model(model):.4f}")


    val_checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        monitor=monitor_metric,
        save_top_k=1,
        mode="min",
        filename=f"{aoi}-{experiment_name}"
        + "-best-monitored-metric-epoch={epoch:02d}",
        auto_insert_metric_name=False,
    )

    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(
            skip_first=10, wait=5, warmup=1, active=5, repeat=2
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    )

    # train with configured model and dataloader
    dataloader = hydra.utils.instantiate(config.dataloader)

    precision = config.get("precision") if config.get("precision") else "32-true"

    # this is necessary to train using antialiasing torch algorithms
    # warn_only is used to log the error but not stop the train

    if config.get("use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Define the limit of test_batches
    limit_test_batches = (
        1.0 if not hasattr(config, "limit_test_batches") else config.limit_test_batches
    )

    devices = config.devices if hasattr(config, "devices") else "auto"
    check_val_every_n_epoch = (
        config.check_val_every_n_epoch
        if hasattr(config, "check_val_every_n_epoch")
        else 1
    )

    strategy = (
        config.get("strategy")
        if config.get("strategy")
        else "ddp_find_unused_parameters_true"
    )

    plugins = None

    if precision == "16-mixed":
        logger.info("---- using mixed precision, adding plugin")
        plugins = [
            MixedPrecisionPlugin(
                "16-mixed", device="cuda", scaler=torch.cuda.amp.GradScaler()
            )
        ]

    callbacks=[]

    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        # accelerator="gpu",
        strategy=strategy,
        devices=devices,
        # add plugins
        plugins=plugins,
        max_epochs=config.max_epochs,
        precision=precision,
        log_every_n_steps=config.log_every_n_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=False,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=limit_test_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        # profiler for
        # profiler=profiler,
        # no sanity check, no validation step:
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=0,  # Disable sanity check validation steps
    )

    if "log_weights_and_gradients" in config and config.log_weights_and_gradients:
        logger.info("logging weights")
        wandb_logger.watch(model, log="all", log_freq=1, log_graph=True)

    trainer.fit(model, dataloader)

    logger.info(f"---- getting best model from {os.getcwd()}")
    best_model = utils.load_ckpt_from_hydra_run(os.getcwd())

    trainer.test(model=best_model, datamodule=dataloader)

if __name__ == "__main__":
    main()
