import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from src.utils import *
import hydra
import os
from omegaconf import DictConfig, OmegaConf
import flatdict


import logging
import wandb 

log = logging.getLogger(__name__)

class LossLogCallback(pl.Callback):
    """
    TensorBoard callback to log training and validation losses.
    """
    def __init__(self, log_dir: str, enable_tensorboard: bool = False):
        self.enable_tensorboard = enable_tensorboard

        # Setup per TensorBoard
        self.writer = None
        if self.enable_tensorboard:
            self.tb_log_dir = os.path.join(log_dir, "tensorboard_losses")
        else:
            self.tb_log_dir = None

        self.train_losses = []
        self.val_losses = []

    def on_fit_start(self, trainer, pl_module):
        self.train_losses = []
        self.val_losses = []
        
        if self.enable_tensorboard:
            os.makedirs(self.tb_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tb_log_dir)
            log.info(f"TensorBoard SummaryWriter initialized by LossLogCallback at: {self.tb_log_dir}")
        else:
            log.info("TensorBoard logging for LossLogCallback is disabled by configuration.")

    def on_train_epoch_end(self, trainer, pl_module):
        current_train_loss = trainer.callback_metrics.get("train/loss")
        if current_train_loss is not None:
            current_train_loss = current_train_loss.item()
            self.train_losses.append(current_train_loss)

            if self.enable_tensorboard and self.writer is not None:
                self.writer.add_scalars('train_val_loss', {'train': current_train_loss}, trainer.global_step)
            
        else:
            log.warning(f"Metric 'train/loss' not found in callback_metrics at epoch {trainer.global_step}. Skipping log.")

    def on_validation_epoch_end(self, trainer, pl_module):
        current_val_loss = trainer.callback_metrics.get("val/loss")
        if current_val_loss is not None:
            current_val_loss = current_val_loss.item()
            self.val_losses.append(current_val_loss)

            if self.enable_tensorboard and self.writer is not None:
                self.writer.add_scalars('train_val_loss', {'val': current_val_loss}, trainer.global_step)
        else:
            log.warning(f"Metric 'val/loss' not found in callback_metrics at epoch {trainer.global_step}. Skipping log.")

    def on_fit_end(self, trainer, pl_module):
        if self.enable_tensorboard and self.writer is not None:
            self.writer.close()
            log.info(f"TensorBoard SummaryWriter closed by LossLogCallback for the run in {self.tb_log_dir}.")

# class LossLogCallback(pl.Callback):
#     def on_fit_start(self, trainer, pl_module):
#         self.train_losses = []
#         self.val_losses = []

#     def on_train_epoch_end(self, trainer, pl_module):
#         self.train_losses.append(trainer.callback_metrics["train_loss"].item())
#         ###
#         log_dir = 'logs/oral/' + get_last_version('logs/oral')
#         writer = SummaryWriter(log_dir=log_dir)
#         writer.add_scalars('train_val_loss', {'train': self.train_losses[-1]}, trainer.current_epoch)
#         writer.close()


#     def on_validation_epoch_end(self, trainer, pl_module):
#         self.val_losses.append(trainer.callback_metrics["val_loss"].item())
#         log_dir = 'logs/oral/' + get_last_version('logs/oral')
#         writer = SummaryWriter(log_dir=log_dir)
#         writer.add_scalars('train_val_loss', {'val': self.val_losses[-1]}, trainer.current_epoch)
#         writer.close()

#     '''
#     def on_train_end(self, trainer, pl_module):
#         log_dir = 'logs/oral/' + get_last_version('logs/oral')
#         writer = SummaryWriter(log_dir=log_dir)
#         for i in range(0, len(self.train_losses)):
#             print("val_loss: ", self.val_losses[i])
#             print("train_loss: ", self.train_losses[i])
#             writer.add_scalars('train_val_loss', {'train': self.train_losses[i],
#                                                   'val': self.val_losses[i]}, i)
#         writer.close()
#     '''

class HydraTimestampRunCallback(pl.Callback):

    #@hydra.main(version_base=None, config_path="./config", config_name="config")
    def on_train_end(self, trainer, pl_module):
        # absolute path
        log_dir = 'logs/oral/' + get_last_version('logs/oral')
        hydra_current_timestamp = f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}'
        # decompose in day and time
        day = hydra_current_timestamp.split(os.sep)[-2]
        time = hydra_current_timestamp.split(os.sep)[-1]
        f = open(log_dir + "/hydra_run_timestamp.txt", "w+")
        # in this way is saved just day and time not all the absolute path
        f.write(day + '/' + time)
        f.close()

def hp_from_cfg(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return dict(flatdict.FlatDict(cfg, delimiter="/"))

# def get_loggers(cfg):
#     """Returns a list of loggers
#     cfg: hydra config
#     """
#     loggers = list()
#     if cfg.log.wandb:
#         from pytorch_lightning.loggers import WandbLogger
#         import wandb
#         hyperparameters = hp_from_cfg(cfg)
#         wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
#         wandb.config.update(hyperparameters)
#         wandb_logger = WandbLogger()
#         loggers.append(wandb_logger)

#     if cfg.log.tensorboard:
#         from pytorch_lightning.loggers import TensorBoardLogger
#         tensorboard_logger = TensorBoardLogger(cfg.log.path, name="oral")
#         loggers.append(tensorboard_logger)

#     return loggers


def get_loggers(cfg: DictConfig, run_output_dir: str):
    """
    Returns a list of PyTorch Lightning loggers and sets up a local file logger.
    """
    loggers = []

    # --- Local File Logger ---
    # This ensures all application logs go to a specific file for the current run.
    os.makedirs('logs/', exist_ok=True)
    log_file_path = os.path.join(run_output_dir, "run.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # Console Logger 
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))


    # Get the root logger and add the file handler.
    # This affects messages from `logging.getLogger()` calls throughout the app.
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(cfg.get('log_level', 'INFO').upper())
    log.info(f"Local file logger enabled. Logs saved to: {log_file_path}")


    # --- Weights & Biases Logger ---
    if cfg.log.get('wandb'):
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        # Ensure datetime is imported if used for run_name, as it was in previous versions.
        from datetime import datetime

        if wandb.run is None:
            run_name = f"{cfg.wandb.get('run_name', 'default_wandb_run')}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

            wandb.init(
                entity=cfg.wandb.get('entity'),
                project=cfg.wandb.get('project'),
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=run_output_dir,
            )
            log.info(f"Initialized Weights & Biases run from get_loggers: {run_name}")
        else:
            log.info("Weights & Biases run already active. Reusing existing run.")

        wandb_logger = WandbLogger(log_model=cfg.wandb.get('log_model', True))
        loggers.append(wandb_logger)
        log.info("PyTorch Lightning's built-in WandbLogger enabled.")

    # --- TensorBoard Logger ---
    if cfg.log.get('tensorboard'):
        from pytorch_lightning.loggers import TensorBoardLogger
        tensorboard_log_dir = os.path.join(run_output_dir, "tensorboard_logs")
        tensorboard_logger = TensorBoardLogger(
            save_dir=os.path.dirname(tensorboard_log_dir),
            name=os.path.basename(tensorboard_log_dir)
        )
        loggers.append(tensorboard_logger)
        log.info(f"PyTorch Lightning's built-in TensorBoardLogger enabled. Logs saved to: {tensorboard_log_dir}")

    return loggers