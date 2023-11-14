from datetime import datetime
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, BackboneFinetuning
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from omegaconf import OmegaConf

from .data import get_dataset
from .model import SitSmartModel

class FreezeBatchNorm(BackboneFinetuning):
    def freeze_before_training(self, pl_module):
        super().freeze_before_training(pl_module)
        self.freeze(pl_module.backbone, train_bn=False)


def get_loggers(log_dir):
    if log_dir is None:
        return False, None  # No loggers
    log_dir = Path(log_dir) / datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    loggers = []
    loggers.append(TensorBoardLogger(log_dir))
    # csv logger
    loggers.append(CSVLogger(log_dir))
    return loggers, log_dir


def train_model(cfg):
    train_dataset, val_dataset = get_dataset(**cfg.dataset)
    model = SitSmartModel(**cfg.model)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.train_batch_size, shuffle=True,
                                                   num_workers=cfg.train.num_workers, pin_memory=True,
                                                   persistent_workers=cfg.train.num_workers > 0, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.train.val_batch_size, shuffle=False,
                                                 num_workers=cfg.train.num_workers, pin_memory=True,
                                                 drop_last=False, persistent_workers=cfg.train.num_workers > 0)
    mode = 'max'
    if 'loss' in cfg.train.monitor:
        mode = 'min'
    early_stopping = EarlyStopping(monitor=cfg.train.monitor, patience=cfg.model.patience * 2, mode=mode)
    callbacks = [early_stopping]

    loggers, logdir = get_loggers(cfg.train.log_dir)
    if logdir is not None:
        # save cfg OmegaConf to logdir as yml file
        with open(logdir / "config.yaml", "w") as f:
            OmegaConf.save(config=cfg, f=f)

    if cfg.train.save_model_dir is not None:
        model_checkpoint = ModelCheckpoint(filename=cfg.model.model_name+'_{epoch}_{val_loss:.2f}',  monitor=cfg.train.monitor, dirpath=cfg.train.save_model_dir,
                                           save_weights_only=True, mode=mode)
        callbacks.append(model_checkpoint)
    trainer = L.Trainer("auto", logger=loggers, num_sanity_val_steps=0, enable_progress_bar=cfg.train.enable_progress_bar,
                        callbacks=callbacks, **cfg.train.trainer)
    trainer.fit(model, train_dataloader, val_dataloader)
    loss = early_stopping.best_score
    return float(loss)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg):
    print('Starting training with config:')
    print(OmegaConf.to_yaml(cfg))
    #loss = train_model(cfg)  # train_model_detection(cfg)
    #print(f'Final loss: {loss:.4f}')

if __name__ == '__main__':
    main()
