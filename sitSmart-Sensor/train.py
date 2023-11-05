from datetime import datetime
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from omegaconf import OmegaConf

from data import get_dataset
from model import SitSmartModel


def get_loggers(log_dir, ):
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
    model = SitSmartModel(**cfg.model)
    train_dataset, val_dataset = get_dataset(**cfg.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                                   num_workers=cfg.num_workers, pin_memory=True,
                                                   persistent_workers=cfg.num_workers > 0, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers, pin_memory=True,
                                                 drop_last=False, persistent_workers=cfg.num_workers > 0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=cfg.patience)
    callbacks = [early_stopping]
    loggers, logdir = get_loggers(cfg.log_dir)
    if logdir is not None:
        # save cfg OmegaConf to logdir as yml file
        with open(logdir / "config.yaml", "w") as f:
            OmegaConf.save(config=cfg, f=f)
    if cfg.setdefault('save_model_dir', None) is not None:
        model_checkpoint = ModelCheckpoint(monitor='val_loss', dirpath=cfg.save_model_dir,
                                           save_weights_only=True, mode='min')
        callbacks.append(model_checkpoint)
    trainer = L.Trainer("auto", logger=loggers, num_sanity_val_steps=0, enable_progress_bar=cfg.enable_progress_bar,
                        callbacks=callbacks, **cfg.train)
    trainer.fit(model, train_dataloader, val_dataloader)
    loss = early_stopping.best_score
    return float(loss)


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg):
    print(cfg)
    loss = train_model(cfg)
    print(f"Best loss: {loss}")


if __name__ == '__main__':
    main()
