import os
from datetime import datetime
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from data import get_dataset
from model import SitSmartModel


def get_loggers(log_dir,):
    if log_dir is None:
        return False  # No loggers
    log_dir = Path(log_dir)/ datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    loggers = []
    loggers.append(TensorBoardLogger(log_dir))
    # csv logger
    loggers.append(CSVLogger(log_dir))
    return loggers


def train_model(cfg):
    model = SitSmartModel(**cfg.model)
    train_dataset, val_dataset = get_dataset(**cfg.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                                                   num_workers=cfg.num_workers,
                                                   persistent_workers=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                                                 num_workers=cfg.num_workers,
                                                 drop_last=False, persistent_workers=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=cfg.patience)
    loggers = get_loggers(cfg.log_dir)
    trainer = L.Trainer("auto", logger=loggers,num_sanity_val_steps=0,
                        callbacks=[early_stopping], **cfg.train)
    trainer.fit(model, train_dataloader, val_dataloader)
    loss = early_stopping.best_score
    print(f"Best loss: {loss}, Type: {type(loss)}")
    return float(loss)


@hydra.main(config_path="config", config_name="config",version_base="1.1")
def main(cfg):
    print(cfg)
    loss = train_model(cfg)
    print(f"Best loss: {loss}")


if __name__ == '__main__':
    main()
