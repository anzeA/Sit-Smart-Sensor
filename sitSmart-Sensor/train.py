import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping

from .data import get_dataset
from .model import SitSmartModel


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
    trainer = L.Trainer("auto", logger=L.loggers.TensorBoardLogger(cfg.log_dir), max_epochs=cfg.max_epochs,
                        callbacks=[early_stopping])
    trainer.fit(model, train_dataloader, val_dataloader)
    loss = early_stopping.best_score
    return loss


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    loss = train_model(cfg)
    print(f"Best loss: {loss}")
