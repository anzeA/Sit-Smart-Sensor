import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics import MetricCollection
# PyTorch
# Torchvision
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC
# from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms, models, datasets

torch.set_float32_matmul_precision('high')


class SitSmartModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resnet = models.resnet18(weights='DEFAULT')
        num_ftrs = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(num_ftrs, 1)
        self.loss_module = nn.BCELoss()
        self.metrics_train = self._create_metrics("_train")
        self.metrics_test = self._create_metrics("_test")
        self.metrics_val = self._create_metrics("_val")

    def _create_metrics(self, postfix):
        return MetricCollection([BinaryAccuracy(), BinaryF1Score(), BinaryAUROC()]).clone(
            postfix=postfix)  # BinaryRecall(), BinaryPrecision(),

    def forward(self, imgs):
        # checl shapes
        B, C, H, W = imgs.shape
        if (C, H, W) != (3, 224, 224):
            raise ValueError(f"Expected input shape (batch_size, 3, 224, 224), got {imgs.shape}")
        # Forward function that is run when visualizing the graph

        imgs = self.transform(imgs)
        emb = self.resnet(imgs)
        return torch.sigmoid(emb)[..., 0]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.resnet.fc.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150, 200], gamma=0.1, verbose=True)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        if batch_idx == 0:
            self._log_tb_images(imgs, batch_idx)
        preds = self.forward(imgs.to(self.device))
        labels = labels.float().to(self.device)
        loss = self.loss_module(preds, labels)
        self.metrics_train(preds, labels.to(self.device))
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log_dict(self.metrics_train, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        # move to same device as self
        imgs.to(self.device)
        preds = self.forward(imgs)
        labels = labels.to(self.device)
        loss = self.loss_module(preds, labels.float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metrics_val(preds, labels)
        # By default logs it per epoch (weighted average over batches)
        self.log_dict(self.metrics_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        imgs.to(self.device)
        labels = labels.to(self.device)

        preds = self.forward(imgs)
        self.metrics_test(preds, labels)
        # By default logs it per epoch (weighted average over batches)
        # self.log("metrics", self.metrics_test)
        self.log_dict(self.metrics_test, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def _log_tb_images(self, images, batch_idx) -> None:

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

            # Log the images (Give them different names)
        tb_logger.add_images(f"Image/{batch_idx}", images, global_step=self.global_step, dataformats='NCHW')


if __name__ == '__main__':
    model = SitSmartModel()

