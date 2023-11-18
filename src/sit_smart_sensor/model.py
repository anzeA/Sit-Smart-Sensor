import cv2
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassF1Score

from torchvision import transforms, models

torch.set_float32_matmul_precision('high')


class SitSmartModel(L.LightningModule):
    def __init__(self, train_n_layers: int = 1, model_name: str = 'resnet34', lr: float = 1e-3,
                 weight_decay: float = 1e-6, patience: int = 5, reduce_factor: float = 0.5,dropout_rate:float=0., **kwargs):
        super().__init__()
        # check types
        assert isinstance(train_n_layers, int), f"train_n_layers must be an integer, but got {type(train_n_layers)}"
        assert isinstance(model_name, str), f"backbone must be a string, but got {type(model_name)}"
        assert isinstance(lr, float), f"lr must be a float, but got {type(lr)}"
        assert isinstance(weight_decay, float), f"weight_decay must be a float, but got {type(weight_decay)}"
        assert isinstance(patience, int), f"patience must be an integer, but got {type(patience)}"
        assert isinstance(reduce_factor, float), f"reduce_factor must be a float, but got {type(reduce_factor)}"
        assert isinstance(dropout_rate, float), f"dropout_rate must be a float, but got {type(dropout_rate)}"
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_n_layers = train_n_layers
        self.patience = patience
        self.reduce_factor = reduce_factor
        self.model_name = model_name
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights='DEFAULT')
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(weights='DEFAULT')
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT')
        else:
            raise ValueError(f"model_name {model_name} not supported. Only resnet18, resnet34 and resnet50 are supported.")

        num_filters = self.backbone.fc.in_features
        layers = list(self.backbone.children())[:-1] # remove fc layer

        self.backbone = nn.Sequential(*layers)
        num_target_classes = 3

        # freeze batch norms
        #for m in self.backbone.modules():
        #    if isinstance(m, nn.BatchNorm2d):
        #        m.track_running_stats = False # freeze running stats


        clf_layer = [nn.Flatten(),nn.Dropout(dropout_rate)]
        for i in range(train_n_layers):
            clf_layer.append(nn.Linear(num_filters, num_filters))
            clf_layer.append(nn.Dropout(dropout_rate))
            clf_layer.append(nn.LeakyReLU())
        clf_layer.append(nn.Linear(num_filters, num_target_classes))
        self.classifier = nn.Sequential(
            *clf_layer
        )
        self.force_grad = False
        self.loss_module = nn.CrossEntropyLoss()
        self.metrics_train = self._create_metrics("_train")
        self.metrics_test = self._create_metrics("_test")
        self.metrics_val = self._create_metrics("_val")

    def _create_metrics(self, postfix):
        return MetricCollection(
            [MulticlassAccuracy(3, average='macro'), MulticlassF1Score(num_classes=3, average='macro'),
             MulticlassAUROC(num_classes=3, average='macro')]).clone(postfix=postfix)

    def forward(self, imgs):
        # checl shapes
        B, C, H, W = imgs.shape
        if C != 3:
            raise ValueError(f"Expected number of channels 3, got {C}")
        if self.force_grad:
            x = self.backbone(imgs)
        else:
            with torch.no_grad():
                x = self.backbone(imgs)
        x = self.classifier(x)
        return x

    def predict_proba(self, imgs):
        if imgs.device != self.device:
            imgs = imgs.to(self.device)
        return torch.softmax(self(imgs), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, verbose=False,
                                                               mode='min',
                                                               factor=self.reduce_factor )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):

        images, labels = batch
        preds = self.forward(images)

        loss = self.loss_module(preds, labels)
        self.metrics_train.update(preds, labels)

        # Log
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
        self.log_dict(self.metrics_train, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0 and self.current_epoch % 10 == 0:
            self._log_tb_images(images, preds, labels, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.forward(images)
        loss = self.loss_module(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        preds = torch.softmax(preds, dim=1)
        self.metrics_val.update(preds, labels)
        self.log_dict(self.metrics_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _log_tb_images(self, images, preds, labels, batch_idx) -> None:
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')
        preds = torch.softmax(preds, dim=1)

        images = (images * 255).type(torch.uint8)  # in [0,1] convert to [0,255]
        images = [self._add_predictions_as_text_to_image(image, probs, label) for i, (image, probs, label) in
                  enumerate(zip(images, preds, labels))]

        images = torch.stack(images)
        tb_logger.add_images(f"Image/{batch_idx}", images, global_step=self.global_step, dataformats='NHWC')

    def _add_predictions_as_text_to_image(self, img, probs, label):
        img = img.cpu().numpy()
        img = np.einsum('chw -> hwc', img)
        img = cv2.UMat(img)
        probability = {'negative': probs[0].item(), 'no_person': probs[1].item(), 'positive': probs[2].item()}
        for i, (k, v) in enumerate(probability.items()):
            text = f'{k.replace("_", " ")}: {int(100 * v)}%'
            color = (0, 0, 0)
            cv2.putText(img, text, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        labels2class = {0: 'negative', 1: 'no_person', 2: 'positive'}
        text = f'Ground Truth: {labels2class[label.item()]}'
        color = (0, 0, 0)
        cv2.putText(img, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return torch.from_numpy(cv2.UMat.get(img))
