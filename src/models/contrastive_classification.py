import torch
import torchvision
import torch.nn.functional as F
from pytorch_grad_cam import HiResCAM
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor
from pytorch_lightning import LightningModule
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import hydra
from src.utils import get_tensorboard_logger, log_confusion_matrix_tensorboard, log_confusion_matrix_wandb
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms
import joblib
from sklearn.model_selection import GridSearchCV



class OralContrastiveClassifierModule(LightningModule):
    def __init__(self, weights, num_classes, output_dim, lr=1e-3, temperature=0.5, max_epochs=100):
        super().__init__()
        self.current_labels = None
        self.current_ids = None
        self.current_imgNames = None
        self.current_positives = None
        self.current_negatives = None
        self.current_positives_imgs = None
        self.current_negatives_imgs = None
        self.save_hyperparameters()
        assert "." in weights, "Weights must be <MODEL>.<WEIGHTS>"
        weights_cls = weights.split(".")[0]
        weights_name = weights.split(".")[1]
        self.model_name = weights.split("_Weights")[0].lower()
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.total_labels = None
        self.total_predictions = None
        
        if "Scratch" in weights_cls:
            
            self.type = "Scratch"

            self.model = torch.nn.Sequential()

            self.model.scratch = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(64, 128, 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(128, 256, 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(256, 512, 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
                torch.nn.Flatten(),
                torch.nn.Linear(512 * 12 * 12, self.output_dim)
            )

            self.model.lastLayer = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.output_dim, num_classes)
            )
        
        else:
            self.type = "Pretrained"
            weights_cls = getattr(torchvision.models, weights_cls)
            weights = getattr(weights_cls, weights_name)
            self.model = getattr(torchvision.models, self.model_name)(weights=weights)
            self._set_model_classifier(weights_cls, num_classes)
            self.preprocess = weights.transforms()

        self.total_predictions = None
        self.total_labels = None
        self.classes = ['Neoplastic', 'Aphthous', 'Traumatic']

        name = str(weights_cls)
        if "ConvNeXt" in name:
            self.feature_extractor = create_feature_extractor(self.model, ['classifier'])
        elif "ResNet" in name:
            self.feature_extractor = create_feature_extractor(self.model, ['fc'])
        elif "Scratch" in name:
            self.feature_extractor = create_feature_extractor(self.model, ['scratch'])

        
    def forward(self, x):
        x = self.model(x)
        return x
    
    def extract_features(self, x):
        return self.feature_extractor(x)
    
    def contrastive_loss(self, anchor, positive, negative):
        positive_similarity = F.cosine_similarity(anchor, positive)
        negative_similarity = F.cosine_similarity(anchor, negative)
        loss = torch.mean(1 - positive_similarity + negative_similarity)
        return loss
    
    def _common_step(self, batch, batch_idx, stage):
        torch.set_grad_enabled(True)

        imgs, labels, ids, names, pstvs, ngtvs, pos_imgs, neg_imgs = batch

        self.current_ids = ids
        self.current_labels = labels
        self.current_imgNames = names
        self.current_positives = pstvs
        self.current_negatives = ngtvs
        self.current_positives_imgs = pos_imgs
        self.current_negatives_imgs = neg_imgs

        if self.type == "Pretrained":
            imgs = self.preprocess(imgs)
            pos_imgs = self.preprocess(pos_imgs)
            neg_imgs = self.preprocess(neg_imgs)

        anchor_features = self(imgs)
        positive_features = self(pos_imgs)
        negative_features = self(neg_imgs)
        
        loss = self.contrastive_loss(anchor_features, positive_features, negative_features)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)

        if stage == 'val' and batch_idx == 0:
            if "resnet" in self.model_name:
                target_layers = [self.model.layer4[-1]]
            elif "convnext" in self.model_name:
                target_layers = [self.model.features[-1][-1]]
            elif "scratch" in self.model_name:
                target_layers = [self.model.scratch[-1]]
            
            cam = HiResCAM(model=self, target_layers=target_layers, use_cuda=False)
            for index, image in enumerate(imgs[0:10]):
                label = labels[index]
                target = [ClassifierOutputTarget(label)]
                grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=target)
                grayscale_cam = grayscale_cam[0, :]
                grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
                image_for_plot = image.permute(1, 2, 0).cpu().numpy()
                fig, ax = plt.subplots()
                ax.imshow(image_for_plot)
                ax.imshow((grayscale_cam * 255).astype('uint8'), cmap='jet', alpha=0.75)  # Overlay saliency map
                os.makedirs(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps',
                            exist_ok=True)
                plt.savefig(os.path.join(
                    f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps/saliency_map_epoch_{self.current_epoch}_image_{index}.pdf'),
                    bbox_inches='tight')
                plt.close()
        
        return loss
        
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-5)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]
    
    def test_step(self, batch, batch_idx):
        
        imgs, labels, ids, names, pstvs, ngtvs, pos_imgs, neg_imgs = batch

        self.current_labels = labels
        self.current_ids = ids
        self.current_imgNames = names
        self.current_positives = pstvs
        self.current_negatives = ngtvs
        self.current_positives_imgs = pos_imgs
        self.current_negatives_imgs = neg_imgs

        self.eval()

        if self.type == "Pretrained":
            imgs = self.preprocess(imgs)

        y_hat = self(imgs)

        predictions = torch.argmax(y_hat, dim=1)

        #print(f"Predictions: {predictions}")
        #print(f"Labels: {labels}")
        
        self.log('test_accuracy', accuracy_score(labels, predictions), on_step=True, on_epoch=True, logger=True)
        self.log('recall', recall_score(labels, predictions, average='micro'), on_step=True, on_epoch=True, logger=True)
        self.log('precision', precision_score(labels, predictions, average='micro'), on_step=True, on_epoch=True, logger=True)
        self.log('f1', f1_score(labels, predictions, average='micro'), on_step=True, on_epoch=True, logger=True)

        # this accumulation is necessary in order to log confusion matrix of all the test and not just the last step
        if self.total_labels is None:
            self.total_labels = labels.numpy()
            self.total_predictions = predictions.numpy()
        else:
            self.total_labels = np.concatenate((self.total_labels, labels.numpy()), axis=None)
            self.total_predictions = np.concatenate((self.total_predictions, predictions.numpy()), axis=None)

        # check if it's the last test step
        if self.trainer.num_test_batches[0] == batch_idx+1:
            # logging confusion matrix on wandb
            log_confusion_matrix_wandb(self.logger.__class__.__name__.lower(), self.logger.experiment, self.total_labels, self.total_predictions, self.classes)
            # get tensorboard logger if present il loggers list
            tb_logger = get_tensorboard_logger(self.trainer.loggers)
            # logging confusion matrix on tensorboard
            log_confusion_matrix_tensorboard(actual=self.total_labels, predicted=self.total_predictions,
                                             classes=self.classes, writer=tb_logger)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        imgs, labels, ids, names, pstvs, ngtvs, pos_imgs, neg_imgs = batch
        if self.type == "Pretrained":
            imgs = self.preprocess(imgs)
        x = self(imgs)
        return x
    
    def _set_model_classifier(self, weights_cls, num_classes):
        weights_cls = str(weights_cls)
        if "ConvNeXt" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Flatten(1),
                torch.nn.Linear(self.model.classifier[2].in_features, self.output_dim)
            )
        elif "ResNet" in weights_cls:
            self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.model.fc.in_features, self.output_dim)
        )

        self.model.lastLayer = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.output_dim, num_classes)
            )
        
