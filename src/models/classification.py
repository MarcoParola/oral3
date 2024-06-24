import torch
import torchvision
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import wandb, TensorBoardLogger
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import hydra
import os
import wandb
from src.utils import log_confusion_matrix_tensorboard, get_tensorboard_logger, log_confusion_matrix_wandb
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.models.feature_extraction import create_feature_extractor


class OralClassifierModule(LightningModule):

    def __init__(self, weights, num_classes, output_dim, lr=10e-3, max_epochs=150):
        super().__init__()
        self.current_labels = None
        self.current_ids = None
        self.current_imgName = None
        self.save_hyperparameters()
        assert "." in weights, "Weights must be <MODEL>.<WEIGHTS>"
        weights_cls = weights.split(".")[0]
        weights_name = weights.split(".")[1]
        self.model_name = weights.split("_Weights")[0].lower()
        self.num_classes = num_classes
        self.output_dim = output_dim
        weights_cls = getattr(torchvision.models, weights_cls)
        weights = getattr(weights_cls, weights_name)
        self.model = getattr(torchvision.models, self.model_name)(weights=weights)
        self._set_model_classifier(weights_cls, num_classes)

        self.preprocess = weights.transforms()
        self.loss = torch.nn.CrossEntropyLoss()
        self.total_predictions = None
        self.total_labels = None
        self.classes = ['Neoplastic', 'Aphthous', 'Traumatic']

        name = str(weights_cls)
        if "SqueezeNet1_1" in name or "SqueezeNet1_0" in name:
            self.feature_extractor = create_feature_extractor(self.model, ['classifier'])
        elif "Swin" in name:
            self.feature_extractor = create_feature_extractor(self.model, ['head'])
        elif "ConvNeXt" in name:
            self.feature_extractor = create_feature_extractor(self.model, ['classifier'])
        elif "ViT" in name:
            self.feature_extractor = create_feature_extractor(self.model, ['heads'])



    def forward(self, x):
        x = self.model(x)
        return x
    
    def extract_features(self, x):
        return self.feature_extractor(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        imgs, labels, ids, names = batch
        self.current_labels = labels
        self.current_ids = ids
        self.current_imgName = names

        self.eval()
        x = self.preprocess(imgs)
        y_hat = self(x)

        predictions = torch.argmax(y_hat, dim=1)

        # print labels and predictions
        #print("Labels:", labels)
        #print("Predictions:", predictions)
        
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
        img, label, id, name = batch
        x = self.preprocess(img)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-5)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def _common_step(self, batch, batch_idx, stage):
        torch.set_grad_enabled(True)

        imgs, labels, ids, names = batch
        x = self.preprocess(imgs)
        y_hat = self(x)
        loss = self.loss(y_hat, labels)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)

        if stage == "val" and batch_idx == 0:
            if "vit" in self.model_name:
                target_layers = [self.model.conv_proj]
            elif "convnext" in self.model_name:
                target_layers = [self.model.features[-1][-1]]
                # swin è da cercare meglio il target layer
            elif "swin" in self.model_name:
                target_layers = [self.model.features[0][0]]
            elif "squeezenet" in self.model_name:
                target_layers = [self.model.features[-1]]

            cam = HiResCAM(model=self, target_layers=target_layers, use_cuda=False)
            for index, image in enumerate(imgs[0:10]):
                label = labels[index]
                target = [ClassifierOutputTarget(label)]
                grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=target)
                grayscale_cam = grayscale_cam[0, :]
                grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
                image_for_plot = image.permute(1, 2, 0).numpy()
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

    def _set_model_classifier(self, weights_cls, num_classes):
        weights_cls = str(weights_cls)
        if "ConvNeXt" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Flatten(1),
                torch.nn.Linear(self.model.classifier[2].in_features, self.output_dim)
            )
        elif "EfficientNet" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[1].in_features, self.output_dim)
            )
        elif "MobileNet" in weights_cls or "VGG" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[0].in_features, self.output_dim)
            )
        elif "DenseNet" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier.in_features, self.output_dim)
            )
        elif "MaxVit" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(self.model.classifier[5].in_features, self.output_dim)
            )
        elif "ResNet" in weights_cls or "RegNet" in weights_cls or "GoogLeNet" in weights_cls:
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.fc.in_features, self.output_dim)
            )
        elif "Swin" in weights_cls:
            self.model.head = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.head.in_features, self.output_dim)
            )
        elif "ViT" in weights_cls:
            self.model.heads = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.hidden_dim, self.output_dim)
            )
        elif "SqueezeNet1_1" in weights_cls or "SqueezeNet1_0" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Conv2d(512, self.output_dim, kernel_size=(1, 1), stride=(1, 1)),
                torch.nn.AvgPool2d(kernel_size=13, stride=1, padding=0)
            )

        self.model.lastLayer = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.output_dim, num_classes)
            )

