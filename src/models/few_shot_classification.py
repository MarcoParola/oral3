import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.decomposition import PCA
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import matplotlib.pyplot as plt
import hydra
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np
from PIL import Image
from src.utils import get_tensorboard_logger, log_confusion_matrix_tensorboard, log_confusion_matrix_wandb

# file da chiamare AUTOENCODER
# 1 modificare loss functino MSE
# forward solo encoder e decoder
# estrazione feature solo encoder
# vedere cae --> extract features --> clustering
# fare tutto test set e poi solo casi ancora
# train.py da utilizzare per testare autoencoder
class OralFewShotClassifierModule(pl.LightningModule):
    def __init__(self, weights, num_classes, lr=1e-4, max_epochs=150):
        super(OralFewShotClassifierModule, self).__init__()
        self.model = nn.Sequential()
        # Encoder
        self.model.encoder = nn.Sequential(
            nn.Conv2d(3, 1024, kernel_size=3, stride=2, padding=1),  # Output size: 112x112
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),  # Output size: 56x56
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),  # Output size: 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),  # Output size: 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  # Output size: 7x7
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Decoder
        self.model.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: 56x56
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: 112x112
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output size: 224x224
            nn.Sigmoid()
        )

        self.fc_rule = nn.Linear(768, 64 * 7 * 7)  # Rule embedding size is 768
        self.classifier = nn.Linear(224*224*3, num_classes)
        self.dropout = nn.Dropout(0.5)
        #self.training_mode = True

        self.num_classes = num_classes
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        self.current_labels = None
        self.current_ids = None
        self.current_imgName = None
        self.total_predictions = None
        self.total_labels = None
        self.classes = ['Neoplastic', 'Aphthous', 'Traumatic']

    def forward(self, image, rule_embedding):
        #print("forward oral few shot classifier module")
        #print("Input size:", x.size())
        encoded = self.model.encoder(image)
        encoded = encoded.view(encoded.size(0), -1)  # Flatten the encoded tensor
        # controlla se rule_embedding è tutto 0
        if not torch.all(rule_embedding == 0):
            print("combined")
            #rule_embedding = self.fc_rule(rule_embedding)
            #combined = encoded * rule_embedding
            combined = encoded
        else:
            print("not combined")
            combined = encoded
        #print("Encoded size:", encoded.size())
        combined = combined.view(-1, 64, 7, 7)  # Reshape back to the original size for the decoder
        decoded = self.model.decoder(combined)
        #print("Decoded size:", decoded.size())
        decoded_flat = decoded.view(decoded.size(0), -1)  # Appiattisce il tensor decodificato per la classificazione
        output = self.classifier(self.dropout(decoded_flat))

        return output
        
    def training_step(self, batch, batch_idx):
        #print("training step")
        #self.model.training_mode = True  
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        #print("validation step")
        #self.model.training_mode = True
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        #print("test step")
        #self.model.training_mode = False
        imgs, labels, ids, imgName, rule_embedding = batch
        self.current_labels = labels
        self.current_ids = ids
        self.current_imgName = imgName

        self.eval()

        decoded = self(imgs, rule_embedding)
        
        #loss = self.criterion(logits, labels)

        predictions = torch.argmax(decoded, dim=1)
        print("Predictions:", predictions)
        print("Labels:", labels)
        accuracy = accuracy_score(labels.cpu(), predictions.cpu())
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, logger=True)
        self.log('recall', recall_score(labels.cpu(), predictions.cpu(), average='macro'), on_step=True, on_epoch=True, logger=True)
        self.log('precision', precision_score(labels.cpu(), predictions.cpu(), average='macro'), on_step=True, on_epoch=True, logger=True)
        self.log('f1', f1_score(labels.cpu(), predictions.cpu(), average='macro'), on_step=True, on_epoch=True, logger=True)

        if self.total_labels is None:
            self.total_labels = labels.cpu().numpy()
            self.total_predictions = predictions.cpu().numpy()
        else:
            self.total_labels = np.concatenate((self.total_labels, labels.cpu().numpy()), axis=None)
            self.total_predictions = np.concatenate((self.total_predictions, predictions.cpu().numpy()), axis=None)

        # Log the confusion matrix at the end of all test batches
        if self.trainer.num_test_batches[0] == batch_idx + 1:
            log_confusion_matrix_wandb(self.logger.__class__.__name__.lower(), self.logger.experiment, self.total_labels, self.total_predictions, self.classes)
            tb_logger = get_tensorboard_logger(self.trainer.loggers)
            log_confusion_matrix_tensorboard(actual=self.total_labels, predicted=self.total_predictions, classes=self.classes, writer=tb_logger)

        #return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img, label, ids, imgName, rule_embedding = batch
        decoded = self(img, rule_embedding)
        return decoded
    
    def _common_step(self, batch, batch_idx, stage):
        torch.set_grad_enabled(True)
        #print("common step")
        imgs, labels, ids, imgName, rule_embedding = batch
        decoded = self(imgs, rule_embedding)
        loss = self.criterion(decoded, labels)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)

        '''if stage == "val" and batch_idx == 0:
            target_layers = [self.model.autoencoder.encoder[-1]]
            cam = HiResCAM(model=self, target_layers=target_layers, use_cuda=False)
            for index, image in enumerate(imgs[0:10]):
                label = labels[index]
                target = [ClassifierOutputTarget(label)]
                rule_embedding_dummy = torch.zeros(768)
                grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=target)
                grayscale_cam = grayscale_cam[0, :]
                grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
                image_for_plot = image.permute(1, 2, 0).numpy()
                fig, ax = plt.subplots()
                ax.imshow(image_for_plot)
                ax.imshow((grayscale_cam * 255).astype('uint8'), cmap='jet', alpha=0.75)
                os.makedirs(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps', exist_ok=True)
                plt.savefig(os.path.join(
                    f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps/saliency_map_epoch_{self.current_epoch}_image_{index}.pdf'),
                    bbox_inches='tight')
                plt.close()
        '''
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-5)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]
    
    