import torch
import torchvision
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np
from sklearn.metrics import classification_report
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
import hydra
import os

# Import utility functions for logging and visualization.
from src.utils import log_confusion_matrix_tensorboard, get_tensorboard_logger, log_confusion_matrix_wandb
from torchvision.models.feature_extraction import create_feature_extractor


class OralClassifierModule(LightningModule):
    """
    A PyTorch Lightning Module for oral classification tasks.
    Encapsulates the model architecture, training, validation, and testing logic.
    """

    def __init__(self, weights, num_classes, output_dim, lr=10e-3, max_epochs=150):
        super().__init__()

        # Configure precision for matrix multiplication to optimize performance.
        torch.set_float32_matmul_precision('medium')
        self.save_hyperparameters()

        # --- Model Configuration ---
        assert "." in weights, "Weights must be specified in the format <MODEL>.<WEIGHTS>"
        weights_cls_name = weights.split(".")[0]
        weights_name = weights.split(".")[1]
        self.model_name = weights.split("_Weights")[0].lower()
        self.num_classes = num_classes
        self.output_dim = output_dim

        weights_cls = getattr(torchvision.models, weights_cls_name)
        weights_obj = getattr(weights_cls, weights_name)
        self.model = getattr(torchvision.models, self.model_name)(weights=weights_obj)

        # Adapt the model's final classification layer for the specific task.
        self._set_model_classifier(weights_cls, num_classes)

        # Define preprocessing, loss function, and class name placeholder.
        self.preprocess = weights_obj.transforms()
        self.loss = torch.nn.CrossEntropyLoss()
        self.classes = None

        # Initialize lists to accumulate outputs for metric calculation.
        self.test_step_outputs = []
        self.validation_step_outputs = []

        # Configure target layers for Grad-CAM based on the model architecture.
        if "vit" in self.model_name:
            self.target_layers = [self.model.conv_proj]
        elif "convnext" in self.model_name:
            self.target_layers = [self.model.features[-1][-1]]
        elif "swin" in self.model_name:
            self.target_layers = [self.model.features[-1][-1].block]
        elif "squeezenet" in self.model_name:
            self.target_layers = [self.model.features[-1]]
        elif "resnet" in self.model_name:
            self.target_layers = [self.model.layer4[-1]]
        else:
            raise NotImplementedError(f"Target layers not defined for model: {self.model_name}")
        
        # Configure the feature extractor for intermediate layer activations.
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
        x = self.model.lastLayer(x) 
        return x
    
    def extract_features(self, x):
        return self.feature_extractor(x)
    
    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, is_original = batch
        x = self.preprocess(imgs)
        y_hat = self(x)
        loss = self.loss(y_hat, labels)
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=imgs.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels, _, _, is_original = batch
        x = self.preprocess(imgs)
        y_hat = self(x)
        loss = self.loss(y_hat, labels)
        predictions = torch.argmax(y_hat, dim=1)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append({'predictions': predictions.cpu(), 'labels': labels.cpu()})

        # Generate and save Grad-CAM visualizations for the first batch of each validation epoch.
        if batch_idx == 0:
            with torch.set_grad_enabled(True):
                self.model.eval()
                cam = HiResCAM(model=self, target_layers=self.target_layers)
                
                for index, image in enumerate(imgs[0:10]):
                    target_label = labels[index]
                    target = [ClassifierOutputTarget(target_label)]
                    
                    grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=target)[0, :]
                    grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
                    
                    image_for_plot = image.permute(1, 2, 0).cpu().numpy()
                    fig, ax = plt.subplots()
                    ax.imshow(image_for_plot)
                    ax.imshow((grayscale_cam * 255).astype('uint8'), cmap='jet', alpha=0.75)
                    
                    output_dir = f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    plt.savefig(
                        os.path.join(output_dir, f'saliency_map_epoch_{self.current_epoch}_image_{index}.pdf'),
                        bbox_inches='tight'
                    )
                    plt.close(fig)

        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs]).numpy()
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).numpy()

        # Use provided class names for the report, otherwise use integer indices.
        if hasattr(self, 'classes') and self.classes and len(self.classes) == self.num_classes:
            target_class_names = self.classes
        else:
            target_class_names = [str(i) for i in range(self.num_classes)]

        # Generate a classification report and log key metrics.
        metrics_dict = classification_report(
            y_true=all_labels, y_pred=all_preds,
            target_names=target_class_names, zero_division=0, output_dict=True
        )

        self.log('val/accuracy', metrics_dict['accuracy'], on_epoch=True, prog_bar=True)
        self.log('val/f1_score_macro', metrics_dict['macro avg']['f1-score'], on_epoch=True, prog_bar=True)
        self.log('val/precision_macro', metrics_dict['macro avg']['precision'], on_epoch=True)
        self.log('val/recall_macro', metrics_dict['macro avg']['recall'], on_epoch=True)

        for class_name in target_class_names:
            if class_name in metrics_dict:
                class_metrics = metrics_dict[class_name]
                sanitized_name = class_name.replace(" ", "_").lower()
                self.log(f'val/f1_score_class_{sanitized_name}', class_metrics['f1-score'], on_epoch=True)

        self.validation_step_outputs.clear()

    def on_test_epoch_start(self):
        self.test_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        imgs, labels, _, _, is_original = batch
        x = self.preprocess(imgs)
        y_hat = self(x)
        predictions = torch.argmax(y_hat, dim=1)
        self.test_step_outputs.append({'predictions': predictions.cpu(), 'labels': labels.cpu()})

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return

        # Aggregate labels and predictions from all test steps.
        ground_truth_labels = torch.cat([x['labels'] for x in self.test_step_outputs]).numpy()
        predicted_labels = torch.cat([x['predictions'] for x in self.test_step_outputs]).numpy()
        self.test_step_outputs.clear()

        # Use provided class names for the report, otherwise use integer indices.
        if hasattr(self, 'classes') and self.classes and len(self.classes) == self.num_classes:
            target_class_names = self.classes
        else:
            target_class_names = [str(i) for i in range(self.num_classes)]

        # Print the final classification report to the console.
        classification_summary_str = classification_report(
            y_true=ground_truth_labels, y_pred=predicted_labels,
            target_names=target_class_names, zero_division=0
        )
        print("\n--- Final Model Performance Evaluation: Test Set ---")
        print(classification_summary_str)
        
        # Generate a dictionary of metrics for detailed logging.
        metrics_dict = classification_report(
            y_true=ground_truth_labels, y_pred=predicted_labels,
            target_names=target_class_names, zero_division=0, output_dict=True
        )

        # Log metrics both for the global and per-class performance
        self.log('test/accuracy', metrics_dict['accuracy'])
        self.log('test/precision_macro', metrics_dict['macro avg']['precision'])
        self.log('test/recall_macro', metrics_dict['macro avg']['recall'])
        self.log('test/f1_score_macro', metrics_dict['macro avg']['f1-score'])

        for class_name in target_class_names:
            if class_name in metrics_dict:
                class_metrics = metrics_dict[class_name]
                sanitized_name = class_name.replace(" ", "_").lower()
                self.log(f'test/precision_class_{sanitized_name}', class_metrics['precision'])
                self.log(f'test/recall_class_{sanitized_name}', class_metrics['recall'])
                self.log(f'test/f1_score_class_{sanitized_name}', class_metrics['f1-score'])

        # Log confusion matrices to configured loggers (WandB, TensorBoard).
        if hasattr(self, 'classes') and self.classes:
            log_confusion_matrix_wandb(
                self.logger.__class__.__name__.lower(), self.logger.experiment,
                ground_truth_labels, predicted_labels, self.classes
            )
            tensorboard_logger = get_tensorboard_logger(self.trainer.loggers)
            if tensorboard_logger:
                log_confusion_matrix_tensorboard(
                    actual=ground_truth_labels, predicted=predicted_labels,
                    classes=self.classes, writer=tensorboard_logger
                )
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img, _, _, _ = batch
        x = self.preprocess(img)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def _set_model_classifier(self, weights_cls, num_classes):

        weights_cls_str = str(weights_cls)
        # Adapt the classifier based on the specific model architecture.
        if "ConvNeXt" in weights_cls_str:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Flatten(1),
                torch.nn.Linear(self.model.classifier[2].in_features, self.output_dim)
            )
        elif "EfficientNet" in weights_cls_str:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[1].in_features, self.output_dim)
            )
        elif "MobileNet" in weights_cls_str or "VGG" in weights_cls_str:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[0].in_features, self.output_dim)
            )
        elif "DenseNet" in weights_cls_str:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier.in_features, self.output_dim)
            )
        elif "MaxVit" in weights_cls_str:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(self.model.classifier[5].in_features, self.output_dim)
            )
        elif "ResNet" in weights_cls_str or "RegNet" in weights_cls_str or "GoogLeNet" in weights_cls_str:
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.fc.in_features, self.output_dim)
            )
        elif "Swin" in weights_cls_str:
            self.model.head = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.head.in_features, self.output_dim)
            )
        elif "ViT" in weights_cls_str:
            self.model.heads = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.hidden_dim, self.output_dim)
            )
        elif "SqueezeNet1_1" in weights_cls_str or "SqueezeNet1_0" in weights_cls_str:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Conv2d(512, self.output_dim, kernel_size=(1, 1), stride=(1, 1)),
                torch.nn.AvgPool2d(kernel_size=13, stride=1, padding=0)
            )

        # Append a final linear layer to map to the number of classes.
        self.model.lastLayer = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.output_dim, num_classes)
        )