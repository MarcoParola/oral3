import copy
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
import gc
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss.vicreg_loss import VICRegLoss

## The projection head is the same as the Barlow Twins one
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.transforms.vicreg_transform import VICRegTransform

class OralVICRegModule(pl.LightningModule):
    def __init__(self, weights, num_classes, output_dim, lr=1e-4, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.output_dim = output_dim
        
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = VICRegProjectionHead(
            input_dim=512,
            hidden_dim=2048,
            output_dim=self.output_dim,
            num_layers=2,
        )
        self.criterion = VICRegLoss()
        
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def extract_features(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
    
    def _common_step(self, batch, batch_idx, stage):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)

        # Log loss
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.max_epochs)
        return [optim], [scheduler]
