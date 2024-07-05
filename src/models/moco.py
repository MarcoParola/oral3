import pytorch_lightning as pl
import torch
import torchvision
from timm.models.vision_transformer import vit_base_patch32_224
from torch import nn
import copy

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule

class OralMOCOModule(pl.LightningModule):
    def __init__(self, weights, num_classes, output_dim, lr=1e-4, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.output_dim = 64
        
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = MoCoProjectionHead(512, 512, self.output_dim)
        
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(memory_bank_size=(4096, self.output_dim))


    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query
    
    def extract_features(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        features = self.projection_head(features)
        return features
    
    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key
    
    def _common_step(self, batch, batch_idx, stage):
        momentum = cosine_schedule(self.current_epoch, 20, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
        
        x_query, x_key = batch[0]
        query = self.forward(x_query)
        key = self.forward_momentum(x_key)
        
        loss = self.criterion(query, key)
        self.log(f"{stage}_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        shceduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.max_epochs)
        return [optim], [shceduler]
        #return optim