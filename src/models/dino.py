import torch
import torchvision
import pytorch_lightning as pl
from torch import nn
import copy
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
import gc
from PIL import Image
import torchvision.transforms.functional as F

class OralDinoModule(pl.LightningModule):
    def __init__(self, weights, num_classes, output_dim, lr=1e-4, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        
        # Load DINO model backbone
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
        input_dim = backbone.embed_dim
        self.output_dim = output_dim

        # Student and teacher backbones and heads
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(input_dim, 512, 64, output_dim, freeze_last_layer=1)
        #self.student_head = DINOProjectionHead(input_dim, 2048, 256, self.output_dim, freeze_last_layer=1)
        
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, output_dim)
        #self.teacher_head = DINOProjectionHead(input_dim, 2048, 256, self.output_dim)
        
        # Deactivate gradients for teacher model
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)
        
        # DINO criterion
        self.criterion = DINOLoss(output_dim=output_dim, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        # Forward pass through student model
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        with torch.no_grad():
            # Forward pass through teacher model
            y = self.teacher_backbone(x).flatten(start_dim=1)
            z = self.teacher_head(y)
        return z

    def extract_features(self, x):
        # Extract features from student model
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z
    
    def _common_step(self, batch, batch_idx, stage):
        # Calculate momentum for teacher model
        momentum = cosine_schedule(self.current_epoch, 20, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        
        # Extract views from batch
        imgs, imgs2, ids, names = batch
        
        imgs = [img.to(self.device) for img in imgs]

        global_imgs = imgs[:2]
        
        teacher_out = [self.forward_teacher(img) for img in global_imgs]
        student_out = [self.forward(img) for img in imgs]
        
        # Calculate loss using DINOLoss
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        shceduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.max_epochs)
        return [optimizer], [shceduler]