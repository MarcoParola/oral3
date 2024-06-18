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

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

class OralDinoModule(pl.LightningModule):
    def __init__(self, weights, num_classes, lr=1e-4, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        
        # Load DINO model backbone
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        input_dim = backbone.embed_dim
        
        # Student and teacher backbones and heads
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(input_dim, 512, 64, 2048, freeze_last_layer=1)
        
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        
        # Deactivate gradients for teacher model
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)
        
        # DINO criterion
        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

        # DINO Transform
        self.transform = DINOTransform()
        
        # Last classification layers
        self.classification_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        self.heads = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 64)
        )

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
        #w = self.heads(z)
        return z
    
    def _common_step(self, batch, batch_idx, stage):
        # Calculate momentum for teacher model
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)
        
        # Extract views from batch
        #imgs, labels, ids, names = batch
        #print(imgs.shape)        

        imgs = batch[0]
        #imgs = self.transform(imgs)
        imgs = [img.to(self.device) for img in imgs]

        # Move images to device and force CPU fallback for resize
        #imgs = [img.cpu().to(self.device) for img in imgs]

        for img in imgs:
            if torch.isnan(img).any() or torch.isinf(img).any():
                print("NaN values detected")

        global_imgs = imgs[:2]
        
        # Forward pass through teacher and student models
        #teacher_out = self.forward_teacher(imgs)  # Use first two images for teacher
        #student_out = self.forward(imgs)  # Use first two images for student

        teacher_out = [self.forward_teacher(img) for img in global_imgs]
        student_out = [self.forward(img) for img in imgs]
        #print(f"Teacher output shape: {teacher_out.shape}")
        #print(f"Student output shape: {student_out.shape}")
        
        # Calculate loss using DINOLoss
        #loss = self.criterion([teacher_out], [student_out], epoch=self.current_epoch)
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        
        # Log loss
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True)

        free_memory()

        return loss

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-5)
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]
        