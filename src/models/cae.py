import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Trainer, loggers
from torch.optim import Adam
import numpy as np
import os
import hydra


class Autoencoder(pl.LightningModule):
    def __init__(self, ae_params, output_dim, lr=100e-4, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()

 
        self.ae_params = ae_params
        self.output_dim = output_dim
        self.lr = lr
        self.max_epochs = max_epochs


        self.encoder = nn.Sequential(
            # input 3 x 224 x 224
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),  # -> 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            # input 32 x 112 x 112
            nn.Conv2d(32, 64, 4, 2, 1, bias=False), # -> 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            # input 64 x 56 x 56
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # -> 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
            # input 128 x 28 x 28
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # -> 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
            # input 256 x 14 x 14
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), # -> 512 x 7 x 7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),
            # input 512 x 7 x 7
            nn.Conv2d(512, 1024, 3, 2, 1, bias=False), # -> 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(True),
            # input 1024 x 4 x 4
            nn.Conv2d(1024, 224, 4, 1, 0, bias=False), # -> 224 x 1 x 1
            nn.BatchNorm2d(224),
            nn.LeakyReLU(True)
        )

        self.decoder = nn.Sequential(
            # input 224 x 1 x 1
            nn.ConvTranspose2d(224, 1024, 4, 1, 0, bias=False), # -> 1024 x 4 x 4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # input 1024 x 4 x 4
            nn.ConvTranspose2d(1024, 512, 3, 2, 1, bias=False), # -> 512 x 7 x 7
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # input 512 x 7 x 7
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), # -> 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # input 256 x 14 x 14
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # -> 128 x 28 x 28
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # input 128 x 28 x 28
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # -> 64 x 56 x 56
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # input 64 x 56 x 56
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False), # -> 32 x 112 x 112
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # input 32 x 112 x 112
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False), # -> 3 x 224 x 224
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def extract_features(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        return self._common_step(batch, batch_idx, "test")

    def _common_step(self, batch, batch_idx, stage):
        imgs, imgs2, image_id, image_name = batch
        y_hat = self(imgs)
        loss = F.mse_loss(y_hat, imgs2)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
        return loss
        


@hydra.main(config_path='../../config', config_name='config')
def main(cfg):

    model = Autoencoder(cfg.ae, cfg.train.lr, cfg.train.max_epochs)

    # define random input
    x = torch.randn(2, 3, cfg.dataset.resize, cfg.dataset.resize)

    print('input shape:', x.shape)
    print('output shape:', model(x).shape)
    print('encoding shape:', model.extract_features(x).shape)

if __name__ == '__main__':
    main()