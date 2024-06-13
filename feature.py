import torch
import pytorch_lightning
import omegaconf

from sklearn.metrics import classification_report
from src.data.few_shot_classification.datamodule import OralFewShotDataModule
from src.models.few_shot_classification import OralFewShotClassifierModule

from src.data.contrastive_classification.datamodule import OralContrastiveDataModule
from src.models.cae import Autoencoder
from src.data.autoencoder.datamodule import OralAutoencoderDataModule
from src.data.masked_classification.datamodule import OralClassificationMaskedDataModule
from src.data.segmentation.datamodule import OralSegmentationDataModule
from src.models.classification import OralClassifierModule
from src.models.contrastive_classification import OralContrastiveClassifierModule
from src.models.saliency_classification import OralSaliencyClassifierModule
from delete.classificationKNN import OralClassifierModuleKNN
from src.data.classification.datamodule import OralClassificationDataModule
from src.data.classification.dataset import OralClassificationDataset
from src.data.saliency_classification.datamodule import OralClassificationSaliencyDataModule
from src.data.saliency_classification.dataset import OralClassificationSaliencyDataset
from src.models.segmentation import FcnSegmentationNet, DeeplabSegmentationNet
from src.saliency.grad_cam import OralGradCam
from src.saliency.lime import OralLime
from src.saliency.shap import OralShap
import hydra
import os
import tqdm
from src.utils import *
from src.log import get_loggers


def predict(trainer, model, data, saliency_map_flag, task, classification_mode):

    if task == 'c' or task == 'classification':
        if classification_mode == 'cae':
            #features_path = os.path.join('./outputs/features/anchor', 'cae')
            #features_path = os.path.join('./outputs/features/test', 'cae')
            #features_path = os.path.join('./outputs/features/anchor', 'cae_anchor')
            #features_path = os.path.join('./outputs/features/test', 'cae_anchor')
            #features_path = os.path.join('./outputs/features/anchor', 'cae_bbox')
            #features_path = os.path.join('./outputs/features/test', 'cae_bbox')
            #features_path = os.path.join('./outputs/features/anchor', 'cae_anchor_bbox')
            features_path = os.path.join('./outputs/features/test', 'cae_anchor_bbox_prova')
            os.makedirs(features_path, exist_ok=True)

            # for with enumerate and tqdm with message
            for j, batch in enumerate(tqdm.tqdm(data.test_dataloader(), desc='Extracting features')):
                images, labels, categories, image_name = batch
                images = images.to(model.device)
                labels = labels.to(model.device)
                with torch.no_grad():
                    # get the features
                    features = model.extract_features(images)
                    #print(features.shape)
                    features = features.squeeze()
                    #print(features.shape)
                    
                    lbl = categories[0].item()
                    #print(lbl)

                    # save the features and labels
                    torch.save(features, os.path.join(features_path, f'features_{j}-{lbl}.pt'))
                    torch.save(labels, os.path.join(features_path, f'labels_{j}-{lbl}.pt'))

        elif classification_mode=='whole':
            #features_path = os.path.join('./outputs/features/anchor', 'convnext')
            #features_path = os.path.join('./outputs/features/test', 'convnext')
            #features_path = os.path.join('./outputs/features/anchor', 'vit')
            #features_path = os.path.join('./outputs/features/test', 'vit')
            #features_path = os.path.join('./outputs/features/anchor', 'swin')
            #features_path = os.path.join('./outputs/features/test', 'swin')
            #features_path = os.path.join('./outputs/features/anchor', 'squeeze')
            features_path = os.path.join('./outputs/features/test', 'squeeze')
            os.makedirs(features_path, exist_ok=True)

            # for with enumerate and tqdm with message
            for j, batch in enumerate(tqdm.tqdm(data.test_dataloader(), desc='Extracting features')):
                images, categories, image_id, image_name = batch
                imgs = images.to(model.device)
                labels = images.to(model.device)
                with torch.no_grad():
                    # get the features
                    features = model.extract_features(imgs)
                    if 'classifier' in features.keys():
                        features = features['classifier']
                    elif 'head' in features.keys():
                        features = features['head']
                    elif 'heads' in features.keys():
                        features = features['heads']
                    #print(features.shape)
                    features = features.squeeze()
                    #print(features.shape)
                    lbl = categories[0].item()
                    #print(lbl)

                    # save the features and labels
                    torch.save(features, os.path.join(features_path, f'features_{j}-{lbl}.pt'))
                    torch.save(labels, os.path.join(features_path, f'labels_{j}-{lbl}.pt'))

        elif classification_mode=='contrastive':
            #features_path = os.path.join('./outputs/features/anchor', 'contrastive90')
            #features_path = os.path.join('./outputs/features/test', 'contrastive90')
            #features_path = os.path.join('./outputs/features/anchor', 'contrastive180')
            #features_path = os.path.join('./outputs/features/test', 'contrastive180')
            #features_path = os.path.join('./outputs/features/anchor', 'contrastive270')
            #features_path = os.path.join('./outputs/features/test', 'contrastive270')
            #features_path = os.path.join('./outputs/features/anchor', 'contrastive360')
            #features_path = os.path.join('./outputs/features/test', 'contrastive360')
            #features_path = os.path.join('./outputs/features/anchor', 'contrastive450')
            features_path = os.path.join('./outputs/features/test', 'contrastive450')
            os.makedirs(features_path, exist_ok=True)

            # for with enumerate and tqdm with message
            for j, batch in enumerate(tqdm.tqdm(data.test_dataloader(), desc='Extracting features')):
                images, categories, image_id, image_name, positive, negative, positive_image, negative_image = batch
                imgs = images.to(model.device)
                labels = images.to(model.device)
                with torch.no_grad():
                    # get the features
                    features = model.extract_features(imgs)
                    features = features['scratch']
                    #print(features.shape)
                    features = features.squeeze()
                    #print(features.shape)
                    lbl = categories[0].item()
                    #print(lbl)

                    # save the features and labels
                    torch.save(features, os.path.join(features_path, f'features_{j}-{lbl}.pt'))
                    torch.save(labels, os.path.join(features_path, f'labels_{j}-{lbl}.pt'))

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    # this main load a checkpoint saved and perform test on it

    # save the passed version number before overwriting the configuration with training configuration
    version = str(cfg.checkpoint.version)
    # save the passed saliency map generation method before overwriting the configuration with training configuration
    saliency_map_method = cfg.generate_map
    # save the passed logging method before overwriting the configuration with training configuration
    loggers = get_loggers(cfg)
    # find the hydra_run_timestamp.txt file
    f = open('./logs/oral/version_' + version + '/hydra_run_timestamp.txt', "r")
    # read the timestamp inside hydra_run_timestamp.txt
    timestamp = f.read()
    # use the timestamp to build the path to hydra configuration
    path = './outputs/' + timestamp + '/.hydra/config.yaml'
    # load the configuration used during training
    cfg = omegaconf.OmegaConf.load(path)

    # to test is needed: trainer, model and data
    # trainer
    trainer = pytorch_lightning.Trainer(
        logger=loggers,
        # callbacks=callbacks,  shouldn't need callbacks in test
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs,
        # gradient_clip_val=0.1,
        # gradient_clip_algorithm="value"
    )

    model = None
    data = None

    print (cfg.task)

    if cfg.task == 'c' or cfg.task == 'classification' or cfg.task == 'f' or cfg.task == 'features':
        # whole classification
        if cfg.classification_mode == 'whole':
            # model
            # get the model already trained from checkpoints, default checkpoint is version_0, otherwise specify by cli
            model = OralClassifierModule.load_from_checkpoint(get_last_checkpoint(version))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            model.to(device)

            # data
            train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
            data = OralClassificationDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                #test="data/test.json",
                #test="data/few_shot_dataset.json",
                batch_size=1,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform
            )

        # contrastive classification
        elif cfg.classification_mode == 'contrastive':
            model = OralContrastiveClassifierModule.load_from_checkpoint(get_last_checkpoint(version))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            model.to(device)

            # data
            train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
            data = OralContrastiveDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                #test="data/test.json",
                #test="data/few_shot_dataset.json",
                batch_size=1,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform
            )
        
        # few shot classification
        elif cfg.classification_mode == 'few_shot':
            model = OralFewShotClassifierModule.load_from_checkpoint(get_last_checkpoint(version))
            model.eval()

            # data
            train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
            data = OralFewShotDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform
            )
        

        elif cfg.classification_mode == 'saliency':
            model = OralSaliencyClassifierModule.load_from_checkpoint(get_last_checkpoint(version))
            model.eval()

            # data
            train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
            data = OralClassificationSaliencyDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform
            )

        elif cfg.classification_mode == 'masked':
            model = OralClassifierModule.load_from_checkpoint(get_last_checkpoint(version))
            model.eval()

            train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
            data = OralClassificationMaskedDataModule(
                sgm_type=cfg.sgm_type,
                segmenter=cfg.model_seg,
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform
            )

        elif cfg.classification_mode == 'cae':
            model = Autoencoder.load_from_checkpoint(get_last_checkpoint(version))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            model.to(device)

            train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
            data = OralAutoencoderDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test="data/test.json",
                #test="data/few_shot_dataset.json",
                batch_size=1,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform
            )
        
        '''# classification with knn
        elif cfg.classification_mode == 'knn':
            model = OralClassifierModuleKNN.load_from_checkpoint(get_last_checkpoint(version))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            model.to(device)

            # data
            train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
            data = OralClassificationKNNDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=1,
                #batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform
            )
        '''

    elif cfg.task == 's' or cfg.task == 'segmentation':
        train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
        data = OralSegmentationDataModule(
            train=cfg.dataset.train,
            val=cfg.dataset.val,
            test=cfg.dataset.test,
            batch_size=cfg.train.batch_size,
            train_transform=train_img_tranform,
            val_transform=val_img_tranform,
            test_transform=test_img_tranform,
            transform=img_tranform
        )
        if cfg.model_seg == 'fcn':
            model = FcnSegmentationNet.load_from_checkpoint(get_last_checkpoint(version))
            model.sgm_type = cfg.sgm_type
        elif cfg.model_seg == 'deeplab':
            model = DeeplabSegmentationNet.load_from_checkpoint(get_last_checkpoint(version))
            model.sgm_type = cfg.sgm_type

        model.eval()

    predict(trainer, model, data, saliency_map_method, cfg.task, cfg.classification_mode)




if __name__ == "__main__":
    main()
