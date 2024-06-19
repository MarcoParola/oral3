# **Documentation**


## Installation

To install the project, clone the repository and get the necessary dependencies:
```sh
git clone https://github.com/MarcoParola/oral3.git
cd oral3
```

Create the virtualenv (you can also use conda) and install the dependencies of *requirements.txt*

```
python -m venv env
env/Scripts/activate
python -m pip install -r requirements.txt
mkdir data
```
Then you can download the oral coco-dataset (both images and json file) from [TODO-put-link]. Copy them into `data` folder and unzip the file `oral1.zip`.

Next, create a new project on Weights & Biases named `oral3`. Edit `entity` parameter in `config.yaml` by sett. Log in and paste your API key when prompted.
```
wandb login
```

## Usage
Regarding the usage of this repo, in order to reproduce the experiments, we organize the workflow in three steps: (i) data preparation and visualization, (ii) case base generation via DL, and (iii) CBR system running via kNN algorithm.

### Data preparation
Due to the possibility of errors in the dataset, such as missing images, run the check-dataset.py script to detect such errors. Returns the elements to be removed from the json file (this can be done manually or via a script).
```
python -m scripts.check-dataset --dataset data\coco_dataset.json
```
In this work, the dataset was annotated with more labels than necessary. Some are grouped under more general labels. To aggregate all the labels of the three diseases studied in this work, the following script is executed. In addition, we split the dataset with the holdout method.
```
python -m scripts.simplify-dataset --folder data
python -m scripts.split-dataset --folder data
```

You can use the `dataset-stats.py` script to print the class occurrences for each dataset.
```
python -m scripts.dataset-stats --dataset data\dataset.json # entire dataset
python -m scripts.dataset-stats --dataset data\train.json # training set
python -m scripts.dataset-stats --dataset data\test.json # test set
```

Use the following command to visualize the dataset bbox distribution: 
```
python -m scripts.plot-distribution --dataset data/dataset.json
```

### Case base generation via DL
You can use different DL strategies to create a feature embedding representation of your images for the base of CBR system:
- Deep learning
- Supervised learning
- Self supervised learning (cae lo farei rientrare in questa categoria)
- Few-shot learning


### CBR system running
#### Deep Learning
Classification on the whole dataset:
- Train CNN classifier on the whole dataset
- Test CNN classifier on the whole dataset
Specify the pre-trained classification model by setting `model.weights`.`classification_mode=whole` specifies we are solving the classification without exploiting the segment information.
```
# TRAIN classifier on whole images
python train.py task=c classification_mode=whole model.weights=ConvNeXt_Small_Weights.DEFAULT 

# TEST classifier whole images
python test.py task=c classification_mode=whole checkpoint.version=123
```

#### Supervised Learning
Classification on the whole dataset:
- Train CNN classifier on the whole dataset
- Test CNN classifier on the whole dataset
- Feature extraction on the anchor test and the test set
- KNN fit on the anchor set and KNN test on the test set
Specify the pre-trained classification model by setting `model.weights`.`classification_mode=whole` specifies we are solving the classification without exploiting the segment information.
```
# TRAIN classifier on whole images
python train.py task=c classification_mode=whole model.weights=ConvNeXt_Small_Weights.DEFAULT 

# TEST classifier whole images
python test.py task=c classification_mode=whole checkpoint.version=123
```

To perform feature extraction from the classifier, it is necessary to execute the following command, specifying the `feature_path` related to the anchor image features and the test set features within the `feature.py` file.
```
# FEATURE EXTRACTION from the classifier
python feature.py task=c classification_mode=whole checkpoint.version=123 
```

Then, to perform the fit and test of the KNN, it is necessary to specify the `features_path_anchor` and `features_path_test` within the `knn.py` file.
```
# FIT knn on anchor images, TEST knn on test images
python knn.py
```

#### Self supervised learning
Classification on the whole dataset:
- Train CAE on the whole dataset
- Feature extraction on the anchor test and the test set
- KNN fit on the anchor set and KNN test on the test set
It is possible to set in the `src/data1/autoencoder/dataset.py` whether to process the whole image or the bounding box.
`classification_mode=cae` specifies we are solving the classification without exploiting the segment information.
```
# TRAIN CAE on whole images
python train.py task=c classification_mode=cae model.weights=Scratch.DEFAULT
```

To perform feature extraction from the CAE, it is necessary to execute the following command, specifying the `feature_path` related to the anchor image features and the test set features within the `feature.py` file.
```
# FEATURE EXTRACTION from the CAE
python feature.py task=c classification_mode=cae checkpoint.version=123 
```

Then, to perform the fit and test of the KNN, it is necessary to specify the `features_path_anchor` and `features_path_test` within the `knn.py` file.
```
# FIT knn on anchor images, TEST knn on test images
python knn.py
```

#### Few-shot learning
Classification on the anchor dataset:
- Train CAE on the anchor dataset
- Feature extraction on the anchor test and the test set
- KNN fit on the anchor set and KNN test on the test set
For the experiment related to few-shot learning, first it is necessary to generate the dataset.
```
python scripts/split-fsl-dataset.py --folder data 
python scripts/fsl_test_generation.py
```

It is possible to set in the `src/data1/autoencoder/dataset.py` whether to process the whole image or the bounding box.
`classification_mode=cae` specifies we are solving the classification without exploiting the segment information.
```
# TRAIN CAE on anchor images
python train.py task=c classification_mode=cae model.weights=Scratch.DEFAULT
```

To perform feature extraction from the CAE, it is necessary to execute the following command, specifying the `feature_path` related to the anchor image features and the test set features within the `feature.py` file.
```
# FEATURE EXTRACTION from the CAE
python feature.py task=c classification_mode=cae checkpoint.version=123 
```

Then, to perform the fit and test of the KNN, it is necessary to specify the `features_path_anchor` and `features_path_test` within the `knn.py` file.
```
# FIT knn on anchor images, TEST knn on test images
python knn.py
```

#### Contrastive learning ?
Classification on the contrastive dataset:
- Train CNN classifier on the contrastive dataset
- Test CNN classifier on the contrastive dataset
- Feature extraction on the anchor test and the test set
- KNN fit on the anchor set and KNN test on the test set
For the experiment related to informed deep learning, first it is necessary to generate the dataset, specifying the number of triplets `num_category` for each category.
```
python scripts/prova_contrastive.py  --num_category 30 
```

After that, perform the split of the dataset.
```
python -m scripts.split-contrastive-dataset --folder data
```

You can use the `dataset-contrastive-stats.py` script to print the class occurrences for each dataset.
```
python -m scripts.dataset-contrastive-stats --dataset data\contrastive_train_min.json # training set
python -m scripts.dataset-contrastive-stats --dataset data\contrastive_test_min.json # test set
python -m scripts.dataset-contrastive-stats --dataset data\contrastive_val_min.json # validation set
```

Specify the scratch classification model by setting `model.weights`.`classification_mode=contrastive` specifies we are solving the classification without exploiting the segment information.
```
# TRAIN classifier on whole images
python train.py task=c classification_mode=contrastive model.weights=Scratch.DEFAULT

# TEST classifier on whole images
python test.py task=c classification_mode=contrastive checkpoint.version=123
```

To perform feature extraction from the classifier, it is necessary to execute the following command, specifying the `feature_path` related to the anchor image features and the test set features within the `feature.py` file.
```
# FEATURE EXTRACTION from the classifier
python feature.py task=c classification_mode=whole checkpoint.version=123 
```

Then, to perform the fit and test of the KNN, it is necessary to specify the `features_path_anchor` and `features_path_test` within the `knn.py` file.
```
# FIT knn on anchor images, TEST knn on test images
python knn.py
```