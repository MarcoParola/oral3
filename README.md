# **Oral3**


This GitHub repo for the design analysis of different Case-Based Reasoning (CBR) systems supporting oral cancer diagnosis:
- DL
- DL-CBR
- IDL-CBR
- SSL-CBR
- FSL-CBR

TODO magari qui mettiamo una foto/una tabella/uno schema che renda l'idea.

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
- Informed deep learning
- Self supervised learning (cae lo farei rientrare in questa categoria)
- Few-shot learning


### CBR system running

