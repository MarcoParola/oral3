import torch
import hydra
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils import load_features

def plot_clusters(features, labels, predictions):
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)
    reduced_features = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=predictions, cmap='viridis')
    legend = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend)
    plt.title('t-SNE Visualization of KNN Predictions')
    plt.show()

@hydra.main(config_path='./config', config_name='config')
def main(cfg):

    # Recupera la directory di lavoro originale
    orig_cwd = hydra.utils.get_original_cwd()
    
    # Usa il percorso assoluto combinato con la directory originale
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/cae')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/cae')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/cae_bbox')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/cae_bbox')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/convnext')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/convnext')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/vit')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/vit')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/swin')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/swin')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/squeeze')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/squeeze')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/contrastive90')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/contrastive90')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/contrastive180')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/contrastive180')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/contrastive270')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/contrastive270')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/contrastive360')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/contrastive360')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/contrastive450')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/contrastive450')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/dino')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/dino')
    features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/vicreg')
    features_path_test = os.path.join(orig_cwd, 'outputs/features/test/vicreg')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/mae')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/mae')
    #features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/moco')
    #features_path_test = os.path.join(orig_cwd, 'outputs/features/test/moco')

    features_train, labels_train = load_features(features_path_anchor)
    features_test, labels_test = load_features(features_path_test)
    
    # min-max normalization column by column
    for i in range(features_train.shape[1]):
        features_train[:, i] = (features_train[:, i] - features_train[:, i].min()) / (features_train[:, i].max() - features_train[:, i].min() + 1e-8)
    
    for i in range(features_test.shape[1]):
        features_test[:, i] = (features_test[:, i] - features_test[:, i].min()) / (features_test[:, i].max() - features_test[:, i].min() + 1e-8)

    X_train = features_train
    y_train = labels_train
    X_test = features_test
    y_test = labels_test

    # Define the parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Initialize the KNN model
    knn = KNeighborsClassifier()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Train the model with GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best parameters and estimator
    best_knn = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f'Best parameters found: {best_params}')

    # Make predictions with the best model
    y_pred = best_knn.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')
    print(f'Confusion Matrix:\n {conf_matrix}')

    features = torch.cat((torch.tensor(X_train), torch.tensor(X_test)), 0)
    labels = torch.cat((torch.tensor(y_train), torch.tensor(y_test)), 0)

    # Plot clusters
    plot_clusters(features, labels, best_knn.predict(features))

if __name__ == '__main__':
    main()
