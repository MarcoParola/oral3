import torch
import hydra
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.utils import load_features
import pandas as pd
import numpy as np
import seaborn as sns

def plot_clusters(features, labels, predictions):
    tsne = TSNE(n_components=2, random_state=0, perplexity=5)
    reduced_features = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=predictions, cmap='viridis')
    legend = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend)
    plt.title('t-SNE Visualization of KNN Predictions')
    plt.show()

def plot_heatmap(df, title):
    n_neighbors = sorted(df['n_neighbors'].unique())
    metrics = sorted(df['metric'].unique())
    heatmap_data = np.zeros((len(n_neighbors), len(metrics)))

    for i, nn in enumerate(n_neighbors):
        for j, metric in enumerate(metrics):
            accuracy = df[(df['n_neighbors'] == nn) & (df['metric'] == metric)]['Test Accuracy'].values
            if len(accuracy) > 0:
                heatmap_data[i, j] = accuracy[0]

    plt.imshow(heatmap_data, cmap='Greens', aspect='auto')
    plt.title(title)
    plt.xlabel('Metric')
    plt.ylabel('n_neighbors')
    plt.xticks(np.arange(len(metrics)), metrics)
    plt.yticks(np.arange(len(n_neighbors)), n_neighbors)
    plt.colorbar(label='Test Accuracy')
    for i in range(len(n_neighbors)):
        for j in range(len(metrics)):
            plt.text(j, i, f'{heatmap_data[i, j]:.2f}', ha='center', va='center', color='black')

@hydra.main(config_path='./config', config_name='config')
def main(cfg):

    # Recupera la directory di lavoro originale
    orig_cwd = hydra.utils.get_original_cwd()
    
    # Usa il percorso assoluto combinato con la directory originale
    features_path_anchor = os.path.join(orig_cwd, 'outputs/features/anchor/swin')
    features_path_test = os.path.join(orig_cwd, 'outputs/features/test/swin')
    
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

    #-----------------------------------------
    # Initialize variables to track the best parameters and corresponding accuracy
    best_params_test_accuracy = None
    best_test_accuracy = 0

    # Initialize lists to store all parameters and test accuracies
    all_params = []
    all_test_accuracies = []

    # Display all combinations tested with their respective test accuracies
    print("All combinations of parameters and their test accuracy scores:")
    for params in grid_search.cv_results_['params']:
        # Use the trained model to predict the test set
        knn.set_params(**params)
        knn.fit(X_train, y_train)
        y_pred_test = knn.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        print(f'Params: {params}, Test Accuracy: {accuracy_test:.4f}')

        # save the combination of parameters and their test accuracy
        all_params.append(params)
        all_test_accuracies.append(accuracy_test)
        
        # Track the best parameters based on test accuracy
        if accuracy_test > best_test_accuracy:
            best_test_accuracy = accuracy_test
            best_params_test_accuracy = params
    
    # Create a DataFrame to hold all test accuracies and parameters
    df = pd.DataFrame({
        'Test Accuracy': all_test_accuracies,
        'n_neighbors': [params['n_neighbors'] for params in all_params],
        'metric': [params['metric'] for params in all_params],
        'weights': [params['weights'] for params in all_params]
    })

    # Create separate DataFrames for 'uniform' and 'distance' weights
    df_uniform = df[df['weights'] == 'uniform']
    df_distance = df[df['weights'] == 'distance']

    # Create heatmap for 'uniform' weights
    plt.figure(figsize=(10, 8))
    plot_heatmap(df_uniform, 'Uniform Weights')

    # Create heatmap for 'distance' weights
    plt.figure(figsize=(10, 8))
    plot_heatmap(df_distance, 'Distance Weights')

    plt.show()

    #-----------------------------------------

    print(f'Best parameters based on test accuracy: {best_params_test_accuracy}')

    # Use the best parameters found on test set to calculate evaluation metrics
    knn.set_params(**best_params_test_accuracy)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

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

    #-----------------------------------------
    # Mostra la matrice di confusione come heatmap con matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues', aspect='auto')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(np.unique(y_test))), np.unique(y_test), rotation=45)
    plt.yticks(np.arange(len(np.unique(y_test))), np.unique(y_test))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(np.unique(y_test))):
        for j in range(len(np.unique(y_test))):
            plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')
    plt.tight_layout()
    plt.show()
    #-----------------------------------------

    features = torch.cat((torch.tensor(X_train), torch.tensor(X_test)), 0)
    labels = torch.cat((torch.tensor(y_train), torch.tensor(y_test)), 0)

    # Plot clusters
    plot_clusters(features, labels, best_knn.predict(features))

if __name__ == '__main__':
    main()
