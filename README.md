## Comparison of SNN and K-Means Algorithms
# Overview
This repository provides a comparison between Shared Nearest Neighbor (SNN) and K-Means clustering algorithms. The project includes data generation, preprocessing, custom implementations of the clustering algorithms, and visualization of the results. The goal is to highlight the strengths and weaknesses of each algorithm using Silhouette scores as the performance metric.

# Table of Contents
- Overview
- Data Generation
- Data Preprocessing
- Clustering Algorithms
- K-Means
- Simple Neural Network (SNN)
- Results
- Silhouette Scores
- Visualizations
- 2D Plots
- 3D Plots
- Conclusion
  
# Data Generation
We generate synthetic data using the make_classification function from sklearn.datasets. The dataset contains 650 samples with 2 informative features and no redundant features. Each class forms a single cluster.

from sklearn.datasets import make_classification

samples = 650
r_state = 10

X, y = make_classification(n_samples=samples,
                           n_features=2, n_informative=2,
                           n_redundant=0,
                           n_clusters_per_class=1,
                           random_state=r_state)
                           
# Data Preprocessing
The generated data is scaled using StandardScaler and split into training and test sets using train_test_split.


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.3, random_state=42)
Clustering Algorithms
K-Means
A custom implementation of the K-Means algorithm is used to cluster the data. The algorithm iterates over different numbers of clusters and calculates the Silhouette score to determine the optimal number of clusters.

def custom_kmeans(X, max_clusters=10, max_iters=100):
    best_score = -1
    best_n_clusters = 2
    best_labels = None

    for n_clusters in range(2, max_clusters + 1):
        centers = X[np.random.choice(range(len(X)), size=n_clusters, replace=False)]
        
        for _ in range(max_iters):
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        
        score = silhouette_score(X, labels)
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_labels = labels
    
    return best_labels, best_n_clusters, best_score
Simple Neural Network (SNN)
An SNN-like algorithm is implemented to cluster the data. The algorithm initializes weights as the means of the class centers and iteratively updates them based on the distance to the data points.


def simple_neural_network(X, y, max_iters=100, radius=1.0):
    n_clusters = 2
    best_score = -1
    best_labels = None

    weights = np.array([X[y == i].mean(axis=0) for i in range(n_clusters)])
    
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - weights, axis=2)
        closest_center_indices = np.argmin(distances, axis=1)
        
        for i in range(n_clusters):
            within_radius = np.where((closest_center_indices == i) & (distances[:, i] <= radius))
            if len(within_radius[0]) > 0:
                weights[i] = np.mean(X[within_radius], axis=0)
    
    labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - weights, axis=2), axis=1)
    score = silhouette_score(X, labels)
        
    if score > best_score:
        best_score = score
        best_labels = labels
    
    return best_labels, best_score
    
Results
Silhouette Scores
The Silhouette scores for both K-Means and SNN algorithms are calculated for the training and test sets.


train_labels_kmeans, best_n_clusters_kmeans_train, best_score_kmeans_train = custom_kmeans(X_train)
test_labels_kmeans, best_n_clusters_kmeans_test, best_score_kmeans_test = custom_kmeans(X_test)

radius = 0.5
train_labels_snn, best_score_snn_train = simple_neural_network(X_train, y_train, radius=radius)
test_labels_snn, best_score_snn_test = simple_neural_network(X_test, y_test, radius=radius)

print("K-Means Silhouette Score (Eğitim):", best_score_kmeans_train)
print("K-Means Silhouette Score (Test):", best_score_kmeans_test)
print("SNN Silhouette Score (Eğitim):", best_score_snn_train)
print("SNN Silhouette Score (Test):", best_score_snn_test)
Visualizations
2D Plots
The clustering results are visualized using 2D scatter plots.


plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
for class_value in range(best_n_clusters_kmeans_test):
    row_ix = np.where(test_labels_kmeans == class_value)
    plt.scatter(X_test[row_ix, 0], X_test[row_ix, 1], label=f'K-Means Class {class_value}')
plt.title('K-Means Clustering Results (Test Set)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(1, 3, 2)
for class_value in range(2):
    row_ix = np.where(test_labels_snn == class_value)
    plt.scatter(X_test[row_ix, 0], X_test[row_ix, 1], label=f'SNN Class {class_value}')
plt.title('SNN Clustering Results (Test Set)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(1, 3, 3)
for class_value in range(2):
    row_ix = np.where(y_test == class_value)
    plt.scatter(X_test[row_ix, 0], X_test[row_ix, 1], label=f'True Class {class_value}')
plt.title('True Classes (Test Set)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()
3D Plots
The clustering results are also visualized using 3D scatter plots.


fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
for class_value in range(best_n_clusters_kmeans_test):
    row_ix = np.where(test_labels_kmeans == class_value)
    ax1.scatter(X_test[row_ix, 0], X_test[row_ix, 1], zs=class_value, depthshade=True, label=f'K-Means Class {class_value}')
ax1.set_title('K-Means Clustering Results (Test Set)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Class')
ax1.legend()

ax2 = fig.add_subplot(132, projection='3d')
for class_value in range(2):
    row_ix = np.where(test_labels_snn == class_value)
    ax2.scatter(X_test[row_ix, 0], X_test[row_ix, 1], zs=class_value, depthshade=True, label=f'SNN Class {class_value}')
ax2.set_title('SNN Clustering Results (Test Set)')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_zlabel('Class')
ax2.legend()

ax3 = fig.add_subplot(133, projection='3d')
for class_value in range(2):
    row_ix = np.where(y_test == class_value)
    ax3.scatter(X_test[row_ix, 0], X_test[row_ix, 1], zs=class_value, depthshade=True, label=f'True Class {class_value}')
ax3.set_title('True Classes (Test Set)')
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.set_zlabel('Class')
ax3.legend()

plt.tight_layout()
plt.show()

Conclusion
This project compares the performance of K-Means and SNN algorithms using synthetic data. The custom implementations and visualizations provide insights into how these algorithms work and their effectiveness in clustering tasks. The Silhouette scores and plots help evaluate the clustering quality and demonstrate the differences between the algorithms.

Feel free to explore the code, experiment with different parameters, and analyze the results to gain a deeper understanding of clustering algorithms.
