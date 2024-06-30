import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import where
from sklearn.datasets import make_classification

# Veri
samples = 650
r_state = 10

X, y = make_classification(n_samples=samples,
                        n_features=2, n_informative=2,
                        n_redundant=0,
                        n_clusters_per_class=1,  # Her sınıf için bir küme olacak
                        random_state=r_state)

# Verileri ölçeklendir
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# Eğitim ve test setlerini oluştur
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.3, random_state=42)

def custom_kmeans(X, max_clusters=10, max_iters=100):
    best_score = -1
    best_n_clusters = 2
    best_labels = None

    for n_clusters in range(2, max_clusters + 1):
        # Başlangıç merkezlerini seç
        centers = X[np.random.choice(range(len(X)), size=n_clusters, replace=False)]
        
        for _ in range(max_iters):
            # Her bir veri noktası için en yakın merkezi bul
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centers, axis=2), axis=1)
            
            # Yeni merkezleri hesapla
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
            
            # Merkezlerin değişimini kontrol et
            if np.allclose(centers, new_centers):
                break
            
            centers = new_centers
        
        # Silhouette skorunu hesapla
        score = silhouette_score(X, labels)
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_labels = labels
    
    return best_labels, best_n_clusters, best_score

def simple_neural_network(X, y, max_iters=100, radius=1.0):
    n_clusters = 2  # 2 sınıfa ayır
    best_score = -1
    best_labels = None

    # Başlangıç ağırlıkları: sınıf merkezlerinin ortalaması olarak belirleyin
    weights = np.array([X[y == i].mean(axis=0) for i in range(n_clusters)])
    
    for _ in range(max_iters):
        # Her veri noktası için en yakın merkezi bul
        distances = np.linalg.norm(X[:, np.newaxis] - weights, axis=2)
        closest_center_indices = np.argmin(distances, axis=1)
        
        # Her merkezi güncelle
        for i in range(n_clusters):
            within_radius = np.where((closest_center_indices == i) & (distances[:, i] <= radius))
            if len(within_radius[0]) > 0:
                weights[i] = np.mean(X[within_radius], axis=0)
    
    # Her bir veri noktası için en yakın merkezi bul
    labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - weights, axis=2), axis=1)
    
    # Silhouette skorunu hesapla
    score = silhouette_score(X, labels)
        
    if score > best_score:
        best_score = score
        best_labels = labels
    
    return best_labels, best_score

# Özel K-Means algoritmasıyla kümeleri bul
train_labels_kmeans, best_n_clusters_kmeans_train, best_score_kmeans_train = custom_kmeans(X_train)
test_labels_kmeans, best_n_clusters_kmeans_test, best_score_kmeans_test = custom_kmeans(X_test)

# SNN benzeri algoritma ile kümeleri bul
radius = 0.5  # Daire yarıçapı
train_labels_snn, best_score_snn_train = simple_neural_network(X_train, y_train, radius=radius)
test_labels_snn, best_score_snn_test = simple_neural_network(X_test, y_test, radius=radius)

# Silhouette skorlarını yazdır
print("K-Means Silhouette Score (Eğitim):", best_score_kmeans_train)
print("K-Means Silhouette Score (Test):", best_score_kmeans_test)
print("SNN Silhouette Score (Eğitim):", best_score_snn_train)
print("SNN Silhouette Score (Test):", best_score_snn_test)

# Eğitim ve test setleri için Silhouette skorlarını listelere ekle
silhouette_scores_train = [best_score_kmeans_train, best_score_snn_train]
silhouette_scores_test = [best_score_kmeans_test, best_score_snn_test]

# Kümeleme algoritmalarının adları
algorithms = ['K-Means', 'SNN']

# Eğitim ve test setleri için Silhouette skorlarını görselleştir
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(algorithms, silhouette_scores_train, color=['blue', 'green'])
plt.title('Silhouette Score (Eğitim)')
plt.ylabel('Silhouette Score')

plt.subplot(1, 2, 2)
plt.bar(algorithms, silhouette_scores_test, color=['blue', 'green'])
plt.title('Silhouette Score (Test)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# YAŞ_GRUBU ve HB sütunlarını ayrı değişkenlere atayalım
yas_grubu_train = X_train[:, 0]
hb_train = X_train[:, 1]

yas_grubu_test = X_test[:, 0]
hb_test = X_test[:, 1]

# K-Means ve SNN algoritmalarının kümeleme sonuçlarını kullanarak nokta grafikleri oluştur

# 2 Boyutlu grafikleri çiz
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
for class_value in range(best_n_clusters_kmeans_test):
    row_ix = np.where(test_labels_kmeans == class_value)
    plt.scatter(yas_grubu_test[row_ix], hb_test[row_ix], label=f'K-Means Sınıf {class_value}')
plt.title('K-Means Kümeleme Sonuçları (Test Seti)')
plt.xlabel('Yaş Grubu')
plt.ylabel('HB')
plt.legend()

plt.subplot(1, 3, 2)
for class_value in range(2):  # 2 sınıf
    row_ix = np.where(test_labels_snn == class_value)
    plt.scatter(yas_grubu_test[row_ix], hb_test[row_ix], label=f'SNN Sınıf {class_value}')
plt.title('SNN Kümeleme Sonuçları (Test Seti)')
plt.xlabel('Yaş Grubu')
plt.ylabel('HB')
plt.legend()

plt.subplot(1, 3, 3)
for class_value in range(2):  # Gerçek sınıf sayısını doğru ayarlayın
    row_ix = np.where(y_test == class_value)
    plt.scatter(yas_grubu_test[row_ix], hb_test[row_ix], label=f'Gerçek Sınıf {class_value}')
plt.title('Gerçek Sınıflar (Test Seti)')
plt.xlabel('Yaş Grubu')
plt.ylabel('HB')
plt.legend()

plt.tight_layout()
plt.show()

# 3 Boyutlu grafikleri çiz
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
for class_value in range(best_n_clusters_kmeans_test):
    row_ix = np.where(test_labels_kmeans == class_value)
    ax1.scatter(yas_grubu_test[row_ix], hb_test[row_ix], zs=class_value, depthshade=True, label=f'K-Means Sınıf {class_value}')
ax1.set_title('K-Means Kümeleme Sonuçları (Test Seti)')
ax1.set_xlabel('Yaş Grubu')
ax1.set_ylabel('HB')
ax1.set_zlabel('Sınıf')
ax1.legend()

ax2 = fig.add_subplot(132, projection='3d')
for class_value in range(2):  # 2 sınıf
    row_ix = np.where(test_labels_snn == class_value)
    ax2.scatter(yas_grubu_test[row_ix], hb_test[row_ix], zs=class_value, depthshade=True, label=f'SNN Sınıf {class_value}')
ax2.set_title('SNN Kümeleme Sonuçları (Test Seti)')
ax2.set_xlabel('Yaş Grubu')
ax2.set_ylabel('HB')
ax2.set_zlabel('Sınıf')
ax2.legend()

ax3 = fig.add_subplot(133, projection='3d')
for class_value in range(2):  # Gerçek sınıf sayısını doğru ayarlayın
    row_ix = np.where(y_test == class_value)
    ax3.scatter(yas_grubu_test[row_ix], hb_test[row_ix], zs=class_value, depthshade=True, label=f'Gerçek Sınıf {class_value}')
ax3.set_title('Gerçek Sınıflar (Test Seti)')
ax3.set_xlabel('Yaş Grubu')
ax3.set_ylabel('HB')
ax3.set_zlabel('Sınıf')
ax3.legend()

plt.tight_layout()
plt.show()
