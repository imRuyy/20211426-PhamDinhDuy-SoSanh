import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
import skfuzzy as fuzz

# Tải dữ liệu Iris
iris = datasets.load_iris()
X = iris.data
y_true = iris.target  # Nhãn thực tế

# Giảm chiều dữ liệu xuống 2 chiều để dễ trực quan hóa
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 1. Phân cụm bằng K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# 2. Phân cụm bằng Fuzzy C-Means (FCM)
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(X.T, 3, 2, error=0.005, maxiter=1000)
y_fcm = np.argmax(u, axis=0)

# 3. Phân cụm bằng (AHC)
ahc = AgglomerativeClustering(n_clusters=3)
y_ahc = ahc.fit_predict(X)

# Vẽ các biểu đồ cho từng phương pháp
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Kết quả của K-Means
axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', s=50)
axs[0].set_title('K-Means')

# Kết quả của Fuzzy C-Means
axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_fcm, cmap='viridis', s=50)
axs[1].set_title('Fuzzy C-Means')

# Kết quả của AHC
axs[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y_ahc, cmap='viridis', s=50)
axs[2].set_title('AHC')

plt.show()
