import torch
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler



# 1. 加载WDBC数据集

column_names = [
    'id', 'diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# 用pandas读取数据
df = pd.read_csv(
    'breast+cancer+wisconsin+diagnostic/wdbc.data',
    header=None,
    names=column_names
)
 
# 取特征和标签
X = df.iloc[:, 2:].values  # 特征（去掉id和diagnosis）
y = (df['diagnosis'] == 'M').astype(int).values  # 标签：M为1，B为0

# 标准化特征 并 转为PyTorch张量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float)

# 3. K-means聚类实现
def kmeans(X, num_clusters, num_iters=100):
    # 随机初始化聚类中心
    indices = torch.randperm(X.size(0))[:num_clusters]
    centroids = X[indices]

    for i in range(num_iters):
        # 计算每个点到各个中心的距离
        distances = torch.cdist(X, centroids)
        # 分配每个点到最近的中心
        labels = distances.argmin(dim=1)
        # 更新中心
        new_centroids = torch.stack([X[labels == k].mean(dim=0) for k in range(num_clusters)])
        # 如果中心不再变化，提前结束
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    return labels, centroids

# 4. 聚类
num_clusters = 2
labels, centroids = kmeans(X_tensor, num_clusters)

# 5. 用PCA降维到2维，方便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. 可视化聚类结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels.numpy(), cmap='viridis')
plt.title('K-means Clustering Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('Ground Truth (Malignant=1, Benign=0)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

# 计算聚类准确率（考虑标签反转）
acc1 = accuracy_score(y, labels.numpy())
acc2 = accuracy_score(y, 1 - labels.numpy())
acc = max(acc1, acc2)
print(f"K-means聚类准确率: {acc:.4f}")
