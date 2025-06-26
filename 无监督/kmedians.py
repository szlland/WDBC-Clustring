import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np


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

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def kmedians(X, n_clusters, max_iter=100):
    # 随机初始化中心
    idx = np.random.choice(len(X), n_clusters, replace=False)
    centers = X[idx]
    for _ in range(max_iter):
        # 计算每个样本到中心的曼哈顿距离
        dists = np.abs(X[:, np.newaxis] - centers).sum(axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([np.median(X[labels == i], axis=0) for i in range(n_clusters)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels, centers

# 用K-Medians聚类
labels, centers = kmedians(X_scaled, n_clusters=2)

# 计算准确率（考虑标签反转）
acc1 = accuracy_score(y, labels)
acc2 = accuracy_score(y, 1 - labels)
acc = max(acc1, acc2)
print(f"K-Medians聚类准确率: {acc:.4f}")

