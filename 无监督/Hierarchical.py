import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score, confusion_matrix

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

df = pd.read_csv(
    'breast+cancer+wisconsin+diagnostic/wdbc.data',
    header=None,
    names=column_names
)

# 取特征和标签
X = df.iloc[:, 2:].values
y = (df['diagnosis'] == 'M').astype(int).values

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 层次聚类
clustering = AgglomerativeClustering(n_clusters=2)
labels = clustering.fit_predict(X_scaled)

# 计算准确率（考虑标签反转）
acc1 = accuracy_score(y, labels)
acc2 = accuracy_score(y, 1 - labels)
acc = max(acc1, acc2)
print(f"Hierarchical聚类准确率: {acc:.4f}")

