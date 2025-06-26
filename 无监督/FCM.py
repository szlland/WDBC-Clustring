import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
import skfuzzy as fuzz
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

# FCM聚类
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_scaled.T, c=2, m=2, error=0.005, maxiter=1000, init=None, seed=42
)

# 获取每个样本的聚类标签
labels = np.argmax(u, axis=0)

# 计算准确率（考虑标签反转）
acc1 = accuracy_score(y, labels)
acc2 = accuracy_score(y, 1 - labels)
acc = max(acc1, acc2)
print(f"FCM聚类准确率: {acc:.4f}")
