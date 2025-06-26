import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score

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


# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 用GMM聚类，设为2类（良性/恶性）
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X_scaled)
y_gmm = gmm.predict(X_scaled)

# GMM聚类标签与真实标签可能有0/1的反转，需要对齐
# 计算两种映射方式的准确率，取较大者
acc1 = accuracy_score(y, y_gmm)
acc2 = accuracy_score(y, 1 - y_gmm)
acc = max(acc1, acc2)

print(f"GMM聚类准确率: {acc:.4f}")

