from sklearn import datasets

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
print(y)
print(X)
from sklearn.preprocessing import StandardScaler

# 对特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 使用网格搜索找到最佳参数
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# 使用最佳参数进行预测
best_svc = grid_search.best_estimator_
from sklearn.decomposition import PCA

# 使用PCA将数据降维到2个主成分
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 在降维后的数据上重新训练最佳SVC模型
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)
best_svc.fit(X_train_pca, y_train_pca)

import numpy as np

# 创建一个网格以绘制决策边界
h = 0.02  # 网格步长
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
# 预测整个网格点的分类
Z = best_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 输出数据集
print("Dataset:")
print(iris.data)

# 输出α（拉格朗日乘子）
print("Alpha:")
print(best_svc.dual_coef_)

# 输出w和b
if best_svc.kernel == "linear":
    w = best_svc.coef_
    b = best_svc.intercept_
    print("w:", w)
    print("b:", b)

# 输出预测结果
y_pred = best_svc.predict(X_test_pca)
print("Predictions:")
print(y_pred)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 绘制散点图
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=ListedColormap(['red', 'green', 'blue']))
plt.title('Scatter Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# 绘制决策边界图
plt.subplot(122)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=ListedColormap(['red', 'green', 'blue']))
plt.title('Decision Boundary Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()