import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 1. 构造二维数据
X = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 7], [8, 6]])
y = np.array([0, 0, 0, 1, 1, 1])

# 2. 生成网格用于绘制决策边界
xx, yy = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))

# 3. 测试不同 K 值
for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(f"KNN 分类边界 (K = {k})")
    plt.grid(True)
    plt.show()