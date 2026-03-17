import torch
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

# ====================== 1. 基础梯度下降 ======================
x = torch.tensor([5.0], requires_grad=True)
lr = 0.1
print("===== 梯度下降迭代过程 =====")
for i in range(20):
    y = x**2 + 2*x + 1
    y.backward()
    
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()
    print(f"迭代 {i+1}: x = {x.item():.4f}, loss = {y.item():.4f}")

print("\n最终结果:")
print(f"最优 x = {x.item():.4f}, 最小损失 y = {y.item():.4f}")

# ====================== 2. 损失曲线 ======================
loss = []
x = torch.tensor([5.0], requires_grad=True)

for i in range(20):
    y = x**2 + 2*x + 1
    loss.append(y.item())
    y.backward()
    with torch.no_grad():
        x -= 0.1 * x.grad
    x.grad.zero_()

plt.figure()
plt.plot(loss, marker='o')
plt.title("损失函数变化曲线")
plt.xlabel("迭代次数")
plt.ylabel("Loss 值")
plt.grid(True)
plt.show()

# ====================== 3. 优化路径 ======================
x_vals = np.linspace(-5, 5, 100)
y_vals = x_vals**2 + 2*x_vals + 1

x_path = []
x = torch.tensor([5.0], requires_grad=True)

for i in range(10):
    y = x**2 + 2*x + 1
    x_path.append(x.item())
    y.backward()
    with torch.no_grad():
        x -= 0.3 * x.grad
    x.grad.zero_()

plt.figure()
plt.plot(x_vals, y_vals, label="目标函数 $f(x)=x^2+2x+1$")
plt.scatter(x_path, [xx**2 + 2*xx + 1 for xx in x_path], color='r', s=80, label="优化路径")
plt.title("梯度下降优化路径")
plt.legend()
plt.grid(True)
plt.show()