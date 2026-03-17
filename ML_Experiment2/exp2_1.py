import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 1. 构造样本数据
data = torch.tensor([1., 1., 0., 1., 0.])

# 2. 计算 MLE
p_mle = torch.mean(data)

# 3. 计算 MAP（Beta 先验 α=2, β=2）
alpha = 2
beta_val = 2
p_map = (torch.sum(data) + alpha - 1) / (len(data) + alpha + beta_val - 2)

print("="*50)
print(f"MLE 估计 p = {p_mle.item():.4f}")
print(f"MAP 估计 p = {p_map.item():.4f}")
print("="*50)

# 4. 可视化
data_np = np.array([1, 1, 0, 1, 0])
N = len(data_np)
sum_x = np.sum(data_np)

p = np.linspace(0, 1, 100)
likelihood = p**sum_x * (1-p)**(N-sum_x)
prior = beta.pdf(p, alpha, beta_val)
posterior = beta.pdf(p, sum_x + alpha, N - sum_x + beta_val)

plt.figure(figsize=(10, 5))
plt.plot(p, likelihood, label="Likelihood (似然)")
plt.plot(p, prior, label="Prior (先验)")
plt.plot(p, posterior, label="Posterior (后验)")
plt.axvline(p_mle.item(), color='r', linestyle='--', label="MLE")
plt.axvline(p_map.item(), color='g', linestyle='--', label="MAP")
plt.legend()
plt.title("MLE vs MAP 参数估计")
plt.grid(True)
plt.show()