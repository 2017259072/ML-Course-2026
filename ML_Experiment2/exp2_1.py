import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# MLE 与 MAP 参数计算
data = torch.tensor([1., 1., 0., 1., 0.])
p_mle = torch.mean(data)

alpha = 2
beta_val = 2
p_map = (torch.sum(data) + alpha - 1) / (len(data) + alpha + beta_val - 2)

print("MLE =", p_mle.item())
print("MAP =", p_map.item())

# MLE 与 MAP 参数计算
data_np = np.array([1, 1, 0, 1, 0])
N = len(data_np)
sum_x = np.sum(data_np)
p = np.linspace(0, 1, 100)

likelihood = p ** sum_x * (1 - p) ** (N - sum_x)
prior = beta.pdf(p, 2, 2)
posterior = p ** (sum_x + 1) * (1 - p) ** (N - sum_x + 1)

p_mle_np = sum_x / N
p_map_np = (sum_x + 1) / (N + 2)

plt.plot(p, likelihood, label="Likelihood")
plt.plot(p, prior, label="Prior")
plt.plot(p, posterior, label="Posterior")
plt.axvline(p_mle_np, color='r', linestyle='--', label="MLE")
plt.axvline(p_map_np, color='g', linestyle='--', label="MAP")

plt.legend()
plt.title("MLE vs MAP Estimation")
plt.grid(True)
plt.show()