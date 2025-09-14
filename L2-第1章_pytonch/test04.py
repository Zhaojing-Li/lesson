import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# ==============================
# 综合应用：线性回归
# ==============================
print("\n" + "=" * 50)
print("综合应用: 线性回归")
print("=" * 50)

# 生成数据
torch.manual_seed(42)
X = torch.linspace(0, 10, 100).reshape(-1, 1)
print(X.shape)
true_weights = 2.5
true_bias = 1.0
y = true_weights * X + true_bias + torch.randn(X.size()) * 1.5


# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 训练设置
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 1000

# 训练循环
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 结果可视化
predicted = model(X).detach().numpy()
plt.scatter(X.numpy(), y.numpy(), label='Original data')
plt.plot(X.numpy(), predicted, 'r-', label='Fitted line')
plt.legend()
plt.title(f'Final weights: {model.linear.weight.item():.2f}, bias: {model.linear.bias.item():.2f}')
plt.show()