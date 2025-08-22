import torch

# 输入张量（需要计算梯度）
x = torch.tensor([5.0], requires_grad=True)

# 前向传播
a = x * 2       # 中间变量a = 10.0
a.add_(1)       # inplace操作：a += 1，此时a变为11.0（原始值10.0被覆盖）
b = a * a       # 中间变量b = 11.0 * 11.0 = 121.0
y = b * 3       # 输出y = 363.0

# 反向传播（计算dy/dx）
y.backward()
print(x.grad)
