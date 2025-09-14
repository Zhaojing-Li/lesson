#张量的基本操作

import torch
# ==============================
# 1. 张量的基本操作
# ==============================
print("="*50)
print("1. 张量基本操作示例")
print("="*50)

# 创建张量  不支持不规则的张量，稀疏张量表示当前张量中大多数元素是0
# 第一种方式是直接创建张量，设置其中的内容和数据类型  
# 第二种方式是随机创建张量，因为是用于ai，所以不在意初始内容，只关注结构
# 第三种定义的所有元素都是0
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)  # dtype 代表data的数据类型,默认是float32，目前大模型中使用的float16，可以理解为是量化之后的数据
y = torch.randn(2, 3)  # 正态分布随机张量
z = torch.zeros(3, 2)
print(f"创建张量:\n x={x}\n y={y}\n z={z}")


# print(f"x.shape={x.shape}\n y.shape={y.shape}\n z.shape={z.shape}")
# print(x.dim())  # 查看张量的维度
# print(x.size()) # 查看张量的形状
# print(x.dtype)  # 查看张量的数据类型
# print(x.device) # 查看张量的设备
# print(x.requires_grad) # 查看张量是否需要梯度
# print(x.grad) # 查看张量的梯度
# print(x.grad_fn) # 查看张量的梯度函数
# print(x.is_leaf) # 查看张量是否是叶子节点
# print(x.numel()) # 查看张量中元素的个数
# print(x.numel()) # 查看张量中元素的个数

# 索引和切片 索引是获取单个值，切片是获取区间值 
print("\n索引和切片:")
print("x[1, 2] =", x[1, 2].item())  # 获取标量值
print("x[:, 1:] =\n", x[:, 1:])

# 形状变换
reshaped = x.view(3, 2)  # 视图操作(不复制数据)  2*3转成3*2
transposed = x.t()       # 转置
squeezed = torch.randn(1, 3, 1).squeeze()  # 压缩维度
print(f"\n形状变换:\n 重塑后: {reshaped.shape}\n 转置后: {transposed.shape}\n 压缩后: {squeezed.shape}")


# 数学运算
add = x + y              # 逐元素加法
matmul = x @ transposed  # 矩阵乘法
sum_x = x.sum(dim=1)     # 沿维度求和
print(f"\n数学运算:\n 加法:\n{add}\n 矩阵乘法:\n{matmul}\n 行和: {sum_x}")


# 广播机制
a = torch.tensor([1, 2, 3])
b = torch.tensor([[10], [20]])
print(a.shape)
print(b.shape)
print(f"\n广播加法:\n{a + b}")

# 内存共享验证
view_tensor = x.view(6)
view_tensor[0] = 100
print("\n内存共享验证(修改视图影响原始张量):")
print(f"视图: {view_tensor}\n原始: {x}")