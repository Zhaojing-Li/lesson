import torch

# ==============================
# 3. 线性代数
# ==============================
print("\n" + "=" * 50)
print("3. 线性代数操作示例")
print("=" * 50)

# 矩阵运算
A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# 基本运算
print(f"矩阵加法:\n{A + B}")
print(f"元素乘法:\n{A * B}")
print(f"矩阵乘法:\n{torch.mm(A, B)}")

# 高级运算
print(f"\n行列式: {torch.det(A):.2f}")
print(f"逆矩阵:\n{torch.inverse(A)}")

# 使用新的特征值计算方法
eigenvalues = torch.linalg.eigvals(A)
print(f"特征值:\n{eigenvalues}")

# 解线性方程组
# AX = B → X = A^{-1}B
X = torch.mm(torch.inverse(A), B)
print(f"\n解线性方程组 AX=B:\n{X}")

# 奇异值分解
U, S, Vh = torch.linalg.svd(A)
print(f"\n奇异值分解:")
print(f"U:\n{U}\nS:\n{torch.diag(S)}\nVh (共轭转置):\n{Vh}")

# 矩阵范数
print(f"\nFrobenius范数: {torch.linalg.matrix_norm(A, ord='fro'):.2f}")
print(f"谱范数: {torch.linalg.matrix_norm(A, ord=2):.2f}")