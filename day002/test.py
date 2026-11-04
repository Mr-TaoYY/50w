import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# 1、索引（Indexing）
x = torch.arange(1,10,device=device).reshape(3,3)
print("原张量：\n",x)
print("第一行:", x[0])
print("第二行第三列:", x[1,2])
print("第一列:", x[:,0])
print("前两行后两列:\n", x[:2,1:])
print("大于5的元素:", x[x>5])

# 2. 广播 (Broadcasting)
a = torch.tensor([[1], [2], [3]], device=device)  # 形状 (3, 1)
b = torch.tensor([10, 20, 30], device=device)     # 形状 (3,)
print("广播相加结果:\n", a+b)

# 3. 矩阵乘法 (Matrix Multiplication)
m1 = torch.randn(2, 3, device=device)
m2 = torch.randn(3, 4, device=device)
result = m1 @ m2  # @ 等价于 torch.matmul(m1, m2)，形状 (2, 4)
print("矩阵乘法结果形状:", result.shape)