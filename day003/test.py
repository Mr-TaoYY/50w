import torch

# 1. 定义变量。requires_grad=True 是灵魂！它告诉 PyTorch：“盯紧这个变量，记录它所有的计算过程，我要对它求导！”
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)  # x 是输入数据，不需要求导，所以默认 False

# 2. 前向传播 (Forward)：计算出 z
z = w * x + b
print(f"前向计算结果 z = {z.item()}") # 应该等于 7.0

# 3. 反向传播 (Backward)：PyTorch 自动沿着刚才记录的计算图，反向求出所有 requires_grad=True 变量的梯度！
z.backward()

# 4. 查看梯度 (Gradients)
# .grad 属性里存放的就是求导结果
print(f"w 的梯度 (dz/dw): {w.grad.item()}") # 手推结果是 2，验证是否相等
print(f"b 的梯度 (dz/db): {b.grad.item()}") # 手推结果是 1，验证是否相等