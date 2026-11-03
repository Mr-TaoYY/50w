import torch
import time

# # 检查系统参数
# print("="*30)
# print("PyTorch 版本:", torch.__version__)
# print("CUDA 是否可用:", torch.cuda.is_available())
#
# if torch.cuda.is_available():
#     print("识别到的显卡名称:", torch.cuda.get_device_name(0))
#     # 创建一个张量并放到 GPU 上
#     x = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
#     print("张量所在设备:", x.device)
#     print("张量运算测试:", x * 2)
# else:
#     print("⚠️ 警告：PyTorch 无法调用 CUDA，请检查安装的版本！")
# print("="*30)

# 创建两个超大矩阵
size = 10000
a = torch.randn(size, size, device='cuda')
b = torch.randn(size, size, device='cuda')

# 预热一下
torch.matmul(a, b)

# 计时算力测试
start_time = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize() # 等待 GPU 算完
end_time = time.time()

print(f"矩阵乘法计算耗时: {end_time - start_time:.4f} 秒")
print(f"结果矩阵大小: {c.shape}")
