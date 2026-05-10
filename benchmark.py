import torch
import time

# 检查 CUDA
if not torch.cuda.is_available():
    print("CUDA 不可用，请检查驱动挂载！")
    exit()

device = torch.device("cuda")
print(f"正在测试设备: {torch.cuda.get_device_name(0)}")

# 设置矩阵大小 (3000x3000)
size = 3000
A = torch.randn(size, size, device=device)
B = torch.randn(size, size, device=device)

# 预热（Warm-up），排除初始化耗时
print("正在预热 GPU...")
for _ in range(10):
    torch.mm(A, B)

# 开始计时
torch.cuda.synchronize() # 确保之前的操作已完成
start_time = time.time()

iterations = 50
print(f"开始执行 {iterations} 次矩阵乘法...")
for _ in range(iterations):
    torch.mm(A, B)

torch.cuda.synchronize() # 等待所有计算完成
end_time = time.time()

# 计算结果
avg_time = (end_time - start_time) / iterations
tflops = (2 * size**3) / (avg_time * 1e12)

print("-" * 30)
print(f"平均耗时: {avg_time*1000:.2f} ms")
print(f"估算算力: {tflops:.2f} TFLOPS")
print("-" * 30)
