import torch
from add2 import add2  # 导入我们定义的包
n = 1024*1024  # 矩阵维度

a = torch.randn(n)
b = torch.randn(n)
c = torch.zeros(n)

print("a:\n",a)
print("\nb:\n",b)

# 将变量传送到GPU显存
a = a.cuda()
b = b.cuda()
c = c.cuda()

import time

# 测试自定义 kernel 的时间
torch.cuda.synchronize()
start = time.time()
for _ in range(10000):
    add2(a, b, c, n)
torch.cuda.synchronize()
end = time.time()
print(f"custom kernel time (1000 runs): {(end - start)*1000:.3f} ms")

# 测试 PyTorch 的时间
torch.cuda.synchronize()
start = time.time()
for _ in range(10000):
    d = a + b
torch.cuda.synchronize()
end = time.time()
print(f"torch add time (1000 runs): {(end - start)*1000:.3f} ms")

# 验证结果一致
print("\nmax diff:", (c - (a + b)).abs().max().item())