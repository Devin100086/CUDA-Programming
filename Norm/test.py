import torch
from norm import two_norm  # 导入我们定义的包
n,m = 32,14  # 矩阵维度

a = torch.randn(n,m)
b = torch.randn(n,m)
c = torch.zeros(n)

print("a:\n",a)
print("\nb:\n",b)

# 将变量传送到GPU显存
a = a.cuda()
b = b.cuda()
c = c.cuda()

two_norm(b,a,c,n,m)  # 调用我们定义的包
torch.cuda.synchronize()  # 等待GPU返回结果
# 打印我们实现的结果和pytorch的实现结果
print("\nresult by two_norm:",c)
print("\nresult by torch.norm:",torch.norm(a-b, dim=1))