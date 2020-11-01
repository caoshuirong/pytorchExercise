import torch
import numpy as np
t = torch.Tensor([1,2,3])
# 编程属性
print(t.dtype)
print(t.device)
# 编程操作
# 1、创建
ary = np.array([1,2,3])
print(type(ary))
# copy
t1 = torch.Tensor(ary)
t2 = torch.tensor(ary)
# share
t3 = torch.as_tensor(ary)
t4 = torch.from_numpy(ary)
print(id(ary))
print(id(t1))
print(id(t2))
print(id(t3))
print(id(t4))
# 验证
t3[0] = 100
print(ary)
print(t1)
print(t2)
print(t3)
print(t4)

# list时，不会share;tuple不可变，肯定不share
list1 = [1,2,3,4]
t5 = torch.as_tensor(list1)
t5[0] = 100
print(list1)
print(t5)

# 特殊数组
t1 = torch.eye(2)
t2 = torch.zeros(2,3)
t3 = torch.ones(2,3)
t4 = torch.rand(2,3)
print(t1)
print(t2)
print(t3)
print(t4)

# 改变形状
ary = [[1,2,3],[4,5,6],[7,8,9],[0,1,2]]
ary = np.array(ary)
print(ary.shape)

tmp = torch.tensor(ary,dtype = torch.float32)
print(tmp.reshape(1,-1))
print(tmp.reshape(3,2,2))
print(tmp.flatten(start_dim = 0))
# 拼接时沿该方向要能对齐。
ary2 = [[1,2,3]]
tmp2 = torch.tensor(ary2,dtype = torch.float32)
print(torch.cat((tmp,tmp2)))

# 叠加，要求张量的形状想同
torch.stack((torch.zeros(2),torch.ones(2)))

# 3.张量的逐元素操作
# 四则运算
print(tmp.shape)
print(tmp2.shape)
print(tmp*2)
print(tmp + tmp2)

# 此例子演示不满足广播兼容的情况
# tmp3 = torch.zeros(3,3)
# print(tmp.shape)
# print(tmp3.shape)
# print(tmp3)
# print(tmp+tmp3)

# 比较
print(tmp >= tmp2)

# abs,sqrt,neg
print(torch.neg(tmp))
print(torch.abs(tmp))
print(torch.sqrt(tmp))

# 4.缩减操作
print(tmp)
print(tmp.sum())
print(tmp.mean(dim = 0))
print(tmp.std(dim = 0))
print(tmp.max(dim = 0))
print(tmp.argmax(dim = 0))
print(tmp.argmax(dim = 1))

# 5.转成其他类型
# 单元素张量转换成数字
print(tmp.sum().item())
print(tmp[1].tolist())
print(tmp[1].numpy())






