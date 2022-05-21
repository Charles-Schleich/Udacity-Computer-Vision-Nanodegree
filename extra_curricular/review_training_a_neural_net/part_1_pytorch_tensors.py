
import numpy as np 
import torch 

x = torch.rand(3,2)
# print(x)
y = torch.ones(x.size())
# print(y)
z = x+y
print(z)
print(z[0])
print(z[:,1:])

# Creates a new tensor and adds 1 to it, z is unchanged
z1 = z.add(1)
print("z",z)
print("z1",z1)
# Inplace version = Changes the value of z
z.add_(3)
print("z",z)

# Resize / reshape tensor
z.resize_(2,3)
print("z",z)


print("==============\n\n==============")
np_arr = np.random.rand(4,3)

torch_arr = torch.from_numpy(np_arr)
print(torch_arr)

np_arr_return = torch_arr.numpy()
print(np_arr_return)

# Achtung ! The 
torch_arr.mul_(2)
print(torch_arr)

print(np_arr)


