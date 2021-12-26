import numpy
import torch
import pandas

x = torch.arange(10)
print(x)
print(x.shape)
print(x.numel())
x = x.reshape(5,2)
print("x[1:3] = ", x[1:3])
#Index of 1, 2(the 2nd and 3rd elements) in 0 dimension(The most outside dimension)
print(x)

y = torch.arange(24)
print(y)
print(y.shape)
y = y.reshape(2,3,4)
#The right most number represent the most inside dimension
print("y = ", y)
print("y.size() = ", y.size())
print("y.shape = ", y.shape)
#shape is a variable which is equal to the return value of size() function

print("y[-1] = ", y[-1])
print("y[1:3] = ", y[1:3])
print("y[1]2][3] = ", y[1][2][3])
#Each[] makes the return value has a less dimension([]) of the original tensor

a = torch.arange(12, dtype = torch.float32)
a = a.reshape(3,4)
print(a)
b = torch.arange(12, 24, dtype = torch.float32).reshape(4,3)
print(b)
b = b.reshape(3,4)
#The above two lines are equal to b = torch.arange(12, 24, dtype = torch.float32).reshape(3, 4)
print(b)
Addition = a + b
#Add each element
print("Addition = ", Addition)
Multiplication = a * b
#Multiple each element
print("Multiplication = ", Multiplication)

c = torch.cat((a,b), dim = 0)
#operate the 1st dimension
print(c)
d = torch.cat((a, b), dim = 1)
#operate the 2nd dimensiond
print(d)

e = torch.arange(3).reshape(3,1)
#Reshape can be used to modify the dimension
f = torch.arange(2).reshape(1,2)
print("e = ", e)
print("f = ", f)
g = f + e
#Using broadcasting mechanism
print(g)

#Menmory in python
before = id(y)
print("id(y) = ", id(y))
#store the id of y into before
print("y = ", y)
y = y + 1
print("y = ", y)
print(id(y) == before)
#The id is changed(to build a new variable with the same name(like functional programming))
print("id(y) = ", id(y))
y[:] = y[:] + 1
#Modify the element in the tensor will not change the menmory
print("y = ", y)
print("id(y) = ", id(y))
y += 1
print("id(y) = ", id(y))

#Change between nummpy and torch
A = x.numpy()
print (A, type(A))
B = torch.tensor(A)
print (B, type(B))
SingleElementTensor = torch.tensor([1.7])
print (SingleElementTensor, SingleElementTensor.item(), float(SingleElementTensor), int (SingleElementTensor))

#Preprocessing data
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms, Alley, Price\n')
    f.write('NaN, Pave, 127500\n')
    f.write('2, NaN, 10600\n')
    f.write('4, NaN, 178100\n')
    f.write('NaN, NaN, 140000\n')

import pandas as pd

data = pd.read_csv(data_file)
print(data)
input, output = data.iloc[:, 0 : 2], data.iloc[:, 2]
#all rows and columns with the index of 1 and 2
input = input.fillna(input.mean())
print(input)

