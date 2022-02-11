from re import A


a = 1
b = 1
c = [1, 2, 3]
d = [1, 2, 3, 4]
f = 3


def func(a):
    a = 10
    print("in function a = ", a)
    b = 10
    print("in function b = ", b)
    c = [1, 2]
    print("in function c = ", c)
    return d[f]


result = func(a)
print('result = ', result)
print("after function a = ", a)
print("after function b = ", b)
print("after function c = ", c)
# in function a =  10
# in function b =  10
# in function c =  [1, 2]
# result =  4
# after function a =  1
# after function b =  1
# after function c =  [1, 2, 3]

"""
Conclution: when variables having the same name of outer variables, 
if the variabel inside is used as right value, it will be consider as a new variable; 
else(be used as left variable) the variable inside will by consider as the outer one
"""
