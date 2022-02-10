from re import A


a = 1
b = 1

def f(a):
    a = 10
    print("in function a = ", a)
    b = 10
    print("in function b = ", b)

f(a)
print("after function a = ", a)
print("after function b = ", b)
# in function a =  10
# in function b =  10
# after function a =  1
# after function b =  1
