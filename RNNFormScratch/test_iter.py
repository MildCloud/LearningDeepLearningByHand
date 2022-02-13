from cgitb import small


class Test_Iter1:
    def __init__(self, list = []):
        self.list = list
    
    def __iter__(self):
        return self.list


# test_iter1 = Test_Iter1
# for i in test_iter1:
#     print('i in test_iter1 = ', i)
# # TypeError: 'type' object is not iterable


# test_iter2 = Test_Iter1()
# for i in test_iter2:
#     print('i in test_iter2 = ', i)
# # TypeError: 'type' object is not iterable


# test_iter3 = Test_Iter1([1, 2, 3])
# for i in test_iter3:
#     print('i in test_iter3 = ', i)
# # TypeError: iter() returned non-iterator of type 'list'


def small_loop():
    for i in range(3):
        yield i


class Test_Iter2:
    def __init__(self):
        self.small_loop = small_loop()
        # To use __iter__function, self.small_loop must be a true function call.

    def __iter__(self):
        return self.small_loop


test_iter4 = small_loop()
for i in test_iter4:
    print('i in test_iter4 = ', i)

test_iter4 = Test_Iter2()
# To create an object, () is needed
for i in test_iter4:
    print('i in test_iter4 = ', i)

