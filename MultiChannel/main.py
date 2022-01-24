import torch


def corr2d(f_x, f_k):
    # Realize 2d convolution
    f_h, f_w = f_k.shape
    f_y = torch.zeros(f_x.shape[0] - f_h + 1, f_x.shape[1] - f_w + 1)
    for i in range(f_y.shape[0]):
        for j in range(f_y.shape[1]):
            f_y[i, j] = (f_x[i: i + f_h, j: j + f_w] * f_k).sum()
    return f_y


def corr2d_multi_in(f_x_set, f_k_set):
    c_i = 0
    y = []
    for f_x, f_k in zip(f_x_set, f_k_set):
        y.append(corr2d(f_x, f_k))
        print("c_i = ", c_i)
        print("y[c_i] = ", y[c_i])
        if c_i > 0:
            y[0] += y[c_i]
        c_i += 1
    return y[0]
    # The above implementation is equivalent to below
    # return sum(corr2d(f_x, f_k) for f_x, f_k in zip(f_x_set, f_k_set))


x = torch.tensor([[[0.0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
print("x.shape = ", x.shape)
# x.shape = torch.Size([2, 3, 3])
k = torch.tensor([[[0.0, 1], [2, 3]], [[1, 2], [3, 4]]])
print("k.shape = ", k.shape)
# k.shape = torch.Size([2, 2, 2])
print("corr2d_multi_in(x, k) = ", corr2d_multi_in(x, k))


def corr2d_multi_in_out(f_x, f_k_set):
    return torch.stack([corr2d_multi_in(f_x, f_k) for f_k in f_k_set], 0)


k = torch.stack((k, k + 1, k + 2), 0)
print("stacked k = ", k)
for i in range(3):
    print("k[", i, "] = ", k[i])

print("corr2d_multi_in_out(x, k) = ", corr2d_multi_in_out(x, k))

# Illustrate that 1 * 1 convolution layer is equivalent to a linear layer


def corr2d_multi_in_out_1(f_x, f_k):
    f_c_i, f_h, f_w = f_x.shape
    f_c_o = f_k.shape[0]
    f_x = f_x.reshape(f_c_i, f_h * f_w)
    f_k = f_k.reshape(f_c_o, f_c_i)
    f_y = torch.matmul(f_k, f_x)
    return f_y.reshape(f_c_o, f_h, f_w)


x = torch.normal(0, 1, (3, 3, 3))
k = torch.normal(0, 1, (2, 3, 1, 1))
print("x = ", x)
print("k = ", k)

y_conv = corr2d_multi_in_out(x, k)
y_linear = corr2d_multi_in_out_1(x, k)

print("abs = ", float(torch.abs(y_conv - y_linear).sum()))
