import numpy as np

W1 = np.array([
    [0.2, -0.1, 0.4, 0.5, -0.3],
    [-0.5, 0.3, -0.2, 0.1, 0.6],
    [0.7, -0.4, 0.1, -0.6, 0.2],
])

W2 = np.array([
    [0.3, -0.2, 0.4],
    [-0.6, 0.5, -0.1],
    [0.2, -0.3, 0.7],
    [0.1, 0.6, -0.5]
])

b1 = np.array([
    [0.1],
    [-0.2],
    [0.3]
])

b2 = np.array([
    [0.05],
    [-0.15],
    [0.25],
    [0.1]
])

X = np.array([
    [0.5],
    [0.2],
    [-0.7],
    [0.9],
    [-0.4]
])


activation = np.tanh

a1 = activation(W1.dot(X) + b1)
dot_out_1 = W1.dot(X)
add_row_1 = dot_out_1 + b1

a2 = activation(W2.dot(a1) + b2)
dot_out_2 = W2.dot(a1)
add_row_2 = dot_out_2 + b2

print(f"{dot_out_1=}")
print(f"{add_row_1=}")
print(f"{a1=}")

print("=========================")

print(f"{dot_out_2=}")
print(f"{add_row_2=}")
print(f"{a2=}")


