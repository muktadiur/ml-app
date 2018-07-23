import numpy as np

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0

def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

def gradient(x,y):
    return 2 * x * (x * w - y)

print('before train prediction {}'.format(forward(4.0)))
for i in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        l = loss(x_val, y_val)
        print(i, w, l)
print('after train prediction {}'.format(forward(4.0)))


