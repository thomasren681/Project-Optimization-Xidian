import numpy as np
import matplotlib.pyplot as plt

def f(x1,x2):
    return x1**2+2*(x2**2)-4*x1-2*x1*x2

x = np.linspace(1,7)
y = np.linspace(0,5)

# print(x)
# print(y)
X,Y = np.meshgrid(x,y)
# print(X)
# print(Y)

# x1 = [1,2]
# x2 = [1,2]
# x1 = np.array(x1)
# x2 = np.array(x2)
# plt.plot(x1,x2,color = 'red', linewidth = 2)

plt.contourf(X, Y, f(X, Y), 8, alpha=0.75)

C = plt.contour(X, Y, f(X, Y), 8, colors='black')
plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())
# plt.show()

def der_x(x,y):
    return 2*x-2*y-4
def der_y(x,y):
    return 4*y-2*x

learning_rate = 0.001
lr = learning_rate
x1 = 1
x2 = 1
iteration = 5000
cache_x1 = [1]
cache_x2 = [1]


for iter in range(iteration):
    cache_x1.append(x1)
    cache_x2.append(x2)
    dx1 = der_x(x1,x2)
    dx2 = der_y(x1,x2)
    x1 -= lr*dx1
    x2 -= lr*dx2


cache_x1.append(x1)
cache_x2.append(x2)
cache_x1 = np.array(cache_x1)
cache_x2 = np.array(cache_x2)
plt.plot(cache_x1,cache_x2,color = 'red', linewidth = 2)
plt.show()
