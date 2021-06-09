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

def phi(x,y,lr):
    dx = der_x(x,y)
    dy = der_y(x,y)
    return f(x-dx*lr,y-dy*lr)

def get_minx(x,y,min,middle,max):
    min_x = 0.5*((middle**2-max**2)*phi(x,y,min)+(max**2-min**2)*phi(x,y,middle)+(min**2-middle**2)*phi(x,y,max))
    min_x /= (middle-max)*phi(x,y,min)+(max-min)*phi(x,y,middle)+(min-middle)*phi(x,y,max)
    return min_x

def get_best_lr_interpolation(x,y):
    x1 = 0
    x3 = 2
    x2 = 1
    epsilon = 0.1
    iteration = 1
    cache = []
    x_bar = get_minx(x, y, x1, x2, x3)

    while (abs(x2 - x_bar) > epsilon):
        cache.append(x_bar)
        if x_bar < x2:
            x3 = x2
            x2 = x_bar
            x_bar = get_minx(x,y,x1, x2, x3)
        else:
            x1 = x2
            x2 = x_bar
            x_bar = get_minx(x,y,x1, x2, x3)
        iteration += 1

    cache.append(x_bar)
    cache.append(iteration)
    cache = np.array(cache)

    return x_bar, cache



epsilon = 0.001
x = 1
y = 1
cache_x = []
cache_y = []
dx = der_x(x,y)
dy = der_y(x,y)
norm = np.sqrt(dx**2+dy**2)
iteration = 0

##########################################################

# Here is the code for fixed step gradient descent method
# learning_rate = 0.1
# learning_rate_decay = 0.9
# iteration = 200

# for iter in range(iteration):
#     cache_x1.append(x1)
#     cache_x2.append(x2)
#     dx1 = der_x(x1,x2)
#     dx2 = der_y(x1,x2)
#     x1 -= learning_rate*dx1
#     x2 -= learning_rate*dx2
#     if iter%100 == 0:
#         learning_rate *= learning_rate_decay

##########################################################

while (norm>epsilon):
    cache_x.append(x)
    cache_y.append(y)
    dx = der_x(x,y)
    dy = der_y(x,y)
    norm = np.sqrt(dx ** 2 + dy ** 2)
    learning_rate,_ = get_best_lr_interpolation(x,y)
    x -= learning_rate*dx
    y -= learning_rate*dy
    iteration += 1


cache_x.append(x)
cache_y.append(y)
cache_x = np.array(cache_x)
cache_y = np.array(cache_y)
plt.plot(cache_x,cache_y,color = 'red', linewidth = 2)
plt.show()



