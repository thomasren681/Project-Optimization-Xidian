import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*(x**3)-4*x+2

x = np.linspace(0,2)
y = f(x)

plt.plot(x,y)


def get_minx(min,middle,max):
    min_x = 0.5*((middle**2-max**2)*f(min)+(max**2-min**2)*f(middle)+(min**2-middle**2)*f(max))
    min_x /= (middle-max)*f(min)+(max-min)*f(middle)+(min-middle)*f(max)
    return min_x

x1 = 0
x3 = 2
x2 = 1
epsilon = 0.2
iteration = 1
cache = []

x_bar = get_minx(x1,x2,x3)
# print(x_bar)
# print(f(x_bar))

while(abs(x2-x_bar)>epsilon):
    cache.append(x_bar)
    if x_bar<x2:
        x3 = x2
        x2 = x_bar
        x_bar = get_minx(x1,x2,x3)
    else:
        x1 = x2
        x2 = x_bar
        x_bar = get_minx(x1,x2,x3)
    iteration += 1

cache.append(x_bar)
cache = np.array(cache)
plt.plot(cache, f(cache), color='red')
plt.show()

