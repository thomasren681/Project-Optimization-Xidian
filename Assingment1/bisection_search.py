import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*(x**3)-4*x+2

def f_der(x):
    return 9*(x**2)-4

x = np.linspace(0,2)
y = f(x)

x1 = 0
x3 = 2
x2 = 1
epsilon = 0.2
iteration = 0
cache = []

while (abs(x3-x1)>epsilon):
    x2 = 0.5*(x1+x3)
    cache.append(x2)
    if f_der(x2)<0:
        x1 = x2
    elif f_der(x2)>0:
        x3 = x2
    else:
        x_bar = x2
        break

    iteration+=1
x_bar = 0.5*(x1+x3)
cache.append(x_bar)
cache = np.array(cache)
plt.plot(x,y)
plt.plot(cache,f(cache),color = 'red')
plt.show()

