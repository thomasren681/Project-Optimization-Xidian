import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*(x**3)-4*x+2

x = np.linspace(0,2)
y = f(x)
a = 0
b = 2
t = (np.sqrt(5)-1)/2
x1 = a + (1-t)*(b-a)
x2 = a + t*(b-a)
epsilon = 0.2
iteration = 0
cache = []

while (abs(b-a)>epsilon):
    if f(x1)-f(x2)>0:
        cache.append(x2)
        a = x1
        x1 = x2
        x2 = a + t*(b-a)
    else:
        cache.append(x1)
        b = x2
        x2 = x1
        x1 = a + (1-t)*(b-a)
    iteration +=1

x_bar = 0.5*(a+b)
cache.append(x_bar)
cache = np.array(cache)
plt.plot(x,y)
plt.plot(cache,f(cache),color='red')
plt.show()

