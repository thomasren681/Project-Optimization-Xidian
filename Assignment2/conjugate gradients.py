import numpy as np
import matplotlib.pyplot as plt

def f(x1,x2):
    return x1**2+2*(x2**2)-4*x1-2*x1*x2

def der_x(x,y):
    return 2*x-2*y-4
def der_y(x,y):
    return 4*y-2*x

def phi(x,y,lr):
    dx = der_x(x,y)
    dy = der_y(x,y)
    return f(x-dx*lr,y-dy*lr)

def phi_p(x,y,lr,p1,p2):
    return f(x-lr*p1,y-lr*p2)

def get_minx(x,y,min,middle,max):
    min_x = 0.5*((middle**2-max**2)*phi(x,y,min)+(max**2-min**2)*phi(x,y,middle)+(min**2-middle**2)*phi(x,y,max))
    min_x /= (middle-max)*phi(x,y,min)+(max-min)*phi(x,y,middle)+(min-middle)*phi(x,y,max)
    return min_x

# print(get_minx(1,1,0,1,2))

def get_minx_p(x,y,min,middle,max,p1,p2):
    min_x = 0.5*((middle**2-max**2)*phi_p(x,y,min,p1,p2)+(max**2-min**2)*phi_p(x,y,middle,p1,p2)+(min**2-middle**2)*phi_p(x,y,max,p1,p2))
    min_x /= (middle-max)*phi_p(x,y,min,p1,p2)+(max-min)*phi_p(x,y,middle,p1,p2)+(min-middle)*phi_p(x,y,max,p1,p2)
    return min_x

# print(get_minx_p(2,0.5,0,1,2,2,1.5))

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

def get_best_conjugate_lr_interpolation(x,y,x0,y0,px,py):
    x1 = 0
    x3 = 2
    x2 = 1
    epsilon = 0.1
    iteration = 1
    cache = []

    dx = der_x(x, y)
    dy = der_y(x, y)
    dx0 = der_x(x0, y0)
    dy0 = der_y(x0, y0)
    norm_g0 = np.sqrt(dx0**2+dy0**2)
    norm_g = np.sqrt(dx**2+dy**2)
    alpha = (norm_g/norm_g0)**2
    px = -dx-alpha*px
    py = -dy-alpha*py

    lr = get_minx_p(x, y, x1, x2, x3, px, py)

    while (abs(x2 - lr) > epsilon):
        cache.append(lr)
        if lr < x2:
            x3 = x2
            x2 = lr
            lr = get_minx_p(x, y, x1, x2, x3, px, py)
        else:
            x1 = x2
            x2 = lr
            lr = get_minx_p(x, y, x1, x2, x3, px, py)
        iteration += 1

    cache.append(lr)
    cache.append(iteration)
    cache = np.array(cache)

    return lr,px,py, cache

# print(get_best_conjugate_lr_interpolation(2,0.5,1,1,-4,2))

x = np.linspace(1,7)
y = np.linspace(0,5)
X,Y = np.meshgrid(x,y)

plt.contourf(X, Y, f(X, Y), 8, alpha=0.75)

C = plt.contour(X, Y, f(X, Y), 8, colors='black')
plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())


x,y = 1,1
px = der_x(x,y)
py = der_y(x,y)
epsilon = 0.001
cache_x = []
cache_y = []
iteration = 1
k = 0
cache_x.append(x)
cache_y.append(y)
learning_rate,_ = get_best_lr_interpolation(x,y)
x -= learning_rate*px
y -= learning_rate*py
norm_g = np.sqrt(px**2+py**2)

while (norm_g>epsilon):
    cache_x.append(x)
    cache_y.append(y)
    if k == 2:
        k = 0
        px = der_x(x, y)
        py = der_y(x, y)
        learning_rate, _ = get_best_lr_interpolation(x, y)
        x -= learning_rate * px
        y -= learning_rate * py
        iteration += 1
        norm_g = np.sqrt(der_x(x,y)**2+der_y(x,y)**2)
        continue
    else:
        learning_rate,px,py,_ = get_best_conjugate_lr_interpolation(x,y,cache_x[-2],cache_y[-2],px,py)
        x -= learning_rate * px
        y -= learning_rate * py
        iteration += 1
        k += 1
        norm_g = np.sqrt(der_x(x, y) ** 2 + der_y(x, y) ** 2)

cache_x.append(x)
cache_y.append(y)
cache_x = np.array(cache_x)
cache_y = np.array(cache_y)
plt.plot(cache_x,cache_y,color = 'red', linewidth = 2)
plt.show()

