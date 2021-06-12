import numpy as np
import matplotlib.pyplot as plt

def f(x1,x2):
    return x1**2+2*(x2**2)-4*x1-2*x1*x2
    # return 2*x1**2+x2**2-4*x1+2
# def f_test(x1,x2):
#     return 2*x1**2+x2**2-4*x1+2

def der_x(x,y):
    return 2*x-2*y-4
    # return 4*x-4
def der_y(x,y):
    return 4*y-2*x
    # return 2*y

def gradient(x):
    return np.array([der_x(x[0],x[1]),der_y(x[0],x[1])]).reshape(-1,1)

def renew_H(H,delta_x,delta_g):
    new_H = H + (delta_x.dot(delta_x.transpose()))/(delta_g.transpose().dot(delta_x)) \
    - (H.dot(delta_g.dot(delta_g.transpose().dot(H))))/(delta_g.transpose().dot(H.dot(delta_g)))
    return new_H

# if __name__ == '__main__':
#     H = np.eye(2)
#     delta_x = np.array([-10/9,-5/9]).reshape(-1,1)
#     delta_g = np.array([-40/9,-10/9]).reshape(-1,1)
#     print(renew_H(H,delta_x,delta_g))
#     print(np.array([[86,-38],[-38,305]])/306)

def phi(x,p,lr):
    temp = x+lr*p
    return f(temp[0],temp[1])

def get_min_lr(x,p,min,middle,max):
    min_lr = 0.5*((middle**2-max**2)*phi(x,p,min)+(max**2-min**2)*phi(x,p,middle)+(min**2-middle**2)*phi(x,p,max))
    min_lr /= (middle-max)*phi(x,p,min)+(max-min)*phi(x,p,middle)+(min-middle)*phi(x,p,max)
    return min_lr

# if __name__ == '__main__':
#     x = np.array([2,1]).reshape(-1, 1)
#     p = np.array([-4,-2]).reshape(-1,1)
#     print(get_min_lr(x,p,0,1,2))

x = np.linspace(1,7)
x2 = np.linspace(0,5)
X,Y = np.meshgrid(x,x2)

plt.contourf(X, Y, f(X, Y), 8, alpha=0.75)

C = plt.contour(X, Y, f(X, Y), 8, colors='black')
plt.clabel(C, inline=True, fontsize=10)

plt.xticks(())
plt.yticks(())

x = np.array([1,1],dtype=float).reshape(-1,1)
epsilon = 0.001
H = np.eye(2)
iteration, k = 0,0
g = gradient(x)
cache_x = []
cache_y = []
# x = x[0,0]
# y = x[1,0]
cache_x.append(x[0,0])
cache_y.append(x[1,0])

while (np.linalg.norm(g)>epsilon):
    if k == 2:
        cache_x.append(x[0])
        cache_y.append(x[1])
        k = 0
        iteration += 1
        continue
    else:
        p = -1*H.dot(g)
        lr = get_min_lr(x,p,0,1,2)
        cache_x.append(x[0, 0])
        cache_y.append(x[1, 0])
        x += lr*p
        # delta_x = x-np.array([cache_x[-2],cache_y[-2]]).reshape(-1,1)
        delta_x = lr*p
        delta_g = gradient(x)-g
        g = gradient(x)
        H = renew_H(H,delta_x,delta_g)
        iteration += 1
        k += 1


cache_x.append(x[0,0])
cache_y.append(x[1,0])
cache_x = np.array(cache_x)
cache_y = np.array(cache_y)
plt.plot(cache_x,cache_y,color = 'red', linewidth = 2)
plt.show()