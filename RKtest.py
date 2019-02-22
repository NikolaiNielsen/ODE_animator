#%%
import numpy as np
import matplotlib.pyplot as plt


# inputs:
# x: np-array, shape (d,n), where d is dimensionality and n is number of 
#    trajectories.
# f: function that calculates the derivatives of x based on parameters p.
# p: object containing the parameters for the function p. For example d,d 
#    matrix for a linear system


def f(x, p):
    res = p @ x
    return res

def rk4(r, dt, f, p):
    k1 = f(r, p)
    k2 = f(r + k1/2, p)
    k3 = f(r + k2/2, p)
    k4 = f(r + k3, p)
    return r + dt * (k1 + 2*k2 + 2*k3 + k4)/6

#%%
dt = 0.01
p = np.array([[0,-1],[1,0]])
x = np.array([0,1])
N = 1000

x_final = np.zeros(list(x.shape) + [N])
x_final[:,0] = x

for n in range(1, N):
    x_final[:,n] = rk4(x_final[:,n-1], dt, f, p)

fig, ax = plt.subplots()
ax.plot(x_final[0], x_final[1])
ax.axis('equal')
#%%
