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
    x1, x2 = x
    res = np.array([-x2, x1])
    return res

def rk4(r, t, dt, f, p):
    k1 = f(r, p)
    k2 = f(r + 0.5*dt*k1, p)
    k3 = f(r + 0.5*dt*k2, p)
    k4 = f(r + dt*k3, p)
    return r + dt * (k1 + 2*k2 + 2*k3 + k4)/6


def rka(x, t, dt, err, f, p):
    tSave = t
    xSave = x
    safe1 = 0.9
    safe2 = 4.
    maxTry = 100
    eps = np.finfo(float).eps
    for i in range(maxTry):
        half_dt = dt/2
        xTemp = rk4(xSave, tSave, half_dt, f, p)
        t = tSave + half_dt
        xSmall = rk4(xTemp, t, half_dt, f, p)
        
        t = tSave + dt
        xBig = rk4(xSave, tSave, dt, f, p)

        scale = err  * (np.abs(xSmall) + np.abs(xBig))/2
        xDiff = xSmall - xBig
        errorRatio = np.max( np.abs(xDiff)/(scale + eps))

        dt_old = dt
        dt = safe1*dt_old*errorRatio**(-0.2)
        dt = max(dt, dt_old/safe2)
        dt = min(dt, safe2*dt_old)

        if errorRatio < 1:
            return xSmall, t, dt
    
    raise Exception('Adaptive Runge-Kutta routine failed')

#%%
p = np.array([[0,-1],[1,0]])
x = np.array([1,0])
N = 10000
t = np.zeros(N)
x_final = np.zeros(list(x.shape) + [N])
x_final[:,0] = x
x_final2= np.zeros(list(x.shape) + [N])
x_final2[:,0] = x
dt = np.zeros(N) + 0.01

Aerr = 10e-5

for n in range(1, N):
    x_final[:, n] = rk4(x_final[:,n-1], 0, dt[0], f, p)
    #x_final[:,n], t[n], dt[n] = rka(x_final[:,n-1], t[n-1], dt[n-1], Aerr, f, p)
    x_final2[:,n] = x_final2[:,n-1] + dt[n]*f(x_final2[:,n-1],p)

diffmag = np.linalg.norm(x_final-x_final2, axis=0)

fig, ax = plt.subplots()
ax.plot(x_final[0],x_final[1])
ax.axis('equal')
#%%
