#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



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

def sim_rk4(f, x0, dt, N, p):
    # create our position vector
    x = np.zeros(list(x0.shape) + [N])
    x[:,0] = x0
    # create our time vector
    t = np.cumsum(np.zeros(N)+dt) - dt

    # perform the simulation
    for n in range(1, N):
        x[:,n] = rk4(x[:, n-1], t[n-1], dt, f, p)
    
    return x, t




#%%
p = np.array([[0,-1],[1,0]])
x0 = np.array([1,0])
N = 100
t = np.zeros(N)
dt = 0.01
xlim = (-2, 2)
ylim = (-2, 2)


fig, ax = plt.subplots()
ax.axis('equal')
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)


def on_mouse(event, fig, ax):    
    x0 = np.array([event.xdata, event.ydata])
    x, t = sim_rk4(f, x0, dt, N, p)
    A = Animator(ax, x)
    ani = animation.FuncAnimation(
        fig, A, init_func=A.init, frames=N, interval=1000/N, blit=True,
        save_count=1000,
        repeat=False)
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event',
                       lambda event: on_mouse(event, fig=fig, ax=ax))

class Animator(object):
    def __init__(self, ax, x, xlim=(-2,2), ylim=(-2,2)):
        self.ax = ax
        self.line, = ax.plot([], [], 'k-')
        self.ax.axis('equal')
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.x = x
        self.N = self.x.shape[1]

    def init(self):
        self.line.set_data([],[])
        return self.line,

    def __call__(self, i):
        self.line.set_data(self.x[0, :i], self.x[1, :i])
        return self.line,

plt.show()

#%%
