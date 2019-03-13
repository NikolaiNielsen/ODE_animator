#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys


# inputs:
# x: np-array, shape (d,n), where d is dimensionality and n is number of 
#    trajectories.
# f: function that calculates the derivatives of x based on parameters p.
# p: object containing the parameters for the function p. For example d,d 
#    matrix for a linear system



# data structure:
# the data structure has the following form:
# data.shape == (no init conditions, no coords, no iterations)


p = np.array([[0, -1], [1, 0]])
x0 = np.array([1, 0])
N = 5000
t = np.zeros(N)
dt = 0.01
xlim = np.array([-1, 1]) * 4
ylim = np.array([-1, 1]) * 4
n_skip = 10
e_stop = 0.001

def f(x, p):
    mu = 2
    x1, x2 = x.T
    dx = x2
    # dy = mu * (1-x1**2) * x2 - x1
    # dx = -x2
    dy = -x1
    return np.array((dx, dy)).T


def rk4(r, t, dt, f, p=None):
    # The four contributions
    k1 = f(r, p)
    k2 = f(r + 0.5*dt*k1, p)
    k3 = f(r + 0.5*dt*k2, p)
    k4 = f(r + dt*k3, p)
    return r + dt * (k1 + 2*k2 + 2*k3 + k4)/6


def get_plot_indices(N, n):
    # Returns a list of indices used for each frame of the animation
    # N: Total number of data points
    # n: number of (new) points to show in each frame
    frames = int(np.ceil(N/n))
    id = np.cumsum(np.ones(frames)*n, dtype=int)
    # If the last entry in id is larger than N, we set it to N, so we dont' get 
    # an index out of bounds error
    if id[-1] > N:
        id[-1] = N
    # And lastly we subtract 1 to account for zero-indexation:
    return id-1


def rka(x, t, dt, err, f, p=None):
    # Adaptive Runge Kutta. Directly lifted from our course in Numerical
    # Methods.
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


def sim_rk4(f, x0, dt, N, p=None):
    # create our position vector
    x = np.zeros(list(np.atleast_2d(x0).shape) + [N])
    x[:,:,0] = x0
    # create our time vector
    t = np.cumsum(np.zeros(N)+dt) - dt

    # perform the simulation
    for n in range(1, N):
        x[:,:,n] = rk4(x[:,:, n-1], t[n-1], dt, f, p)

        # Calculate the minimum distance to preceeding points. Stop if smaller
        # than e_stop.
        min_dist = get_min_dists(x[:,:,n], x[:,:,:n])
        if (min_dist <= e_stop).all():
            N = n + 1
            return x[:,:,:N], t[:N]
    return x, t

def get_min_dists(x, x_pre):
    x_cur = np.atleast_3d(x)
    x_pre = np.atleast_3d(x_pre)
    return np.amin(np.sqrt(np.sum((x_pre-x_cur)**2, axis=1)), axis=1)


class Animator(object):
    # The animator object, created for every simulation we want to plot
    def __init__(self, fig, ax, xlim=(-2, 2), ylim=(-2, 2)):
        # plug stuff into the object and create the empty line
        self.ax = ax
        self.fig = fig
        self.ax.axis('equal')
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.max_frames = []
        self.current_frame = []
        self.ids = []
        self.data = []
        self.artists = []
    
    def add_artist(self, x, n_skip):
        for xi in x:
            ids = self.get_plot_indices(xi.shape[-1], n_skip)
            self.data.append(xi)
            self.ids.append(ids)
            self.max_frames.append(ids.size)
            self.current_frame.append(0)
            self.artists.append(self.ax.plot([],[],'k-')[0])
    
    def get_plot_indices(self, N, n):
        # Returns a list of indices used for each frame of the animation
        # N: Total number of data points
        # n: number of (new) points to show in each frame
        frames = int(np.ceil(N/n))
        id = np.cumsum(np.ones(frames)*n, dtype=int)
        # If the last entry in id is larger than N, we set it to N, so we dont'
        # get an index out of bounds error
        if id[-1] > N:
            id[-1] = N
        # And lastly we subtract 1 to account for zero-indexation:
        return id-1

    def init(self):
        # function to clear the line every time it is plotted (init function 
        # for FuncAnim)
        for artist in self.artists:
            artist.set_data([], [])
        return self.artists

    def __call__(self, i):
        # The function for updating the plot
        for n in range(len(self.artists)):
            i = self.current_frame[n]
            id_ = self.ids[n][i]
            # if i == self.max_frames[n] - 1:
                # on the last step we stop the animation and just plot the 
                # finished product instead
                # self.fig.canvas.close_event()
                # self.ax.plot(self.data[n][0], self.data[n][1], 'k-')
                # print('animation done')
            self.artists[n].set_data(self.data[n][0, :id_], 
                                     self.data[n][1, :id_])
            self.current_frame[n] += 1
        return self.artists


def on_mouse(event, A, fig, ax, n_skip, N=N):
    # Pressing the mouse button grabs the x and y position, simulates the 
    # system and animates it.

    # But only if it's a left mouse click
    if event.button != 1: return

    # Grab the initial conditions
    x0 = np.array([event.xdata, event.ydata])

    # simulate with RK4
    x, _ = sim_rk4(f, x0, dt, N, p)

    A.add_artist(x, n_skip)

    # Run the animation
    ani = animation.FuncAnimation(
        fig, A, init_func=A.init, interval=2, blit=True,
        save_count=1000,
        repeat=False)
    
    # And remember to draw!
    fig.canvas.draw()


def quitter(event):
    # Quit the program if ctrl+q is pressed
    if event.key == 'ctrl+q':
        print('Quitting')
        sys.exit()


def disconnect_mouse(event, fig, cid):
    # Disconnect the on_mouse event when ctrl+z is pressed
    if event.key == 'ctrl+z':
        fig.canvas.mpl_disconnect(cid)
        print('on_mouse disconnected')


def main(f=f):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    A = Animator(fig, ax, xlim, ylim)

    cid = fig.canvas.mpl_connect('button_press_event',
                        lambda event: on_mouse(event, A=A, fig=fig, ax=ax, 
                                                  n_skip=n_skip))

    cid2 = fig.canvas.mpl_connect('key_press_event', quitter)

    cid3 = fig.canvas.mpl_connect(
        'key_press_event', lambda event: disconnect_mouse(event, fig, cid))
    plt.show()

if __name__ == "__main__":
    main()
