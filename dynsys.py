import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

dt = 0.05
xlim = np.array([-1, 1]) * 4
ylim = np.array([-1, 1]) * 4
n_skip = 10
e_stop = 0
N = 1000

def f(x):
    x, y = x.T
    dx = x + y - x*np.sqrt(x*x + y*y)
    dy = -x + y - y*np.sqrt(x*x + y*y)
    return np.array((dx, dy)).T


def rk4(r, t, dt, f):
    # The four contributions
    k1 = f(r)
    k2 = f(r + 0.5*dt*k1)
    k3 = f(r + 0.5*dt*k2)
    k4 = f(r + dt*k3)
    return r + dt * (k1 + 2*k2 + 2*k3 + k4)/6


def rka(x, t, dt, err, f):
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
        xTemp = rk4(xSave, tSave, half_dt, f)
        t = tSave + half_dt
        xSmall = rk4(xTemp, t, half_dt, f)

        t = tSave + dt
        xBig = rk4(xSave, tSave, dt, f)

        scale = err * (np.abs(xSmall) + np.abs(xBig))/2
        xDiff = xSmall - xBig
        errorRatio = np.max(np.abs(xDiff)/(scale + eps))

        dt_old = dt
        dt = safe1*dt_old*errorRatio**(-0.2)
        dt = max(dt, dt_old/safe2)
        dt = min(dt, safe2*dt_old)

        if errorRatio < 1:
            return xSmall, t, dt

    raise Exception('Adaptive Runge-Kutta routine failed')


def sim_rk4(f, x0, dt=dt, N=N, e_stop=e_stop):
    # create our position vector. Assume x0.shape == (n, 2), where n is number
    # of initial conditions.
    x = np.zeros(list(np.atleast_2d(x0).shape) + [N])
    x[:, :, 0] = x0
    # create our time vector
    t = np.cumsum(np.zeros(N)+dt) - dt

    # perform the simulation
    for n in range(1, N):
        x[:, :, n] = rk4(x[:, :, n-1], t[n-1], dt, f)

        # Calculate the minimum distance to preceeding points. Stop if smaller
        # than e_stop.
        min_dist = _get_min_dists(x[:, :, n], x[:, :, :n])
        if (min_dist <= e_stop).all():
            N = n + 1
            return x[:, :, :N], t[:N]
    return x, t


def _get_min_dists(x, x_pre):
    x_cur = np.atleast_3d(x)
    x_pre = np.atleast_3d(x_pre)
    return np.amin(np.sqrt(np.sum((x_pre-x_cur)**2, axis=1)), axis=1)


def add_traj_to_fig(x0, f, vec_perc=0.25, fig=None, ax=None,
                    dt=dt, N=N, e_stop=e_stop):
    # First simulate the new trajectories
    x, _ = sim_rk4(f, x0, dt, N, e_stop)
    if fig == None or ax == None:
        fig, ax = plt.subplots()
    

    starts = []
    datas = []
    for n in x:
        ax.plot(n[0], n[1], 'k-', linewidth=1)
        N = n.shape[1]
        n_arrow = round(vec_perc*(N-1))
        start_point = n[:, n_arrow]
        end_point = n[:, n_arrow+1]
        data = end_point - start_point
        datas.append(data)
        starts.append(start_point)
    starts = np.array(starts)
    datas = np.array(datas)

    # let's normalize the arrows
    norm = np.linalg.norm(datas, axis=1, keepdims=True)
    # length = 0.001
    datas /= norm
    ax.quiver(starts[:,0], starts[:,1], datas[:,0], datas[:,1], 
              pivot='middle', units='dots', width=3)
    
    
    fig.canvas.draw()
    return fig, ax
    

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
        self.to_be_removed = []

    def add_artist(self, x, n_skip):
        for xi in x:
            ids = self._get_plot_indices(xi.shape[-1], n_skip)
            self.data.append(xi)
            self.ids.append(ids)
            self.max_frames.append(ids.size - 1)
            self.current_frame.append(0)
            self.artists.append(self.ax.plot([], [], 'k-')[0])

    def _get_plot_indices(self, N, n):
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

    def _init(self):
        # function to clear the line every time it is plotted (init function
        # for FuncAnim)
        for artist in self.artists:
            artist.set_data([], [])
        return self.artists

    def _remove_finished_animations(self):
        for n in sorted(self.to_be_removed, reverse=True):
            del self.data[n]
            del self.artists[n]
            del self.current_frame[n]
            del self.max_frames[n]
            del self.ids[n]
        self.to_be_removed = []

    def __call__(self, i):
        # The function for updating the plot
        for n in range(len(self.artists)):
            i = self.current_frame[n]
            id_ = self.ids[n][i]
            if i == self.max_frames[n]:
                self.to_be_removed.append(n)
                # on the last step we stop the animation and just plot the
                # finished product instead
                # self.fig.canvas.close_event()
                self.ax.plot(self.data[n][0], self.data[n][1], 'k-')
                print('animation done')
            self.artists[n].set_data(self.data[n][0, :id_],
                                     self.data[n][1, :id_])
            self.current_frame[n] += 1
        self._remove_finished_animations()
        return self.artists


def _on_mouse(event, A, fig, ax, n_skip=n_skip, N=N, e_stop=e_stop, dt=dt):
    # Pressing the mouse button grabs the x and y position, simulates the
    # system and animates it.

    # But only if it's a left mouse click
    if event.button != 1:
        return

    # Grab the initial conditions
    x0 = np.array([event.xdata, event.ydata])

    # simulate with RK4
    x, _ = sim_rk4(f, x0, dt, N, e_stop)
 
    A.add_artist(x, n_skip)

    # Run the animation

    # And remember to draw!
    fig.canvas.draw()


def _quitter(event):
    # Quit the program if ctrl+q is pressed
    if event.key == 'ctrl+q':
        print('Quitting')
        sys.exit()


def _disconnect_mouse(event, fig, cid):
    # Disconnect the on_mouse event when ctrl+z is pressed
    if event.key == 'ctrl+z':
        fig.canvas.mpl_disconnect(cid)
        print('on_mouse disconnected')


def animation_window(f=f, xlim=xlim, ylim=ylim, n_skip=n_skip, dt=dt,
                     e_stop=e_stop):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    A = Animator(fig, ax, xlim, ylim)

    cid = fig.canvas.mpl_connect('button_press_event',
                                 lambda event: _on_mouse(event, A=A, fig=fig,
                                                         ax=ax,
                                                         n_skip=n_skip,
                                                         e_stop=e_stop,
                                                         dt=dt))

    cid2 = fig.canvas.mpl_connect('key_press_event', _quitter)

    cid3 = fig.canvas.mpl_connect(
        'key_press_event', lambda event: _disconnect_mouse(event, fig, cid))

    ani = animation.FuncAnimation(
        fig, A, init_func=A._init, interval=2, blit=True,
        save_count=1000,
        repeat=False)
    plt.show()


def phase_plane_builder(f=f, xlim=xlim, ylim=ylim, vec_perc=0.25, dt=dt, N=N,
                        e_stop=e_stop):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    cid = fig.canvas.mpl_connect('button_press_event',
                                 lambda event: _on_mouse_no_anim(
                                     event, f=f, vec_perc=vec_perc, fig=fig,
                                     ax=ax, dt=dt, N=N, e_stop=e_stop))
    cid2 = fig.canvas.mpl_connect('key_press_event', _quitter)

    cid3 = fig.canvas.mpl_connect(
        'key_press_event', lambda event: _disconnect_mouse(event, fig, cid))
    plt.show()


def _on_mouse_no_anim(event, f, vec_perc=0.25, fig=None, ax=None,
                      dt=dt, N=N, e_stop=e_stop):
    # Pressing the mouse button grabs the x and y position, simulates the
    # system and animates it.

    # But only if it's a left mouse click
    if event.button != 1:
        return

    # Grab the initial conditions
    x0 = np.array([event.xdata, event.ydata])

    # Add trajectory to fig
    add_traj_to_fig(x0=x0, f=f, vec_perc=vec_perc, fig=fig, ax=ax,
                    dt=dt, N=N, e_stop=e_stop)


if __name__ == "__main__":
    xlim = [-np.pi, np.pi]
    ylim = [-np.pi, np.pi]
    phase_plane_builder(xlim=xlim, ylim=ylim)
