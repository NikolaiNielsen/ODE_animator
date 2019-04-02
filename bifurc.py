import matplotlib.pyplot as plt, numpy as np
import dynsys
import iterative_maps

def f(x, r):
    return r  + x**2

def df(x, r):
    return 2*x

f = np.vectorize(f)
df = np.vectorize(df)


def newton(x0, f, df, r=None, N=15, atol = 1e-08, rtol = 1e-05):
    # perform the Newton method for N iterations. Needs initial guesses,
    # function and its derivative. Also takes into account zeros of the
    # derivative.
    
    x0_arr = np.atleast_1d(x0)
    x = np.zeros((*x0_arr.shape, N))
    x[:, 0] = x0_arr
    for i in range(1, N):
        dfs = df(x[:, i-1], r)
        zeros = dfs == 0
        if np.sum(zeros) != 0:
            # we have zeros, do something! (gently nudge by +- epsilon)
            epsilon = 0.01
            # 50/50 either a -1 or 1.
            direction = np.random.randint(0, 2, np.sum(zeros)) * 2 - 1
            dfs[zeros] = epsilon*direction
        x[:, i] = x[:, i-1] - (f(x[:, i-1], r) / dfs)

    # we only return the result
    x_end = x[:, -1]

    # and make sure to only include those that are nearly zero
    is_zero = np.isclose(f(x_end, r), 0, rtol=rtol, atol=atol)
    x_end = x_end[is_zero]
    return x_end


def limit_roots(x, lim):
    # return the values of x within the limits (inclusive)
    x = x[(x >= lim[0]) * (x <= lim[1])]
    return x


def find_approx_unique(x, atol=1e-08, rtol=1e-05):
    # returns an array of unique numbers from the parent array x, where
    # uniqueness is determined by np.islose. The returned number is the mean of
    # all numbers returned true by isclose
    x_left = x.copy()
    x_unique = []
    num_left = x_left.size

    while num_left > 0:
        # We take the first element of x_left, and find all those that are
        # close to it.
        trial = x_left[0]
        hits = np.isclose(x_left, trial, atol=atol, rtol=rtol)

        # We then append the mean of these values to x_unique, and delete them
        # from x_left.
        values = x_left[hits]
        x_unique.append(np.mean(values))
        x_left = x_left[~hits]
        num_left = x_left.size
    
    return np.array(x_unique)


def bifurc(f, df, rlim, xlim, nr=100, nx=100, N=15, atol=1e-08, rtol=1e-05):
    # calculates the bifurcation diagram. For each value of r, the roots of f
    # within xlim are found using Newtons method, Points are colored according
    # to their stability.
    
    # Colors: blue is stable, red is unstable, green is tangent 
    colors = ['b', 'r', 'g']

    # the y-values
    roots = []
    # and corresponding x-values
    roots_r = []
    x0 = np.linspace(*xlim, num=nx)
    rs = np.linspace(*rlim, num=nr)
    for r in rs:
        # first we find the roots
        x_end = newton(x0, f, df, r, N, atol, rtol)
        x_limit = limit_roots(x_end, xlim)
        x_unique = find_approx_unique(x_limit, atol, rtol)
        # Then we add the roots to a list
        roots += x_unique.tolist()
        # And the r-value for each root
        roots_r += [r] * x_unique.size
    
    # gather roots in arrays
    roots = np.array(roots)
    roots_r = np.array(roots_r)

    # calculate the derivatives
    dfs = df(roots, roots_r)

    # find stability and instabilities
    stable = dfs < 0
    unstable = dfs > 0
    tangent = dfs == 0

    # color the roots
    roots_color = np.array(['b'] * dfs.size)
    roots_color[stable] = colors[0]
    roots_color[unstable] = colors[1]
    roots_color[tangent] = colors[2]

    # now we create the figure
    fig, ax = plt.subplots()
    ax.scatter(roots_r[stable], roots[stable], 
               c=colors[0], s=10, label='stable')
    ax.scatter(roots_r[unstable], roots[unstable],
               c=colors[1], s=10, label='unstable')
    ax.scatter(roots_r[tangent], roots[tangent],
               c=colors[2], s=10, label='tangent')
    
    # plot xaxis
    ax.plot(rlim, [0,0], 'k-', linewidth=1)

    ax.set_xlim(*rlim)
    ax.set_ylim(*xlim)
    ax.set_xlabel('r')
    ax.set_ylabel('x')
    ax.set_title('Bifurcation diagram for f')
    ax.legend()
    fig.tight_layout()
    return fig, ax
    



def main():
    xlim = [-2, 2]
    rlim = [-1, 2]
    nx = 200
    nr = 200
    fig, ax = bifurc(f, df, rlim, xlim, nr, nx)
    plt.show()

if __name__ == "__main__":
    main()
