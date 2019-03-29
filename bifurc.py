import matplotlib.pyplot as plt, numpy as np
import dynsys
import iterative_maps

def f(x):
    return np.sin(x)

def df(x):
    return np.cos(x)


def newton(x0, f, df, N=15, atol = 1e-08, rtol = 1e-05):
    # perform the Newton method for N iterations. Needs initial guesses,
    # function and its derivative. Also takes into account zeros of the
    # derivative.
    
    x0_arr = np.atleast_1d(x0)
    x = np.zeros((*x0_arr.shape, N))
    x[:, 0] = x0_arr
    for i in range(1, N):
        dfs = df(x[:, i-1])
        zeros = dfs == 0
        if np.sum(zeros) != 0:
            # we have zeros, do something! (gently nudge by +- epsilon)
            epsilon = 0.01
            # 50/50 either a -1 or 1.
            direction = np.random.randint(0, 2, np.sum(zeros)) * 2 - 1
            dfs[zeros] = epsilon*direction
        x[:, i] = x[:, i-1] - (f(x[:, i-1]) / dfs)

    # we only return the result
    x_end = x[:, -1]

    # and make sure to only include those that are nearly zero
    is_zero = np.isclose(f(x_end), 0, rtol=rtol, atol=atol)
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


def main():
    lim = [-1, 3*np.pi]
    n = 100
    x0 = np.linspace(*lim, n)
    x = newton(x0, f, df)

    x_inside = limit_roots(x, lim)
    fig, ax = plt.subplots()
    ax.scatter(x_inside, np.ones_like(x_inside))
    # ax.set_xlim(*lim)
    print(x_inside)
    x_unique = find_approx_unique(x_inside)
    print(x_unique)
    plt.show()

if __name__ == "__main__":
    main()

# idea for getting (approximately) unique values
# loop through values of x_inside.
# For each (unique) value, take the ones that are approximately equal to that one, and
# note their index in x_inside.
# Only compare those
