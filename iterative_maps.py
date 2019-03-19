import numpy as np
import matplotlib.pyplot as plt


xlim = np.array([-2, ])
N = 100
x0 = 0.1

def f(x):
    r = 3.1
    return r*x*(1-x)


def sim(x0, f=f, N=N):
    x = np.zeros(N)
    x[0] = x0
    for i in range(1,N):
        x[i] = f(x[i-1])

    return x


def main(x0, f=f, N=N):
    fig, ax = plt.subplots()

    x = sim(x0, f, N)

    ax.plot(x)

    plt.show()


if __name__ == "__main__":
    main(x0)