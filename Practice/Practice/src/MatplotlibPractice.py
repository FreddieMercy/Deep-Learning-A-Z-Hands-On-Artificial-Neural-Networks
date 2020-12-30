import numpy as np
from matplotlib import pylab, pyplot


def matplotlibPractice():
    x = np.linspace(0, 10, 25)  # random number within range 0 ~ 10, len(x) = 25

    print("******************************")
    print("=== x: \n")
    print(x)
    y = x ** 2 + 2
    print("******************************")
    print("=== y: \n")
    print(y)

    print("******************************")
    print("=== [x,y] reshape: \n")

    print(
        np.array([x, y])  # 2 by 25 array
            .reshape(25, 2)  # reshape to 25 by 2 (25 rows, 2 columns)
            # so, looks like "reshape" chops from left to right, top to btm, index 0 recursively
            .reshape(2, 25)
    )

    pylab.plot(x, y, 'r')
    fig, axis = pyplot.subplots(2, 3)  # build a context: 2 by 3, will display 6 plot figs.

    x = np.linspace(0, 8, 1000)

    axis[0, 0].plot(x, np.sin(x), 'g')  # row=0, col=0
    axis[0, 1].plot(range(100), 'b')  # row=0, col=1
    axis[0, 2].plot(x + 100, 'b', alpha=0.3)  # row=0, col=2 # alpha means transparency
    axis[1, 0].plot(x, np.tan(x), 'k', linewidth=10)  # row=1, col=0 # "linewidth=10" can be written as "lw=10"
    axis[1, 1].plot(x, np.cos(x), 'r')  # row=1, col=1
    axis[1, 1].set_xlabel("x_label")
    axis[1, 1].set_ylabel("y_label")
    axis[1, 1].set_title("my title")
    axis[1, 1].legend(["legend 1", "legend 2", "legend 3"], loc=4)

    # No row=1, col=2, so it will be blank

    fig.show()

    # -------------------------------------------------------------------------

    # subplot (note: no 's')
    # - index has to be unique, otherwise newer figure will replace the older figure,
    #   unless: nrow and ncol are also the same, in that case, two (or more) figures combined into one holistic figure
    # - otherwise, each time call "subplot", means saved a spot for new graph that is coming (below)

    pylab.subplot(1, 2, 1)  # row, col, index
    pylab.plot(x, np.tan(x), 'r--')
    pylab.plot(x, np.cos(x), 'g*-')
    pylab.subplot(1, 2, 2)  # row, col, index
    pylab.plot(x, np.tan(x), 'r--')

    # - then, if new canvas (nrow and ncol) is created, the old canvas is not changed but is re-proportioned.
    # uncomment below to see:
    # pylab.subplot(2,2, 1)  # row, col, index
    # pylab.subplot(2,2, 2)  # row, col, index
    # pylab.subplot(2,2, 3)  # row, col, index
    # pylab.subplot(2,2, 4)  # row, col, index
    # pylab.subplot(2,2, 5)  # row, col, index # throw error since out of boundary
    # pylab.plot(x, np.cos(x), 'g*-')

    # -------------------------------------------------------------------------

    fig = pyplot.figure()  # holistic figure, can do "fig in fig"

    axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # posit fig 1
    axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])  # posit fig 2

    axes1.plot(x, 'r')  # draw fig 1
    axes2.grid(True)
    axes2.plot(y, 'g')  # draw fig 2

    # -------------------------------------------------------------------------

    fig = pyplot.figure(figsize=(16, 9), dpi=300)  # add size and dpi

    fig.add_subplot()  # new canvas

    pyplot.plot(y + 10, 'r', marker='o', markersize=10,
                markerfacecolor="blue")  ## marker -> draw on the face of the line
    pyplot.plot(y, 'g', linestyle='-')  ## linestyle -> it is the line itself

    # -------------------------------------------------------------------------

    fig = pyplot.figure(figsize=(16, 9), dpi=300)  # add size and dpi
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # posit fig 1
    axes.scatter(y, np.cos(y))

    # -------------------------------------------------------------------------

    fig = pyplot.figure(figsize=(16, 9), dpi=300)  # add size and dpi
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # posit fig 1
    axes.bar(y, np.cos(y))

    # -------------------------------------------------------------------------

    fig = pyplot.figure(figsize=(16, 9), dpi=300)  # add size and dpi
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # posit fig 1
    axes.step(y, np.cos(y))

    # -------------------------------------------------------------------------

    fig = pyplot.figure(figsize=(16, 9), dpi=300)  # add size and dpi
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # posit fig 1
    axes.fill_between(y, np.cos(y))

    # -------------------------------------------------------------------------

    fig = pyplot.figure(figsize=(16, 9), dpi=300)  # add size and dpi
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)  # posit fig 1
    axes.scatter(y, np.cos(y))

    # -------------------------------------------------------------------------

    fig = pyplot.figure(figsize=(16, 9), dpi=300)  # add size and dpi
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # posit fig 1
    axes.hist(y)

    # -------------------------------------------------------------------------

    fig = pyplot.figure(figsize=(16, 9), dpi=300)  # add size and dpi
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # posit fig 1
    axes.hist(y, cumulative=True, bins=5)

    # -------------------------------------------------------------------------
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-X ** 2 - Y ** 2)
    Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2

    print(X)
    print(Y)

    fig, ax = pyplot.subplots()
    CS = ax.contour(X, Y, Z)

    # -------------------------------------------------------------------------

    fig = pyplot.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=4)

    # -------------------------------------------------------------------------

    lang = ['Java', 'Python', 'C++', 'Ruby', 'Go']
    popularity = ['19.1', '18.2', '33.33', '7', '22.37']
    colors = ['red', 'blue', 'pink', 'yellow', 'green']
    explode = (0.1, 0, 0, 0, 0)
    # pyplot.pie(popularity, explode=explode, labels=lang, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
