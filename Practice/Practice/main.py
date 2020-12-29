import numpy as np
from matplotlib import pylab, pyplot

x = np.linspace(0, 10, 25)  # random number within range 0 ~ 10, len(x) = 25
print(x)
y = x ** 2 + 2
print(y)

print(
    np.array([x, y])  # 2 by 25 array
        .reshape(25, 2)  # reshape to 25 by 2 (25 rows, 2 columns)
        # so, looks like "reshape" chops from left to right, top to btm, index 0 recursively
        .reshape(2, 25)
)

pylab.plot(x, y, 'r')
fig, axis = pyplot.subplots(2, 3)  # build a context: 2 by 3, will display 6 plot figs.

# ???
# axis[?,?] = fig.add_axes([.....])
# ???

x = np.linspace(0, 8, 1000)

axis[0, 0].plot(x, np.sin(x), 'g')  # row=0, col=0
axis[0, 1].plot(range(100), 'b')  # row=0, col=1
axis[0, 2].plot(x + 100, 'b')  # row=0, col=2
axis[1, 0].plot(x, np.tan(x), 'k')  # row=1, col=0
axis[1, 1].plot(x, np.cos(x), 'r')  # row=1, col=1

# No row=1, col=2, so it will be blank

fig.show()  # pyplot.figure().show()

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
