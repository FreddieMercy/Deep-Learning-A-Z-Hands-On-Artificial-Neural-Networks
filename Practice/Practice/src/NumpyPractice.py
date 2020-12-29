import numpy as np


def numpyPractice():
    a = np.array([1, 2, 3])
    print(a)

    b = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]])
    print(b)

    print(a.ndim)  # number of dimension of array "a"

    print(b.shape)  # shape = ({# of dimension}, {len of each item})

    print(b[1, 1:-1:2])  # second row, exclude first column and skip 1 each time

    # replace
    # b[:, 1, :] = [[9, 9, 9], [8, 8]]  # suppose to fail
    # b[:, 1, :] = 4 # suppose to fail

    b[:, 1] = 4
    b[:, 1] = [5]

    allZeros2by3by3by2 = np.zeros((2, 3, 3, 2))
    allOnes4by2by2in32 = np.ones((4, 2, 2), dtype='int32')

    all99in2by2Float = np.full((2, 2), 99, dtype='float32')

    creatFrom_Attrs = np.full(a.shape, 5)  # np.full((1,3), 5)
    print(creatFrom_Attrs)

    creatFrom_Like = np.full_like(a, 5)
    print(creatFrom_Like)
    # print(a)

    rndNumSample = np.random.random_sample((5, 3))
    print(rndNumSample)

    rndNumArr = np.random.rand(5, 3)  # Note: its input is two individuals, and random_sample's input is tuple
    print(rndNumArr)

    rndNumSth = np.random.randint(4, 7, size=(5, 3))  # randint: (from) 4 (to) 7 in 5 by 3
    print(rndNumSth)

    print(np.identity(5))  # identity matrix

    arr = np.array([1, 2, 3])
    repeat1d = np.repeat(arr, 3)
    repeatNd_axis_1 = np.repeat([arr], 3)
    repeatNd_axis_0 = np.repeat([arr], 3, axis=0)

    print(repeat1d)
    print(repeatNd_axis_1)
    print(repeatNd_axis_0)

    c = np.array([1, 2, 3])  # np array copy: pass by reference
    d = c
    d[0] = 100

    print(c)

    e = np.array([1, 2, 3])
    f = e.copy()  # use *.copy() instead
    f[0] = 100

    print(e)

    # And can do math ops to np arr directly, just like R
    # g = [1, 2, 3] + 100 # throws error

    e += 100
    print(e)

    # Linear algebra

    m1 = np.ones((2, 3))
    m2 = np.full((3, 2), 2)

    print(np.matmul(m1, m2))  # stands for "matrix multiply" : multiply m1 and m2

    m3 = np.identity(3)
    print(np.
          linalg.  # stands for: lin(ear )alg(ebra)
          det  # determinant
          (m3))

    # np.min(), np.max(), np.sin(), np.cos()

    # reorganize
    original = np.array([[2, 2, 2, 2], [2, 2, 2, 2]])

    reshaped = original.reshape((2, 2, 2))
    print(reshaped)

    print(np.vstack([original, original, original,
                     original]))  # vstack == vertical stack, all the input arrays must have same number of dimensions
    filedata = np.genfromtxt('data/data.txt', delimiter=',')
    print(filedata)

    print("np.any(filedata > 50, axis=0):\n")
    print(np.any(filedata > 50, axis=0))

    print()

    print("np.any(filedata > 50, axis=1)")
    print(np.any(filedata > 50, axis=1))

    print()

    print("np.any(filedata > 50)")
    print(np.any(filedata > 50))

    print()

    print("np.all(filedata > 50, axis=0):\n")
    print(np.all(filedata > 50, axis=0))

    print()

    print("np.all(filedata > 50, axis=1)")
    print(np.all(filedata > 50, axis=1))

    print()

    print("np.all(filedata > 50)")
    print(np.all(filedata > 50))
