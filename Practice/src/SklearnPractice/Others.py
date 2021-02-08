def otherPractice():
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import numpy as np

    zero_to_two = np.arange(0, 3)
    lb = LabelBinarizer()
    print(zero_to_two)

    # tmp = [0,1,2]
    # [1, 0 ,0] means first item in tmp (a.k.a: 0)
    # [0, 1 ,0] means second item in tmp (a.k.a: 1)
    # [0, 0 ,1] means third item in tmp (a.k.a: 2)
    print(lb.fit_transform(zero_to_two))

    from sklearn.feature_selection import VarianceThreshold

    X = [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1],
         [0, 1, 1],
         [0, 0, 1],
         [0, 0, 0]]

    sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))  # remove all columns that are more than 80% same

    print(sel.fit_transform(X))

    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectKBest, \
        chi2  # chi2 is a math formula: https://www.google.com/search?q=chi2&oq=chi2&aqs=chrome..69i57j0i10i457j0l2j0i10j0j0i10j0i395l3.261j1j9&sourceid=chrome&ie=UTF-8#wptab=s:H4sIAAAAAAAAAONgVuLSz9U3MDItNitJfsToyi3w8sc9YSmbSWtOXmM04-IKzsgvd80rySypFNLgYoOy5Lj4pJC0aTBI8XAh8Xl2MUm6ppQmJ5Zk5ucl5jjn5yWnFpS45RflluYkLmKVSM7I1C0uLE0sSk1RKEktLlFIg0gBAOCLz8-NAAAA

    X, Y = load_iris(return_X_y=True)

    print(X)

    kBest_1 = SelectKBest(chi2, k=1)
    kBest_2 = SelectKBest(chi2, k=2)
    kBest_3 = SelectKBest(chi2, k=3)
    kBest_4 = SelectKBest(chi2, k=4)

    print(kBest_1.fit_transform(X, Y))
    print(kBest_2.fit_transform(X, Y))
    print(kBest_3.fit_transform(X, Y))
    print(kBest_4.fit_transform(X, Y))

    from sklearn.feature_selection import SelectFromModel
    from sklearn import svm
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, train_size=0.7)

    svc = svm.LinearSVC(max_iter=9000)

    svc.fit(X_train, Y_train)
    print("score of original: ", svc.score(X_test, Y_test))

    X_new = SelectFromModel(svc, prefit=True).transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, random_state=1, train_size=0.7)

    svc.fit(X_train, Y_train)
    print("score of new: ", svc.score(X_test, Y_test))

    print(X)
    print(X_new)

    from sklearn.model_selection import KFold

    arr = [i for i in range(0, 10)]

    # print(arr)

    nFold = KFold(n_splits=5)  # split to n rows, each row train : test rate is (total - total/n) : total/n
    # i.e: split to 5 rows, each row train : test rate is 8 : 2
    # i.e: split to 2 rows, each row train : test rate is 5 : 5
    for train, test in nFold.split(arr):
        print("%s %s" % (train, test))

    from sklearn.model_selection import RepeatedKFold

    rnFolder = RepeatedKFold(n_splits=2,
                             n_repeats=2)  # split to n_splits * n_repeats rows, each row train : test rate is (total - total/n) : total/n
    # i.e: split to 5*2 rows, each row train : test rate is 8 : 2
    # i.e: split to 2*2 rows, each row train : test rate is 5 : 5
    for train, test in rnFolder.split(arr):
        print("%s %s" % (train, test))

    from sklearn.model_selection import LeaveOneOut, LeavePOut

    oneOut = LeaveOneOut()

    # len(arr) rows
    for train, test in oneOut.split(arr):
        print("%s %s" % (train, test))

    pOut = LeavePOut(p=2)

    # p * len(arr) rows
    for train, test in pOut.split(arr):
        print("%s %s" % (train, test))
