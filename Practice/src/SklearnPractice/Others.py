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
