def imputationPractice():
    import random
    import numpy as np
    from sklearn.impute import SimpleImputer, KNNImputer

    def splitTrainTest(limit=5, gap=5, seed=0):
        expected = []
        matrix = []

        itera = 0

        random.seed(seed)

        for r in range(0, limit):
            expected.append([i for i in range(0, limit)])
            matrix.append([i for i in range(0, limit)])
            for c in range(0, limit):
                itera += gap
                expected[r][c] = itera
                if random.randint(0, 100) > 20:
                    matrix[r][c] = itera
                else:
                    matrix[r][c] = np.nan

        return expected, matrix

    def evaluateImputation(expected, matrix):
        score = 0
        total = 0
        for r in range(0, len(matrix)):
            for c in range(0, len(matrix[0])):
                if matrix[r][c] == expected[r][c]:
                    score += 1
                total += 1

        return (score / total) * 100

    expected, matrix = splitTrainTest()

    print(expected)
    print(matrix)

    print()

    simple = SimpleImputer(missing_values=np.nan)

    print(evaluateImputation(simple.fit_transform(matrix), expected))
    print(evaluateImputation(KNNImputer(n_neighbors=2).fit_transform(matrix), expected))

    print(
        "=============================================================================================================")

    knn1 = KNNImputer(n_neighbors=1)
    knn2 = KNNImputer(n_neighbors=2)
    knn3 = KNNImputer(n_neighbors=3)
    knn4 = KNNImputer(n_neighbors=4)
    knn5 = KNNImputer(n_neighbors=5)
    knn6 = KNNImputer(n_neighbors=6)

    print("* KNNImputer:\n")

    print(evaluateImputation(knn1.fit_transform(matrix), expected))
    print(evaluateImputation(knn2.fit_transform(matrix), expected))
    print(evaluateImputation(knn3.fit_transform(matrix), expected))
    print(evaluateImputation(knn4.fit_transform(matrix), expected))
    print(evaluateImputation(knn5.fit_transform(matrix), expected))
    print(evaluateImputation(knn6.fit_transform(matrix), expected))
