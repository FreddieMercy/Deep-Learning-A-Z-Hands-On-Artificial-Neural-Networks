from sklearn import svm  # S.V. Machine (Support Vector Machine)

data = [[0, 0], [1, 1]]
target = [0, 1]

clf = svm.SVC()  # why no alpha?
clf.fit(data, target)
