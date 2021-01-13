from sklearn import svm  # S.V. Machine (Support Vector Machine)
import numpy as np

data = np.array([np.arange(0, 9), np.arange(0, 9)])
data = data.T
target = np.arange(0, 9)

print(data)
print(target)

clf = svm.SVC()  # why no alpha?
clf.fit(data, target)

print(clf.predict([[100000, -100]]))
