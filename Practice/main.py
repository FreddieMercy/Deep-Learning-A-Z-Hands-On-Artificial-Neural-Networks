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

est_orig = svm.SVC()
est_new = svm.SVC()

est_orig.fit(X, Y)
print("original: ")
