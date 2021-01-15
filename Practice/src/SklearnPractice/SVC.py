def svcPractice():
    from sklearn import svm  # S.V. Machine (Support Vector Machine)
    import numpy as np

    data = np.array([np.arange(0, 9), np.arange(0, 9)])
    data = data.T
    target = np.arange(0, 9)

    print(data)
    print(target)

    clf = svm.SVC()  # decision_function_shape = 'ovr' by default. ('ovr' means 'one-versus-rest')

    clf.fit(data, target)

    print(clf.predict([[100000, -100]]))

    print(clf.support_)  # target
    print(clf.support_vectors_)  # data
    print(clf.n_support_)  # idk...

    print(clf.decision_function(data))  # SVC uses 'decision_function' to predict result

    clf.decision_function_shape = 'ovo'  # one-versus-one, so a lot of classifier will be created.
    clf.fit(data, target)

    print(clf.predict([[100000, -100]]))

    print(clf.support_)  # target
    print(clf.support_vectors_)  # data
    print(clf.n_support_)  # idk...
    print(clf.decision_function(data))

    # in the other hand, LinearSVC:

    lin_clf = svm.LinearSVC(max_iter=10000)
    lin_clf.fit(data, target)
    print(lin_clf.decision_function(data))
    print(lin_clf.predict(data))

    # Regression

    reg = svm.SVR()  # support vector regression, a regression bases on svc
    reg.fit(data, target)
    print(reg.predict([[100000, -100]]))

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    s_svc = make_pipeline(StandardScaler(), svm.SVC())
    s_lsvc = make_pipeline(StandardScaler(), svm.LinearSVC())
    s_reg = make_pipeline(StandardScaler(), svm.SVR())

    s_svc.fit(data, target)
    s_lsvc.fit(data, target)
    s_reg.fit(data, target)

    print(clf.predict([[100000, -100]]))
    print(lin_clf.predict([[100000, -100]]))
    print(reg.predict([[100000, -100]]))

    print(s_svc.predict([[100000, -100]]))
    print(s_lsvc.predict([[100000, -100]]))
    print(s_reg.predict([[100000, -100]]))

    # kernel function 大约等于 decision function

    clf_linear = svm.SVC(kernel='linear')
    clf_rbf = svm.SVC(kernel='rbf')
    clf_poly = svm.SVC(kernel='poly')
    clf_sig = svm.SVC(kernel='sigmoid')

    clf_linear.fit(data, target)
    clf_rbf.fit(data, target)
    clf_poly.fit(data, target)
    clf_sig.fit(data, target)

    print(clf_linear.predict([[100000, -100]]))
    print(clf_rbf.predict([[100000, -100]]))
    print(clf_poly.predict([[100000, -100]]))
    print(clf_sig.predict([[100000, -100]]))

    def my_kernel(X, Y):
        return np.dot(X, Y.T)

    clf_custom = svm.SVC(kernel=my_kernel)
    clf_custom.fit(data, target)

    print(clf_custom.predict([[100000, -100]]))

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    print("\n\nmake_classification:\n\n")
    data_m, target_m = make_classification(n_samples=13, n_features=17)  # generate 13 by 17 matrix
    train_data, test_data, train_target, test_target = train_test_split(data_m, target_m)

    print(train_data)
    print(train_target)
    print(test_data)

    clf_prec = svm.SVC(kernel='precomputed')  # （我要）自带一个kernel（或者换句话说：我要自带一个decision function）
    pseudo_decision_function = np.dot(train_data, train_data.T)  # for example....
    clf_prec.fit(pseudo_decision_function, train_target)
    print(clf_prec.predict(np.dot(test_data, train_data.T)))

    print(test_target)
