from sklearn.datasets import load_iris, make_classification, make_multilabel_classification
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor, \
    StackingClassifier, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import neighbors, svm
from sklearn.linear_model import LogisticRegression, BayesianRidge, LassoLars, LinearRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier


def Comparison():
    data_i, targeti = load_iris(return_X_y=True)
    data_train, data_test, target_train, target_test = train_test_split(data_i, targeti, random_state=42,
                                                                        stratify=targeti, test_size=0.7)

    nca = neighbors.NeighborhoodComponentsAnalysis(random_state=42)
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)

    # 重新起一个，不然会打脸的 - -||
    # nca_knn = Pipeline([('nca', nca), ('knn', knn)])

    nca_knn = Pipeline([('nca', neighbors.NeighborhoodComponentsAnalysis(random_state=42)),
                        ('knn', neighbors.KNeighborsClassifier(n_neighbors=3))])
    nca_knn.fit(data_train, target_train)
    # nca_knn.fit(data2, target2)
    knn.fit(data_train, target_train)

    nc = neighbors.NearestCentroid()
    svc = svm.SVC()
    nc.fit(data_train, target_train)
    svc.fit(data_train, target_train)

    bay = GaussianNB()

    bay.fit(data_train, target_train)

    from sklearn import tree

    dt_clf = tree.DecisionTreeClassifier()

    dt_clf.fit(data_train, target_train)

    dt_reg = tree.DecisionTreeRegressor()
    dt_reg.fit(data_train, target_train)

    bagging = BaggingClassifier(max_samples=0.5,  # half rows
                                max_features=0.5,  # half columns
                                base_estimator=KNeighborsClassifier())

    # better than Bagging: can define how many estimators, rather than unknown number of estimators
    # worse than Bagging: cannot define the base_estimator

    rndForest = RandomForestClassifier(n_estimators=10)

    exTree = ExtraTreesClassifier(max_samples=0.5,  # half rows
                                  max_features=0.5,  # half columns
                                  n_estimators=10)

    bagging.fit(data_train, target_train)
    rndForest.fit(data_train, target_train)
    exTree.fit(data_train, target_train)

    ada_clf = AdaBoostClassifier(n_estimators=100)  # strengthen weakness
    ada_clf.fit(data_train, target_train)

    gbc = GradientBoostingClassifier(n_estimators=100)
    gbc.fit(data_train, target_train)

    gbr = GradientBoostingRegressor(n_estimators=100)
    gbr.fit(data_train, target_train)

    # hard vote: 少数服从多数， 如果平票，那按字母排列选第一个
    # soft vote：take average
    vc = VotingClassifier(
        estimators=[("someone", rndForest), ("bagging", bagging), ("LogisticRegression", LogisticRegression()),
                    ("SVC", svm.SVC())],  # like Pipeline
        voting="hard")

    vc.fit(data_train, target_train)

    vr = VotingRegressor(
        estimators=[("BayesianRidge", BayesianRidge()), ("LassoLars", LassoLars(alpha=0.05)),
                    ("LinearRegression", LinearRegression()),
                    ("SVR", svm.SVR())],
        weights=[2, 1, 2, 1])

    vr.fit(data_train, target_train)

    sc = StackingClassifier(
        estimators=[("someone", rndForest), ("bagging", bagging), ("LogisticRegression", LogisticRegression())],  # FIFO
        final_estimator=svm.SVC(), cv=5)  # final_estimator is trained by cross validation

    sc.fit(data_train, target_train)

    sr = StackingRegressor(
        estimators=[("BayesianRidge", BayesianRidge()), ("LassoLars", LassoLars(alpha=0.05)),
                    ("LinearRegression", LinearRegression())],
        final_estimator=svm.SVR(), cv=5)

    sr.fit(data_train, target_train)

    # OneVsRestClassifier and OneVsOneClassifier are decorator that decorates the 'estimator'
    # mulSVC should be same to svc (svc by default is ovr), 但怎么打脸了，还不容易converge？
    mulSVC = OneVsRestClassifier(estimator=svm.LinearSVC(random_state=0, max_iter=9000))
    mulSVC.fit(data_train, target_train)

    # 哦，看起来我之前错了：svc by default is ovo。
    # oneSVC should be same to svc
    oneSVC = OneVsOneClassifier(estimator=svm.LinearSVC(random_state=0, max_iter=9000))
    oneSVC.fit(data_train, target_train)

    print(nca_knn.score(data_test, target_test))
    print(knn.score(data_test, target_test))
    print(nc.score(data_test, target_test))
    print(svc.score(data_test, target_test))
    print(bay.score(data_test, target_test))
    print(dt_clf.score(data_test, target_test))
    print(dt_reg.score(data_test, target_test))
    print(bagging.score(data_test, target_test))
    print(rndForest.score(data_test, target_test))
    print(exTree.score(data_test, target_test))
    print(ada_clf.score(data_test, target_test))
    print(gbc.score(data_test, target_test))
    print(gbr.score(data_test, target_test))
    print(vc.score(data_test, target_test))
    print(vr.score(data_test, target_test))
    print(sc.score(data_test, target_test))
    print(sr.score(data_test, target_test))
    print(mulSVC.score(data_test, target_test))
    print(oneSVC.score(data_test, target_test))

    # OutputCodeClassifier is also a decorator that decorates the 'estimator'
    # The most important part is: user can define how many classes are there (code_size)

    outputClass1 = OutputCodeClassifier(code_size=1, estimator=svm.SVC())
    outputClass2 = OutputCodeClassifier(code_size=2, estimator=svm.SVC())
    outputClass3 = OutputCodeClassifier(code_size=3, estimator=svm.SVC())
    outputClass4 = OutputCodeClassifier(code_size=4, estimator=svm.SVC())
    outputClass5 = OutputCodeClassifier(code_size=5, estimator=svm.SVC())
    outputClass6 = OutputCodeClassifier(code_size=6, estimator=svm.SVC())
    outputClass7 = OutputCodeClassifier(code_size=7, estimator=svm.SVC())

    outputClass1.fit(data_train, target_train)
    outputClass2.fit(data_train, target_train)
    outputClass3.fit(data_train, target_train)
    outputClass4.fit(data_train, target_train)
    outputClass5.fit(data_train, target_train)
    outputClass6.fit(data_train, target_train)
    outputClass7.fit(data_train, target_train)

    print("\n* OutputCodeClassifier: ")
    print(outputClass1.score(data_test, target_test))
    print(outputClass2.score(data_test, target_test))
    print(outputClass3.score(data_test, target_test))
    print(outputClass4.score(data_test, target_test))
    print(outputClass5.score(data_test, target_test))
    print(outputClass6.score(data_test, target_test))
    print(outputClass7.score(data_test, target_test))

    X, Y = make_multilabel_classification(n_samples=10, n_features=100, n_classes=3)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7)

    # What is the use of it... at least I know # of jobs doesn't affect results

    mulOutputClass1 = MultiOutputClassifier(n_jobs=1, estimator=RandomForestClassifier())
    mulOutputClass2 = MultiOutputClassifier(n_jobs=2, estimator=RandomForestClassifier())
    mulOutputClass3 = MultiOutputClassifier(n_jobs=3, estimator=RandomForestClassifier())
    mulOutputClass4 = MultiOutputClassifier(n_jobs=4, estimator=RandomForestClassifier())
    mulOutputClass5 = MultiOutputClassifier(n_jobs=5, estimator=RandomForestClassifier())
    mulOutputClass6 = MultiOutputClassifier(n_jobs=6, estimator=RandomForestClassifier())
    mulOutputClass7 = MultiOutputClassifier(n_jobs=7, estimator=RandomForestClassifier())

    mulOutputClass1.fit(X_train, Y_train)
    mulOutputClass2.fit(X_train, Y_train)
    mulOutputClass3.fit(X_train, Y_train)
    mulOutputClass4.fit(X_train, Y_train)
    mulOutputClass5.fit(X_train, Y_train)
    mulOutputClass6.fit(X_train, Y_train)
    mulOutputClass7.fit(X_train, Y_train)

    print("\n+ MultiOutputClassifier: ")
    print(mulOutputClass1.score(X_test, Y_test))
    print(mulOutputClass2.score(X_test, Y_test))
    print(mulOutputClass3.score(X_test, Y_test))
    print(mulOutputClass4.score(X_test, Y_test))
    print(mulOutputClass5.score(X_test, Y_test))
    print(mulOutputClass6.score(X_test, Y_test))
    print(mulOutputClass7.score(X_test, Y_test))
