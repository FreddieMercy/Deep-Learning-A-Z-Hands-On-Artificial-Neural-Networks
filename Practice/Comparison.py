from sklearn.datasets import load_iris, make_classification, make_multilabel_classification, load_linnerud
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, VotingRegressor, \
    StackingClassifier, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import neighbors, svm
from sklearn.linear_model import LogisticRegression, BayesianRidge, LassoLars, LinearRegression
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.calibration import CalibratedClassifierCV


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

    mb = MultinomialNB()

    mb.fit(data_train, target_train)

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
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=nca_knn, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))
    print(knn.score(data_test, target_test))
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=knn, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))
    print(nc.score(data_test, target_test))
    print(svc.score(data_test, target_test))
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=svc, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))
    print(bay.score(data_test, target_test))
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=bay, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))

    print(mb.score(data_test, target_test))
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=mb, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))

    print(dt_clf.score(data_test, target_test))
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=dt_clf, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))
    print(dt_reg.score(data_test, target_test))
    print(bagging.score(data_test, target_test))
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=bagging, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))
    print(rndForest.score(data_test, target_test))
    print(
        " -- calibrated: {}".format(
            CalibratedClassifierCV(base_estimator=rndForest, cv=5).fit(data_train, target_train).score(
                data_test,
                target_test)))
    print(exTree.score(data_test, target_test))
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=exTree, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))
    print(ada_clf.score(data_test, target_test))
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=ada_clf, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))
    print(gbc.score(data_test, target_test))
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=gbc, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))
    print(gbr.score(data_test, target_test))
    print(vc.score(data_test, target_test))
    print(vr.score(data_test, target_test))
    print(sc.score(data_test, target_test))
    print(
        " -- calibrated: {}".format(CalibratedClassifierCV(base_estimator=sc, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))
    print(sr.score(data_test, target_test))
    print(mulSVC.score(data_test, target_test))
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=mulSVC, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))
    print(oneSVC.score(data_test, target_test))
    print(" -- calibrated: {}".format(
        CalibratedClassifierCV(base_estimator=oneSVC, cv=5).fit(data_train, target_train).score(
            data_test,
            target_test)))

    # Not a lot of improvement with greater 'hidden_layer_sizes' ...
    mlpc100 = MLPClassifier(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(100,))
    mlpc200 = MLPClassifier(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(200,))
    mlpc300 = MLPClassifier(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(300,))
    mlpc400 = MLPClassifier(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(400,))
    mlpc500 = MLPClassifier(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(500,))
    mlpc600 = MLPClassifier(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(600,))
    mlpc700 = MLPClassifier(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(700,))
    mlpc800 = MLPClassifier(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(800,))

    mlpc100.fit(data_train, target_train)
    mlpc200.fit(data_train, target_train)
    mlpc300.fit(data_train, target_train)
    mlpc400.fit(data_train, target_train)
    mlpc500.fit(data_train, target_train)
    mlpc600.fit(data_train, target_train)
    mlpc700.fit(data_train, target_train)
    mlpc800.fit(data_train, target_train)

    print("\n$ MLPClassifier: ")
    print(mlpc100.score(data_test, target_test))
    print(mlpc200.score(data_test, target_test))
    print(mlpc300.score(data_test, target_test))
    print(mlpc400.score(data_test, target_test))
    print(mlpc500.score(data_test, target_test))
    print(mlpc600.score(data_test, target_test))
    print(mlpc700.score(data_test, target_test))
    print(mlpc800.score(data_test, target_test))

    # well, even worse ...
    mlpr100 = MLPRegressor(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(100,))
    mlpr200 = MLPRegressor(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(200,))
    mlpr300 = MLPRegressor(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(300,))
    mlpr400 = MLPRegressor(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(400,))
    mlpr500 = MLPRegressor(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(500,))
    mlpr600 = MLPRegressor(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(600,))
    mlpr700 = MLPRegressor(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(700,))
    mlpr800 = MLPRegressor(solver='lbfgs', alpha=0.05, max_iter=9000, random_state=1, hidden_layer_sizes=(800,))

    mlpr100.fit(data_train, target_train)
    mlpr200.fit(data_train, target_train)
    mlpr300.fit(data_train, target_train)
    mlpr400.fit(data_train, target_train)
    mlpr500.fit(data_train, target_train)
    mlpr600.fit(data_train, target_train)
    mlpr700.fit(data_train, target_train)
    mlpr800.fit(data_train, target_train)

    print("\n@ MLPRegressor: ")
    print(mlpr100.score(data_test, target_test))
    print(mlpr200.score(data_test, target_test))
    print(mlpr300.score(data_test, target_test))
    print(mlpr400.score(data_test, target_test))
    print(mlpr500.score(data_test, target_test))
    print(mlpr600.score(data_test, target_test))
    print(mlpr700.score(data_test, target_test))
    print(mlpr800.score(data_test, target_test))

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

    X, Y = make_multilabel_classification(n_samples=10, n_features=100, n_classes=3, random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7, random_state=1)

    # What is the use of it... at least I know # of jobs doesn't affect results

    mulOutputClass1 = MultiOutputClassifier(n_jobs=1, estimator=RandomForestClassifier(random_state=1))
    mulOutputClass2 = MultiOutputClassifier(n_jobs=2, estimator=RandomForestClassifier(random_state=1))
    mulOutputClass3 = MultiOutputClassifier(n_jobs=3, estimator=RandomForestClassifier(random_state=1))
    mulOutputClass4 = MultiOutputClassifier(n_jobs=4, estimator=RandomForestClassifier(random_state=1))
    mulOutputClass5 = MultiOutputClassifier(n_jobs=5, estimator=RandomForestClassifier(random_state=1))
    mulOutputClass6 = MultiOutputClassifier(n_jobs=6, estimator=RandomForestClassifier(random_state=1))
    mulOutputClass7 = MultiOutputClassifier(n_jobs=7, estimator=RandomForestClassifier(random_state=1))

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

    # Why the scores are always negative?

    A, B = load_linnerud(return_X_y=True)
    A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.7)

    mulOutputReg1 = MultiOutputRegressor(n_jobs=1, estimator=GradientBoostingRegressor(random_state=1))
    mulOutputReg2 = MultiOutputRegressor(n_jobs=2, estimator=GradientBoostingRegressor(random_state=1))
    mulOutputReg3 = MultiOutputRegressor(n_jobs=3, estimator=GradientBoostingRegressor(random_state=1))
    mulOutputReg4 = MultiOutputRegressor(n_jobs=4, estimator=GradientBoostingRegressor(random_state=1))
    mulOutputReg5 = MultiOutputRegressor(n_jobs=5, estimator=GradientBoostingRegressor(random_state=1))
    mulOutputReg6 = MultiOutputRegressor(n_jobs=6, estimator=GradientBoostingRegressor(random_state=1))
    mulOutputReg7 = MultiOutputRegressor(n_jobs=7, estimator=GradientBoostingRegressor(random_state=1))

    mulOutputReg1.fit(A_train, B_train)
    mulOutputReg2.fit(A_train, B_train)
    mulOutputReg3.fit(A_train, B_train)
    mulOutputReg4.fit(A_train, B_train)
    mulOutputReg5.fit(A_train, B_train)
    mulOutputReg6.fit(A_train, B_train)
    mulOutputReg7.fit(A_train, B_train)

    print("\n- MultiOutputRegressor: ")
    print(mulOutputReg1.score(A_test, B_test))
    print(mulOutputReg2.score(A_test, B_test))
    print(mulOutputReg3.score(A_test, B_test))
    print(mulOutputReg4.score(A_test, B_test))
    print(mulOutputReg5.score(A_test, B_test))
    print(mulOutputReg6.score(A_test, B_test))
    print(mulOutputReg7.score(A_test, B_test))
