from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn import neighbors, svm
from sklearn.linear_model import LogisticRegression


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
