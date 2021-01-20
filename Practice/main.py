from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
# ensemble means "vote"
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

data_i, targeti = load_iris(return_X_y=True)
data_train, data_test, target_train, target_test = train_test_split(data_i, targeti, random_state=42,
                                                                    stratify=targeti, test_size=0.7)

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

print(cross_val_score(bagging, data_test, target_test, cv=5).mean())
print(cross_val_score(rndForest, data_test, target_test, cv=5).mean())
print(cross_val_score(exTree, data_test, target_test, cv=5).mean())

print(cross_val_predict(bagging, data_test, target_test, cv=5))
print(cross_val_predict(rndForest, data_test, target_test, cv=5))
print(cross_val_predict(exTree, data_test, target_test, cv=5))
