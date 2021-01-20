from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
# ensemble means "vote"
from sklearn.neighbors import KNeighborsClassifier

bagging = BaggingClassifier(max_samples=0.5,  # half rows
                            max_features=0.5,  # half columns
                            base_estimator=KNeighborsClassifier())

# better than Bagging: can define how many estimators, rather than unknown number of estimators
# worse than Bagging: cannot define the base_estimator

rndForest = RandomForestClassifier(n_estimators=10)

exTree = ExtraTreesClassifier(max_samples=0.5,  # half rows
                              max_features=0.5,  # half columns
                              n_estimators=10)
