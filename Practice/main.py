from numpy import ravel
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import pandas as pd

doc = pd.read_csv("data/raw/raw2.txt", delimiter='\t')

oneEncoder = OneHotEncoder(drop=None)
print(doc)
print(oneEncoder.fit_transform(doc).toarray())  # somehow ignores the first line
print(oneEncoder.categories_)
print("------------------------------------------------------------")
labelEncoder = LabelEncoder()
print(labelEncoder.fit_transform(ravel(doc)))
print(list(labelEncoder.classes_))
