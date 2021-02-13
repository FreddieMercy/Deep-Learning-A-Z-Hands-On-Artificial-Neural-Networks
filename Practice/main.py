from sklearn.preprocessing import OneHotEncoder

import pandas as pd

doc = pd.read_csv("data/raw/raw2.txt", delimiter='\t')

oneEncoder = OneHotEncoder(drop=None)
print(doc)
print(oneEncoder.fit_transform(doc).toarray())  # somehow ignores the first line
print(oneEncoder.categories_)
