from sklearn.feature_extraction import text
import pandas as pd

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

doc = pd.read_csv("data/raw.txt", delimiter='\t')

counter = text.CountVectorizer()
counter.fit(corpus)
names = counter.get_feature_names()

print(
    counter.transform(
        corpus).toarray())  # has four rows, same as 'corpus', and these numbers mean the # of occurance of the string in 'names' on that index
print(names)
print("-----------------------------------------------------------")
vector = text.TfidfVectorizer()
print(vector.fit_transform(doc))  # some texts may be filtered out b/c the algorithm thinks these are noises.
# "I" will be filtered out because "Fredd'i'e" has 'i'
print(vector.get_feature_names())
print("-----------------------------------------------------------")
transformer = text.TfidfTransformer()
print(transformer.fit_transform(vector.fit_transform(doc)))
print("-----------------------------------------------------------")
counter2 = text.CountVectorizer()
print(counter2.fit_transform(doc))  # see, same as 'transformer', which is TfidfVectorizer + TfidfTransformer
