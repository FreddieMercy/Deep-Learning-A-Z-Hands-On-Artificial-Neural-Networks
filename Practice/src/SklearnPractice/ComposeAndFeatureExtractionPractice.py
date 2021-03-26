def composeAndFeatureExtractionPracticeTreePractice():
    from sklearn.feature_extraction import text
    import pandas as pd

    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]

    doc = pd.read_csv("data/raw/raw1.txt", delimiter='\t')

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

    print("-----------------------------------------------------------")

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import Normalizer

    nums = [[0., 1., 2., 2., 123.],
            [1., 1., 0., 1., 456.],
            [0., 0., 1., 2., 789.]]

    print(ColumnTransformer(
        # [("norm1", Normalizer(norm='l1'), [0, 1, 4]),
        [("norm1", Normalizer(norm='l1'), [0, 1]),  # transform column 0 and 1
         ("norm2", Normalizer(norm='l1'), slice(2, 4))  # transform column 2 to 3 (because column 4 is not inclusive)
         # so, column 4 will not be included
         ]).fit_transform(nums))
