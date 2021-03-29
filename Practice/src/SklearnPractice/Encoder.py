from sklearn.preprocessing import LabelBinarizer

def encoderPractice():
    from numpy import ravel
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, LabelBinarizer, MultiLabelBinarizer

    import pandas as pd

    doc = pd.read_csv("data/raw/raw2.txt", delimiter='\t')

    oneEncoder = OneHotEncoder(drop=None)  # 按图索骥

    print(doc)
    print(oneEncoder.fit_transform(doc).toarray())  # somehow ignores the first line
    print(oneEncoder.categories_)
    print("------------------------------------------------------------")
    labelEncoder = LabelEncoder()  # horizontal
    print(labelEncoder.fit_transform(ravel(doc)))
    print(list(labelEncoder.classes_))

    print("------------------------------------------------------------")

    ordinalEncoder = OrdinalEncoder()  # vertical
    print(ordinalEncoder.fit_transform(doc))
    print(ordinalEncoder.categories_)

    print("------------------------------------------------------------")

    labelBinarizer = LabelBinarizer()  # 真·二维，干嘛这么费劲儿
    print(labelBinarizer.fit_transform(doc))
    print(list(labelBinarizer.classes_))

    print("------------------------------------------------------------")

    multiLabelBinarizer = MultiLabelBinarizer()  # I really don't know...
    print(multiLabelBinarizer.fit_transform(doc))

    print(list(multiLabelBinarizer.classes_))