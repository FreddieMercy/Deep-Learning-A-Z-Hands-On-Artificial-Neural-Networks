from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

data_i, targeti = load_iris(return_X_y=True)
data_train, data_test, target_train, target_test = train_test_split(data_i, targeti, random_state=42,
                                                                    stratify=targeti, test_size=0.7)

zero_to_two = np.arange(0, 3)
lb = LabelBinarizer()
print(zero_to_two)

# tmp = [0,1,2]
# [1, 0 ,0] means first item in tmp (a.k.a: 0)
# [0, 1 ,0] means second item in tmp (a.k.a: 1)
# [0, 0 ,1] means third item in tmp (a.k.a: 2)
print(lb.fit_transform(zero_to_two))
