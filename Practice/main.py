from sklearn import datasets, neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

iris = datasets.load_iris()
x = iris.data
y = iris.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # so train size -> 1 - 0.2 = 0.8?

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(x_test.shape)

data = pd.read_csv('data/data.txt', delimiter='\t')

print(data)
