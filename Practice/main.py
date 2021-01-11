from sklearn import model_selection, linear_model
from matplotlib import pyplot

reg = linear_model.LinearRegression()  # Like R, reg should be a line
data = [[0, 0], [1, 1], [2, 2]]
target = [0, 1, 2]
reg.fit(data, target)

print(reg.coef_)

pyplot.plot(data, reg.predict(data))
pyplot.show()
