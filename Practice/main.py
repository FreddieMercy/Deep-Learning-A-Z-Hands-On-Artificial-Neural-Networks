from src.MatplotlibPractice import matplotlibPractice
from src.NumpyPractice import numpyPractice
from src.PandasPractice import pandasPractice
from src.SklearnPractice.LinearRegression import linearRegressionPractice
from src.SklearnPractice.Others import otherPractice
from src.SklearnPractice.SVC import svcPractice
from src.SklearnPractice.NearestNeighbors import nearestNeighborsPractice
from src.SklearnPractice.DecisionTree import decisionTreePractice
from src.SklearnPractice.Ensemble import ensemblePractice
from Comparison import Comparison


def main():
    numpyPractice()
    matplotlibPractice()
    pandasPractice()
    linearRegressionPractice()
    svcPractice()
    nearestNeighborsPractice()
    decisionTreePractice()
    ensemblePractice()
    otherPractice()


main()
print("--------------------------------------------------------------------")
Comparison()
