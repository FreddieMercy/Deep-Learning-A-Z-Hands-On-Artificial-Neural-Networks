from sklearn import neighbors
import numpy as np

data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

# default n_neighbors = 5, it has to be >= data.length
nbrs = neighbors.NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(data)
distances_to_data, indicesOf_data = nbrs.kneighbors(data)

# length of each row of indicesOf_data is equal to n_neighbors
print(indicesOf_data)  # indices of data. (i.e: if indices is [1 0 3], it means data[1], data[0] and data[3])
# length of each row of distances_to_data is equal to n_neighbors
print(distances_to_data)  # distance to each point. i.e: if indices is [1 0 3], it means:
# [{distance to data[1]}, {distance to data[0]}, {distance to data[3]}])

print(nbrs.kneighbors_graph(data))  # (A, B) $(A and B are neighbors or not)
# i.e: there is a row: (2, 1) 1.0 -> are data[2] and data[1] neighbors or not? 1 means yes, 0 means no
# (or return the possibility from 0.0 to 1.0)

# nbrs can use KD Tree rather than ball tree, which looks like this:
kdt = neighbors.KDTree(data,
                       leaf_size=30,  # only effects performance, won't affect result.
                       metric='euclidean')
print(kdt.query(data,
                k=6,  # number of neighbors, can be at most data.length
                return_distance=False))

nc = neighbors.NearestCentroid()
from sklearn import svm

svc = svm.SVC()

data2 = np.array([np.arange(0, 9), np.arange(0, 9)])
data2 = data2.T
target2 = np.arange(0, 9)

nc.fit(data2, target2)
svc.fit(data2, target2)

print(nc.predict([[100000, -100]]))
print(svc.predict([[100000, -100]]))

knt_d = neighbors.KNeighborsTransformer(n_neighbors=1, mode='distance')
# prints: (point A, point B) $(distance)
print(knt_d.fit_transform(data2, target2))  # either (point itself, point itself), or (point A, point B)

# prints: (point A, point B) $(is neighbor or not: 1 == yes, 0 == no). Just like the kneighbors_graph
knt_c = neighbors.KNeighborsTransformer(n_neighbors=1, mode='connectivity')
print(knt_c.fit_transform(data2, target2))

from sklearn.pipeline import make_pipeline
from sklearn.manifold import Isomap

estimator = make_pipeline(
    neighbors.KNeighborsTransformer(n_neighbors=5, mode='distance'),
    # Isomap is a 降维 method, 用来提速
    Isomap(neighbors_algorithm='precomputed'))
