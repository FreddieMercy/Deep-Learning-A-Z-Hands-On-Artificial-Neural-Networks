from sklearn import tree

dt_clf = tree.DecisionTreeClassifier()
data = [[0, 0], [1, 1], [2, 2], [3, 3]]
target = [0, 1, 2, 3]

dt_clf.fit(data, target)

print(dt_clf.predict([[100., -10.]]))
print(dt_clf.predict_proba([[100., -10.]]))  # probability of being one of the $(target.length) classes
tree.plot_tree(dt_clf)

print(tree.export_text(dt_clf))

import graphviz

graph = graphviz.Source(tree.export_graphviz(dt_clf))  # 'Source' means 'import/load'?
graph.render("tree_graph")
