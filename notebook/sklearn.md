# sklearn

## 内置数据集

```python
from sklearn import datasets
diabetes = datasets.load_diabetes().data # ndarray类型数据
```



## 线性回归

推导：误差本身符合独立同分布的均值为0的正太分布，对数似然函数求最大得最佳参数

```python

```

## 决策树

```python
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
```

Parameters：

- criterion

  决策树构建原则可使用 gini 或 entropy

- splitter

  best   在所有特征当中选最好的切分点

  random    在随机的特征当中找切分点（数据量较大时使用），多用于连续值特征最优点切分

- max_features

  使用特征个数：None(所有)，log2, sqrt, N 特征小于50的时候使用所有的

- max_depth

  最大深度，数据少或者特征少时可以不用设置，如果特征和样本较多可以尝试限制作为预剪枝

- min_samples_split

  如果某节点的样本数少于此值，则不会继续尝试选择最优特征来进行划分；如果样本量不大不需要设置此值，如果样本量非常大，则推荐增大此值。

- min_samples_leaf

  限制叶子节点最少的样本数，如果某叶子节点样本数小于该值，则会和兄弟节点一起被剪枝；如果样本量不大则不需要管此值，大些如10w可以尝试下5

- min_weight_fraction_leaf

  这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝，默认值为0，即不考虑权重问题；一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差较大，就会引入样本权重，这是我们就需要注意这个值了。

- max_leaf_node

  限制最大的叶子节点数，可以防止过拟合，默认是None，即不限制最大的叶子节点数。如果加了限制，算法会在最大叶子节点数内建立最优的决策树；如果特征不多可以不考虑此值，如果特征较多时，可以加以限制，具体的值可以通过交叉验证得到。

- class_weight

  指定样本各类别的权重，主要为了防止训练集某些类别的样本过多，导致训练决策树过于偏向这些类别，这里可以指定各个样本的权重，如果使用’balance‘ ，则算法会自己计算权重，样本量较少的类别所对应的样本权重会高。

- min_impurity_split

  这个值限制了决策树的生长，如果某节点的不纯度（基尼系数，信息增益，均方差，绝对差）小于这个阈值则该节点不在生成子节点，即为叶子节点。

  ​

  ```python
  from sklearn.tree import DecisionTreeClassifier # 导入决策树分类器
  tree = DecisionTreeClassifier()
  tree.fit(train_x,train_y)
  tree.score(test_x,test_y)
  ```

  ​

  ```python
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.model_selection import cross_val_score
  >>> from sklearn.tree import DecisionTreeClassifier
  >>> clf = DecisionTreeClassifier(random_state=0)
  >>> iris = load_iris()
  >>> cross_val_score(clf, iris.data, iris.target, cv=10)
  ...                             
  ...
  array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
          0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
  ```

- 可视化

  ```python
  import pydotplus
  from Ipython.display import display, Image
  dot_data = tree.export_graphviz(clf, 
                                  out_file=None, 
                                  feature_names=features.columns,
                                  class_names = ['y1','y2'],
                                  filled=True,
                                  rounded=True)
  graph = pydotplus.graph_from_dot_data(dot_data)
  display(Image(graph.create_png()))r
  ```

  ​

