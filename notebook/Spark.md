# Spark

## Spark 核心概念和操作

spark可以分为1个driver（笔记本电脑或者集群网关机器上）和若干executor在（各个节点上），通过SparkContext（简称sc）连接Spark集群、创建RDD、累加器(accumlaor)、广播变量（broadcast variable），简单可以认为SparkContext是Spark程序的根本。

Driver会把计算任务分成一系列小的task，然后送到executor执行。executor之间可以通信，在每个executor完成自己的task以后，所有的信息会被传回。

![](images/spark01.png)

在Spark里，所有的处理和计算任务都会被组织成一系列Resilient Distributed Dataset（弹性分布式数据集，简称RDD）上的transformation（转换）和actions（动作）。

RDD是一个包含诸多元素、被划分到不同节点上进行并行处理的数据集合，可以将RDD持久化到内存中，这样就可以有效地进在并行操作中复用（在机器学习这种需要反复迭代的任务中非常有效）。在节点发生错误时RDD也可以自动恢复。

RDD就像一个numpy中的array或者Pandas中的Series，可以视作一个有序的item集合。只不过这些item并不存在driver端的内存里，而是被分割成很多partitions，每个partition的数据存在集群的executor的内存中。

RDD是最重要的载体，我们看看如何初始化这么一个对象：

## 初始化RDD方法1

如果你本地内存中已经有一份序列数据（比如python的list），你可以通过`sc.parallelize`去初始化一个RDD

当你执行这个操作以后，list中的元素将被自动分块（partitioned），并且把每一块送到集群上的不同机器上。

```python
import pyspark
from pyspark import SparkContext
from pyspark import SparkConf

conf = SparkConf().setAppName('miniProject').setMaster('local[*]')
sc = SparkContext.getOrCreate(conf)
rdd = sc.parallelize([1,2,3,4,5])
rdd
# ParallelCollectionRDD[0] at parallelize at PythonRDD.scala:475
```

查看数据分块个数

```python
rdd.getNumPartitions()
# 4
```

如果你想看看分区状况怎么办

```python
rdd.glom().collect()
# [[1], [2], [3], [4, 5]]
```

在这个例子中，是一个4个节点的Spark集群。Spark创建了4个executor，然后把数据分成4个块

**tips：使用 `sc.parallelize`，你可以把python list，Numpy array 或者 Pandas Series，Pandas DataFrame 转成Spark RDD。**

## 初始化RDD方法2

第二种方式当时是直接把文本读到RDD了

你的每一行都会被当做item，不过需要注意的一点是，Spark一般默认你的路径指向HDFS的，如果你要从本地读取文件的话，给一个file://开头的全局路径。

```python
# Record current path for future use
import os
cwd = os.getcwd()  # 获取当前路径
rdd = sc.textFile("file://"+cwd+"/names/yob1880.txt")
rdd
# Out[]:file:///home/ds/notebooks/spark/names/yob1880.txt MapPartitionsRDD[3] at textFile at NativeMethodAccessorImpl.java:-2
rdd.first()
# 显示第一个rdd中的第一个元素
```

你甚至可以很粗暴的读入整个文件夹的所有文件。

但是要特别注意，这种读法，RDD中的每个item实际上是一个形如（文件名，文件所有内容）的元组。

```python
rdd = sc.wholeTextFiles('file://' + cwd + 'names')
rdd
# out[]: org.apache.spark.api.java.JavaPairRDD@6b954745
rdd.first()
# out[]:(u'file:/home/ds/notebooks/spark/names/yob1880.txt', u'Mary,F,70...')
```

