# Spark

## Spark 核心概念和操作

spark可以分为1个driver（笔记本电脑或者集群网关机器上）和若干executor在（各个节点上），通过SparkContext（简称sc）连接Spark集群、创建RDD、累加器(accumlaor)、广播变量（broadcast variable），简单可以认为SparkContext是Spark程序的根本。

Driver会把计算任务分成一系列小的task，然后送到executor执行。executor之间可以通信，在每个executor完成自己的task以后，所有的信息会被传回。

![](images/spark01.png)

在Spark里，所有的处理和计算任务都会被组织成一系列Resilient Distributed Dataset（弹性分布式数据集，简称RDD）上的transformation（转换）和actions（动作）。

RDD是一个包含诸多元素、被划分到不同节点上进行并行处理的数据集合，可以将RDD持久化到内存中，这样就可以有效地进在并行操作中复用（在机器学习这种需要反复迭代的任务中非常有效）。在节点发生错误时RDD也可以自动恢复。

RDD就像一个numpy中的array或者Pandas中的Series，可以视作一个有序的item集合。只不过这些item并不存在driver端的内存里，而是被分割成很多partitions，每个partition的数据存在集群的executor的内存中。

RDD是最重要的载体，我们看看如何初始化这么一个对象：

### 初始化RDD方法1

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

### 初始化RDD方法2

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

但是要特别注意，这种读法，**RDD中的每个item实际上是一个形如（文件名，文件所有内容）的元组。**

```python
rdd = sc.wholeTextFiles('file://' + cwd + 'names')
rdd
# out[]: org.apache.spark.api.java.JavaPairRDD@6b954745
rdd.first()
# out[]:(u'file:/home/ds/notebooks/spark/names/yob1880.txt', u'Mary,F,70...')
```

### 其余初始化RDD的方法

RDD还可以通过其他的方式初始化，包括

+ HDFS上的文件
+ Hive中的数据库与表
+ Spark SQL得到的结果

### RDD transformation 的那些事

大家还对python的list comprehension有印象吗，RDD可以进行一系列的变换得到新的RDD，有点类型那个过程，我们先给大家提一下RDD上最常用的transformation：

* `map() ` 对RDD的每一个item都执行同一个操作
* `flatMap()` 对RDD中的item执行统一操作以后得到一个list，然后以平铺的方式把这些list里所有的结果组成新的list
* `filter() `筛选出来满足条件的item
* `distinct()` 对RDD中的item去重
* `sample()` 从RDD中的item中采样一部分出来，有放回或者无放回
* `sortBy()` 对RDD中的item进行排序

**如果你想到操作后的结果，可以用一个叫做`collect()`的action把所有的item转换成一个Python list。**

简单的例子如下：

```python
numbersRDD = sc.parallelize(range(1,10+1))
print(numbersRDD.collect())
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
squaresRDD = numbersRDD.map(lambda x:x**2)
# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
filteredRDD = numbersRDD.filter(lambda x:x%2==0)
# [2, 4, 6, 8, 10]
```

然后咱们看看`flatMap()`的平展功能：

```python
sentencesRDD = sc.parallelize(['Hello word', 'My name is Patrick'])
wordsRDD = sentencesRDD.flatMap(lambda sentence: sentence.split(' '))
print(wordsRDD.collect())
# ['Hello', 'world', 'My', 'name', 'is', 'Patrick']
print(wordsRDD.count())
# 6
```

为了做一个小小的对应，咱们看看python里对应的操作大概是什么样的：

```python
l=['Hello World', 'My name is Patrick']
ll = []
for sentence in l:
    ll = ll + sentence.split(' ')
print(ll)
# ['Hello', 'world', 'My', 'name', 'is', 'Patrick']
```

比较酷炫的是，前面提到的Transformation， 可以一个接一个的串联，比如：

```python
def doubleIfOdd(x):
    if x % 2 ==1:
        return 2 * x
    else:
        return x
resultRDD = (numberRDD  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            .map(doubleIfOdd)  # [2, 2, 6, 4, 10, 6, 14, 8, 18, 10]
            .filter(lambda x : x > 6) # [10, 14, 8, 18 ,10]
            .distinct())
resultRDD.collect()
# [8, 10, 18, 14]
```

### RDD间的操作

如果你手头上有两个RDD了，下面的这些操作能够帮你对他们以各种方式组合得到1个RDD：

* `rdd1.union(rdd2)` ：所有rdd1和rdd2中的item组合
* `rdd1.intersection(rdd2)`：rdd1和rdd2的交集
* `rdd1.substract(rdd2)`：所有在rdd1中但不在rdd2中的item（差集）
* `rdd1.cartesian(rdd2)`：rdd1和rdd2中所有的元素笛卡尔乘机

简单的例子如下：

```python
numbersRDD = sc.parallelize([1,2,3])
moreNumberRDD = sc.parallelize([2,3,4])
print(numbersRDD.union(moreNumberRDD).collect()) # 并集
# [1, 2, 3, 2, 3, 4]
numbersRDD.intersection(moreNumberRDD).collect() # 交集
# [2, 3]
numbersRDD.subtract(moreNumberRDD).collect() # 差集
# [1]
numbersRDD.cartesian(moreNumbersRDD).collect() # 笛卡尔积
# [(1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)]
```

特别注意：Spark的一个核心概念是**惰性计算**。当你把一个RDD转换成另一个的时候，这个转换不会立刻生效执行！！！Spark慧把它先记在心里，等到真的需要拿到转换结果的时候，才会重新组织你的transformations（因为可能有一连串的变换）这样可以避免不必要的中间结果存储和通信。

刚才提到了**惰性计算**，那么什么东西能让它真的执行转换和计算呢？是的，就是我们马上提到的Actions， 下面是常见action，当他们出现的时候，表明我们需要执行刚才定义的transformations了：

+ collect() ：计算所有的items并返回所有的结果到driver端，接着collect()会以python list的形式返回结果
+ first()：和上面的类似，不过只返回第一个item
+ tack(n)：类似，但是返回n个item
+ count()：计算RDD中item的个数
+ top(n)：返回头n个items，按照自然结果排序
+ reduce()：对RDD中的item做聚合

我们之前已经看到collect(),first()和count()的例子了， 俺们看看reduce()如何使用。比如Spark里从1到10你可以这么做。

```python
rdd = sc.parallelize(range(1,10+1))
rdd.reduce(lambda x,y : x+y)
# 55
```

如果你想理解一下reduce的细节的话，其实可能会先在每个分区（partition）里完成reduce操作，然后全局的进行reduce。这个过程你可以从如下代码大致理解

```python
def f(x,y):
    return x+y:
l = [1,2,3,4]
f(f(f(l[0],l[1]),l[2]),l[3])  # 10
```

有一个很有用的操作，我们试想一下，有时候我们需要重复用到某个transformation序列得到的RDD结果。但是一遍遍重复计算显然是要开销的，所以我们可以通过一个叫做`cache()`的操作把它暂时存储在内存中：

```python
# Caluculate the average of all the squares from 1 to 10
import numpy as np
numbersRDD = sc.parallelize(np.linspace(1.0, 10.0, 10))
squaresRDD = numbersRDD.map(lambda x : x**2)

squaresRDD.cache()  # Preserve the actual items of this RDD in memory
avg = squaresRDD.reduce(lambda x,y:x+y)/squaresRDD.count()

# 38.5
```

缓存RDD结果对于重复迭代的操作非常有用，比如很多机器学习的算法，训练过程需要重复迭代。

#### 练习作业
**我们知道牛顿法求$\sqrt{n}$(达到eps准确度)的算法是这样的：**
* **给定一个初始值 $x = 1.0$.**
* **求$x$和$n / x$的平均$(x + n/x)/2$**
* **根据$(x + n/x)/2$和$\sqrt{n}$的大小比较，确定下一步迭代的2个端点，同时不断迭代直至$x*x$与$n$之间的差值小于$eps$.**

**在Spark中完成上述算法**  
```python
rdd = sc.parallelize([2])
x = 1.0
while abs(x**2-2)>1e-9:
    x = rdd.map(lambda n: (n, (x+n/x)/2)).first()[1]
print(x)
# 1.4142135623746899
```

### 针对更复杂结构的transformation和action

咱们刚才已经见识了Spark中最常见的transformation和action，但是有时候我们会遇到更复杂的结构，比如非常经典的是以元组形式组成的k-v对（key，value），我们把它叫做pair RDDs，而Spark中针对这种item结构的数据，定义了一些transformation和action：

* `reduceByKey()`：对所有有着相同key的items执行reduce操作
* `groupByKey()`：返回类似`（key，listOfValues）`元组的RDD，后面的value list是同一个key下面的
* `sortByKey()`：按照key排序
* `countByKey()`：按照key去对item个数进行统计
* `collectAsMap()`：和collect有些类似，但是返回的是k-v字典

```python
rdd = sc.parallelize(["Hello hello", "Hello New York", "York says hello"])
resultRDD = (
    rdd
    .flatMap(lambda sentence:sentence.split(' '))  # split into words
    .map(lambda word:word.lower())  # lowercase
    .map(lambda word:(word, 1))   # count each appearance
    .reduceByKey(lambda x,y:x+y)  # add counts for each word
)
resultRDD.collect()
# [('says', 1), ('new', 1), ('hello', 4), ('york', 2)]
```

我们可以将结果以k-v字典的形式返回

```python
result = resultRDD.collectAsMap()
print(result)
# {'hello': 4, 'new': 1, 'says': 1, 'york': 2}

resultRDD.sortBy(keyfunc=lambda (word, count):count, ascending=False).take(2)
# 统计出现频次最高的2个词
# out[]: [('hello', 4), ('york', 2)]
```

还有一个很有意思的操作时，在给定2个RDD后，我们可以通过一个类似SQL的方式去join他们。

```python
# Home of different people
homesRDD = sc.parallelize([
        ('Brussels', 'John'),
        ('Brussels', 'Jack'),
        ('Leuven', 'Jane'),
        ('Antwerp', 'Jill'),
    ])

# Quality of life index for various cities
lifeQualityRDD = sc.parallelize([
        ('Brussels', 10),
        ('Antwerp', 7),
        ('RestOfFlanders', 5),
    ])
homesRDD.join(lifeQualityRDD).collect()
# [('Antwerp', ('Jill', 7)),
#  ('Brussels', ('John', 10)),
#  ('Brussels', ('Jack', 10))]
```

## Spark SQL&DataFrame

Spark SQL 是Spark处理数据结构化数据的一个模块，与基础的SparkRDD API不同，SparkSQL提供查询结构化数据及计算结果等信息的结构。在内部，SparkSQL使用这个额外的信息去执行额外的优化，有几种方式进行交互，包括SQL和Dataset API。当使用相同执行引擎进行计算时，无论使用哪种API 语言都可以快速的计算。

**SQL**

Spark SQL的功能之一就是执行SQL查询，Spark SQL也能够被用于从已存在的Hive环境中读取数据。当以另外的编程语言运行SQL时，查询结构将以Dataset/DataFrame的形式返回，也可以使用命令行或者通过JDBC/ODBC与SQL接口交互。

**DataFrames**

从RDD里可以生成类似大家在pandas中的DataFrame，同时可以方便地在上面完成各种操作。

### 构建SparkSession

Spark SQL中所有功能的入口点时SparkSession类。要创建一个SparkSession，仅使用SparkSession.buider()就可以了：

```python
from pyspark.sql import SparkSession
spark = (
    SparkSession
    .builder
    .appName('python spark sql')
    .config('spark.some.config.option','some-value')
    .getOrCreate()
)
```

### 创建DataFrames

在一个SparkSession中，应用程序可以从一个已经存在的RDD或者hive表，或者从Spark数据源中创建一个DataFrame。

举个例子，下面就是基于一个json文件创建一个DataFrame：

```python
df = spark.read.json('data/people.json')
df.show()
# +----+-------+
# | age|   name|
# +----+-------+
# |null|Michael|
# |  30|   Andy|
# |  19| Justin|
# +----+-------+
```

### DataFrame操作

DataFrames 提供一个特定的语法用在Scala，java，Python and R中机构化数据的操作。

在Python中，可以通过`df.age`或者`df['age']`来获取DataFrame的列。虽然前者便于交互式操作，但是还是建议使用后者，这样不会破坏列名，也能引用DataFrame的类

#### select 操作

```python
df.printSchema()  # 相当于df.info()
df.select('name').show()  # 选择单列
# +-------+
# |   name|
# +-------+
# |Michael|
# |   Andy|
# | Justin|
# +-------+
df.select(['name','age']).show()  # 选择多列
# +-------+----+
# |   name| age|
# +-------+----+
# |Michael|null|
# |   Andy|  30|
# | Justin|  19|
# +-------+----+
df.select(df['name'], df['age']+1).show()
# +-------+---------+
# |   name|(age + 1)|
# +-------+---------+
# |Michael|     null|
# |   Andy|       31|
# | Justin|       20|
# +-------+---------+
```

#### filter操作

```python
df.filter(df['age']>21).show()
#+---+----+
#|age|name|
#+---+----+
#| 30|Andy|
#+---+----+
df.groupBy('age').count().show()
#+----+-----+
#| age|count|
#+----+-----+
#|  19|    1|
#|null|    1|
#|  30|    1|
#+----+-----+
```

### Spark SQL

SparkSession的sql函数可以让应用程序以编程的方式运行SQL查询，并将结果作为一个DataFrame返回。

```python
df.createOrReplaceTempView('people')
sqlDF = spark.sql('SELECT * FROM people')
sqlDF.show()
#+----+-------+
#| age|   name|
#+----+-------+
#|null|Michael|
#|  30|   Andy|
#|  19| Justin|
#+----+-------+
```

### Spark DataFrame于RDD交互

Spark SQL支持两种不同的方法用于转换以存在的RDD成为Dataset

第一种方式是使用反射去推断一个包含指定的对象类型的RDD的Schema。在你的Spark应用程序中，当你一直Schema时这个机遇方法的反射可以让你的代码更简洁。

第二种用于创建Dataset的方法是通过一个允许你构造一个Schema然后把它应用到一个已存在的RDD的编程接口。然而这种方法更繁琐，当列和他们的类型知道运行时都是未知时它允许你去构造Dataset

**反射推断**

```python
from pyspark.sql import Row
sc = spark.sparkContext
lines = sc.textFile('data/people.txt')
parts = lines.map(lambda l:l.split(',')) # rdd
people = parts.map(lambda p:Row(name=p[0], age=int(p[1])))

# Infer the schema, and register the DataFrame as a table
schemaPeople = spark.createDataFrame(people) # df
schemaPeople.createOrReplaceTempView('people')  # schema

teenagers = spark.sql("SELECT name FROM people WHERE age >= 13 AND age <= 19")
type(teenagers)  # pyspark.sql.dataframe.DataFrame
type(teenagers.rdd)  # pyspark.rdd.RDD
teenagers.rdd.first()  # Row(name='Justin')

teenNames = teenagers.rdd.map(lambda p: "Name: " + p.name).collect()
for name in teenNames:
    print(name)
# Name: Justin
```

以编程的方式指定Schema