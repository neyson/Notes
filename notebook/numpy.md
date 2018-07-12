

# numpy

## 数据类型

ndarray内要求数据具有相同的数据类型，改变一个值，所有元素都进行类型转换。

创建方法：

```python
np.array([1,2,3,4])
```

## 矩阵初始化

```python
np.zeros((3,4))
# array([[0., 0., 0., 0.],
#        [0., 0., 0., 0.],
#        [0., 0., 0., 0.]])
np.ones((2,3,4), dtype=np.int32)
# array([[[1, 1, 1, 1],
#         [1, 1, 1, 1],
#         [1, 1, 1, 1]],
#
#        [[1, 1, 1, 1],
#         [1, 1, 1, 1],
#         [1, 1, 1, 1]]])
np.arange(10,30,5) # 5 代表每次叠加5
# array([10, 15, 20, 25])
np.arange(0,2,0.3)
# array([0., 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
np.random.random((2,3))
# 创建形状为（2,3）矩阵，元素取值范围0-1的随机变量
np.random.randn(2,3)
# 创建形状为（2,3）矩阵，元素取值是标准正太分布中的随机变量
np.linspace(0, 2*np.pi, 100)
# 创建array制定区间等分数值，从0到2pi取100个值
np.sin(np.linspace(0, 2*np.pi, 100))
# 创建0-2pi正弦函数取值
```

## 读取txt文件方法

```python
array = np.genfromtxt(
	'file.txt',
	delimiter=',', # 分割符号
  	dtype=str,
  	skip_header=1
)#返回的也是2维的
```

## 切片

```python
matrix = np.array([
  	[5,10,15],
  	[20,25,30],
  	[35,40,45]
])
print(matrix[:,1])  # 打印第2列
# [10 25 40]
print(matrix[:,0:2])  # 打印第1列和第2列
# [[ 5 10]
#  [20 25]
#  [35 40]]
print(matrix[1:3,0:2]) # 打印第2，3行和第1,2列区域内的元素
# [[20 25]
#  [35 40]]
```

## 判断

```python
vector = np.array([5, 10, 15, 20])
vector == 10
# array([False, True, False, False], dtype=bool)
matrix = np.array([
  	[5,10,15],
  	[20,25,30],
  	[35,40,45]
])
matrix == 25
# array([[False,False,False],
#        [False, True,False],
#        [False,False,False]],dtype=bool)
```

## 筛选

```python
vector = numpy.array([5, 10, 15, 20])
equal_to_ten = (vector == 10)
print(equal_to_ten)
# [False, True, False, False]
print(vector[equal_to_ten])
# [10]
```

2D筛选

```python
matrix = np.array([
  	[5,10,15],
  	[20,25,30],
  	[35,40,45]
])
second_column_25 = (matrix[:,1] == 25)
print(second_column_25)
# [False True False]
print(matrix[second_column_25,:])
# [[20 25 30]]
```

与或判断

```python
vector = numpy.array([5, 10, 15, 20])
equal_to_ten_and_five = (vector==10)&(vector==5)
print(equal_to_ten_and_five)
# [False False False False]
equal_to_ten_or_five = (vector==10)|(vector==5)
print(equal_to_ten_or_five)
# [True True False False]
vector[equal_to_ten_or_five] = 50
print(vector)
# [50 50 15 20]
```

## 类型转换

```python
vector = numpy.array(['1', '2', '3'])
print(vector.dtype)
# < U1
print(vector)
vector = vector.astype(float)
print(vector.dtype)
print(vector)
# ['1' '2' '3']
# float64
# [1. 2. 3.]
```

## 求极值

```python
vector = numpy.array([5, 10, 15, 20])
vector.min()
# 5
```

## 求和

```python
matrix = np.array([
  	[5,10,15],
  	[20,25,30],
  	[35,40,45]
])
matrix.sum(axis=1)
# array([30, 75, 120])
matrix.sum(axis=0)
# array([60, 75, 90])
```

## 矩阵变换

```python
print(np.arange(15))
# [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14]
a = np.arange(15).reshape(3,5)
print(a)
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]])
a.shape
# (3, 5)
a.ndim
# 2
a.dtype.name
# 'int32'
a.size
# 15
```

## 矩阵计算

如果两个array的形状相同，则对应位置进行计算；如果形状不同，则可能使用广播的方式。特殊的如果使用 np.dot(A,B)或者A.dot(B)则进行点乘。

## 矩阵变换

```python
np.exp(B)
# 表示求e的B矩阵次幂

np.sqrt(B)
# 表示求矩阵B中每个元素的平方根

np.floor(A)
# 表示对矩阵向下取整

a.ravel()
# 把矩阵a拉平成1维矩阵

a.shape=(6,2)
# 制定矩阵的形状

a.T
# 矩阵a的转置

a.reshape(3, -1)
# 矩阵变形，-1表示自动计算

np.linalg.inv(X)
# 求矩阵X的逆矩阵
```

## 矩阵拼接

```python
a = np.floor(10 * np.random.random((2,2)))
b = np.floor(10 * np.random.random((2,2)))
print(a)
print(b)
print(np.hstack((a,b))) # 按列拼接
# [[3. 1.]
#  [0. 1.]]
#
# [[4. 2.]
#  [9. 2.]]
#
# [[3. 1. 4. 2.]
#  [0. 1. 9. 2.]]
print(np.vstack((a,b))) # 按行拼接
# [[3. 1.]
#  [0. 1.]
#  [4. 2.]
#  [9. 2.]]
```

## 矩阵分解

```python
a = np.floor(10*np.random.random((2,12)))
print(a)
# [[8. 3. 3. 5. 9. 0. 1. 1. 6. 2. 7. 2.]
#  [7. 1. 9. 7. 5. 2. 5. 7. 0. 3. 1. 1.]]
print(np.hsplit(a,3)) # 均分
# [array([[8. 3. 3. 5.]
#         [7. 1. 9. 7.]]),
#  array([[9. 0. 1. 1.]
#         [5. 2. 5. 7.]]),
#  array([[6. 2. 7. 2.]
#         [0. 3. 1. 1.]])]
print(np.hsplit(a,(3,4))) #指定位置切分
# [array([[8. 3. 3.]
#         [7. 1. 9.]]),
#  array([[5.]
#         [7.]]),
#  array([[9. 0. 1. 1. 6. 2. 7. 2.]
#         [5. 2. 5. 7. 0. 3. 1. 1.]])]
print(np.vsplit(a,2)) #按行切分同理
```

## 链接复制

```python 
a = np.arange(12)
b = a
print(b is a)
# True
b.shape = 3,4
print(a.shape)
# (3, 4)
print(id(a))
print(id(b))
# 1974659569504
# 1974659569504
```

## 浅复制

```python
c = a.view() # 浅复制
print(c is a)
# False
c.shape = 2,6
print(a.shape)
# (3, 4)
c[0, 4] = 1234
print(a)
# [[0 1 2 3 ]
#  [1234 5 6 7]
#  [8 9 10 11]]
```

## 复制

```python
d = a.copy()
d is a
d[0,0] = 9999
print(d)
print(a)
# 此时两个值不相同了
```

## 求极值

```python
data = np.sin(np.arange(20)).reshape(5,4)
ind = data.argmax(axis=0)
# 返回每列最大值的索引号
data_max = data[ind,range(data.shape[1])]
# 返回每列最大值
```

##  扩展

```python
a = np.arange(0,40,10)
# [0 10 20 30]
b = np.tile(a (3, 5))
# [[0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30]
#  [0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30]
#  [0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30  0 10 20 30]]
```

## 排序

```python
a = np.array([[4, 3, 5],
			  [1, 2, 1]])
b = np.sort(a,axis=1)
#[[4 3 5]
# [1 1 2]]
a.sort(axis=1)
#[[4 3 5]
# [1 1 2]]
a = np.array([4, 3, 1, 2])
j = np.argsort(a)
# [2 3 1 0]
a[j]
# [1 2 3 4]
```

## 插入数据

```python
np.insert(x, 0, 1, axis=1) # 插入数据集x，插入索引位置是0，插入1，插入整列
```

