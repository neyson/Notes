# pandas

## 数据读取

```python
pd.read_csv('*.csv')
```

读进来数据类型为DataFrame，可当成2维矩阵结构，df.dtype显示各个列的数据类型，其中字符的类型为’object'

```python
df.head()  # 显示前5条数据，也可传参显示其他数目条数
df.tail(4)  # 显示最后4条数据
```

## 取列名

```python
df.columns
df.columns = ['a','b','c','d'] # 设置列名
```

## 数据维度

```python
df.shape
```

## 取数据

按行取

```python
df.loc[0] # 读取第0行数据
df.loc[3:6] # 读取第3~5行数据
df.loc[[2,5,10]] # 读取第2，5，10行数据

two_five_ten = [2, 5, 10]
df.loc[two_five_ten] # 读取第2,5,10行数据
```

按列取

```python
df['name'] # 取列名为“name“的列

columns = ['name1', 'name2']
df[columns] # 取列名为'name1'和'name2'的列数据

col_names = df.columns.tolist() # 取df中所有的列名并转化成list类型
sel_columns = []

for c in col_names:
  if c.endswith('a'):
    sel_columns.append(c)
df_sel = df[sel_columns]  # 取出所有列名中以'a'结尾的列数据
```



## 数据类型

| 类型       | 表示          |
| -------- | ----------- |
| object   | string类型数据  |
| int      | integer类型数据 |
| float    | float类型数据   |
| datatime | time类型数据    |
| bool     | boolean类型数据 |

数据类型查看方法

```python
df.dtype
```

## 数据计算

```python
div_1000 = df['name'] / 1000 # 'name'列中的每个元素都除以1000
new_series = df['name1'] * df['name2'] # 'name1'和'name2'每个元素相乘
df['name'].max() # 取列的最大值
df['name']/df['name'].max() # 列所有元素除最大值
```

## 添加数据

```python
# 添加列要求行数与原df数目一致
df['new'] = series # 直接声明新的列名并赋值
```

## 排序

排序时缺失值会放到最后

```python
df.sort_values('name', inplace=True)  # 针对name列进行排序，从小到大
df.sort_values('name', inplace=True, ascending=False) # 从大到小排序 
# 排序后index值是乱的
df.reset_index(drop=True) # 形成新的重新排序的index值，其他数据不变 drop=True 表示原df丢弃
```

## 缺失值处理

```python
df_null = pd.isnull(df['name'])
null_se = df['name'][df_null]
len(null_se) # 统计缺失值个数1

pd.isnull(df['name']).sum() # 缺失值个数方法2
df.dropna(axis=1) # 去除列
df.dropna(axis=0, subset=['a','b']) # 如果a列或者b列有缺失值，则丢弃该行

# 每列缺失值个数
def not_null_count(col):
  col_null = pd.isnull(col)
  null = col[col_null]
  return len(null)
col_null_count = df.apply(not_null_count)

```

## 求均值

```python
df['name'].mean()
```

## 统计处理

```python
# 根据列a分类，并计算每类中b属性的均值，返回类型为Series
df.pivot_table(index='a',values='b',aggfunc=np.mean）

# 根据列a分类，并计算每类中b,c属性值的求和，返回类型为Series
df.pivot_table(index='a',values=['b','c'],aggfunc=np.sum))

# 根据前两特征，为横轴和纵轴，生成表格数据
df.pivot('a','b','c')

# 显示df中统计的均值方差四个分位数，最大值，最小值，个数
df.describe()
```

## 自定义函数

```python
# 显示每列第100行
def hundredth_row(column):
  hundredth_item = column.loc[99]
  return hundredth_item
hundredth_row = df.apply(hundredth_row)

# 每列缺失值个数
def not_null_count(col):
  col_null = pd.isnull(col)
  null = col[col_null]
  return len(null)
col_null_count = df.apply(not_null_count)
```

## Series结构

dataframe其中一行或一列为series结构

```python
ser.values # ser的值，类型为ndarray
pd.Series(array, index=indarray) # 创建Ser的，并以indarray的值作为索引
ser.reindex(sorted_index) # 按索引序列排序
ser.sort_index() # 以索引进行排序
ser.sort_values() # 以值进行排序
ser[ser>50] # 筛选
ser.value_counsts() # 统计ser中各种类样本的个数
```

## 设置索引

```python
df.set_index('name', drop=False) # 返回以name列作为索引的df
```

## 时间类型转换

```python
pd.to_datatime(ser) # 1999/09/09 字符串转换成 1999-09-09
```

