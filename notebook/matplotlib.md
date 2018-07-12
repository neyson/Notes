# matplotlib

## 引入方法

```python
import matplotlib.pyplot as plt
```

## 图形大小

```python
fig = plt.figure(figsize=(8,6)) # 设置整体作画区域的长和高
# fig可用来设置图形属性
```

## 折线

```python
plt.plot(x, y)
plt.show()

# 在一张图上画多条折线
plt.plot(x1, y1, c='red') # 第一条折线红色
plt.plot(x2, y2, c='blue') # 第二条折线蓝色
plt.show()

plt.plot(x1, y1, c='red', label='lable1') # 设置红色折线的图例标签
plt.plot(x2, y2, c='blue', label='label2') # 设置蓝色折线的图例标签
plt.show()

plt.plot(x, y, linewidth=10) # 设置线宽度
```

## 条形图

```python
plt.bar(bar_positions, bar_height, 0.3) # 0.3表示柱型的宽度
plt.barh(bar_positions, bar_height, 0.5) # barh 设置柱形图横着画
```

## 柱形图

```python
plt.hist(ser, bins=20, range=(4,5)) # bins设置x的区间范围，range设置起始区间4到5之间
```



## 散点图

```python
plt.scatter(x,y) # 简单散点图
```

## 盒图

```python
plt.boxplot(ser) # 四分图
```



## 横纵轴设置

```python
plt.xticks(rotation=45) # 指定横轴坐标刻度旋转45读
plt.set_xticklabels(labels, rotation=45) # 设置数列label为x轴刻度
plt.set_ylim(0, 50) # 设置y轴的显示范围是0到50
```

## 坐标标签

```python
plt.xlabel('xname') # 设置x轴标签
plt.ylabel('yname') # 设置y轴标签
plt.title('title name') # 设置整个画面的title
```

## 子图设置

```python
# 在一个区域化多个子图
fig = plt.figure()
ax1 = fig.add_subplot(4,3,x) # 分成4行3列个子图，并在x个子图中作画，x从1开始数
ax2 = fig.add_subplot(4,3,y)
ax1.plot(x1, y1) # 在ax1的子图上作画
ax2.plot(x2, y2) # 在ax2的子图上作画

plt.subplot(211) # 接下的图在211子图位置作画
```

## 图例设置

```python
plt.legend(loc='best') # 设置显示图例，位置best，自动选择图例位置
```

## 色彩相关

```python
cb_dark_blue = (0/255, 107/255, 164/255) # 设置RGB三色参数方法
plt.plot(x,y,c=cb_dark_blue)
```

## 在图中直接加文字

```python
plt.text(2005, 87, 'text') # 在指定位置添加文本
```

## 设置全局的参数

```python
matplotlib.rc('figure', figsize= (14,7)) # 设置绘图大小
matplotlib.rc('font', size=14)  # 设置字体大小为14
matplotlib.rc('axes', grid=False)  # 不显示网格
matplotlib.rc('axes', facecolor='white')  # 设置背景颜色是白色
```

## 不显示顶部和右层的坐标线

```python
fig, ax = plt.subplots()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
```

