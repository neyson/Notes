# seaborn

在matplotlib基础上封装的画图库，共有5中主题风格

## 引用方法

```python
import seaborn as sns
```

## 测试数据集导入

```python
sns.load_dataset('tips')
sns.load_dataset('iris')
sns.load_dataset('titanic')
sns.load_dataset('flights')
```



## 参数配置

```python
sns.set() # 设置sns的默认参数  包括绘图面积，线条，颜色
sns.set_context('paper') # 设置整体大小和线的大小
sns.set_context('talk') # 也可设置为poster
sns.set_context('notebook', font_scale=1.5, rc={'lines.linewidth': 2.5}) # font字体大小
```

## 风格设置

5中风格包括：

- darkgrid
- whitegrid
- dark
- white
- ticks

```python
sns.set_style('whitegrid') # 设置主题风格
```

## 风格细节设置

```python
sns.despine() # 去掉上边缘线和右边缘线
sns.despine(offset=10) # 设置图于轴线间的间隔
sns.despine(left=True) # 隐藏左边缘线

# 设置不同子图不同风格
with sns.axes_style('darkgrid'): # 第一个子图是darkgrid风格
  plt.subplot(211)
  plt.plot(x1,y1)
plt.subplot(212) # 第二个子图使用默认风格
plt.plot(x2,y2)
```

## 调色板

- color_palette()能传入任何Matplotlib所支持的颜色
- color_palette()不写参数则默认颜色
- set_palette()设置所有图的颜色
- 6个默认的颜色循环主题：deep, muted, pastel, bright, dark, colorblind

```python
current_palette = sns.color_palette() # 取颜色
sns.palplot(current_palette) # 画调色板中所有颜色
sns.color_palette('his', 8) # 使用his颜色空间 传出来8种颜色
sns.color_palette('Paired', 8) # 使用Paired颜色空间 深浅颜色
sns.boxplot(data=data, palette=sns.color_palette('his', 8)) # 在画盒图时引用调色板颜色

sns.his_palette(8, l=.3, s=.8) # l-亮度设置，s-饱和度设置

# 使用xkcd颜色命名调用颜色
sns.xkcd_rgb['pale red']
sns.xkcd_rgb['medium greed']
sns.xkcd_rgb['denim blue']
colors = ['windows blue', 'amber', 'greyish', 'faded green', 'dusty purple']

# 连续性画板
sns.color_palette('Blues') # 默认由浅到深
sns.color_palette('BuGn_r') # 加上_r颜色由深到浅

# 线性调色板
sns.color_palette('cubehelix', 8)
sns.cubehelix_palette(8, start=.5, rot=-.75) #线性变化的8中颜色
sns.light_palette('greed') # 根据亮度线性变换颜色 由浅到深
sns.dark_palette('purple') # 根据暗度变换不同的紫色 由深到浅
sns.light_palette('navy', reverse=True) # reverse变成由深到浅

# cmap调用
pal = sns.dark_palette('green', as_map=True)
sns.kdeplot(x, y, cmap=pal)
```

## 直方图（柱形图 分析单变量分布）

```python
sns.distplot(x, kde=False) # 查看数据的分布
sns.distplot(x, bins=20, kde=False) # 分成20等分
sns.distplot(x, kde=False, fit=stats.gamma) # 显示stats.gamma指标的曲线
```

## 散点图（分析两个特征直接关系）

同时画出两个特征本身的分布和相互之间的散点图，并计算相关系数

```python
sns.jointplot(x='x', y='y', data=df) # x,y分别是df中的column键
sns.jointplot(x='x', y='y', kind='hex', color='k') # 数据量大时，生成6变形灰度图
```

## 特征关联图

```python
sns.pairplot(df) # 画出df中所有col之间的散点图，和自己的直方图

g = sns.PairGrid(df) # 显示所有变量间关联
g.map(plt.scatter)

g = sns.PairGrid(df, hue='a') # 设置区分项 
g.map_diag(plt.hist) # 设置主对角线画图方式
g.map_offdiag(plt.scatter) # 设置非对角线画图方式

g = sns.PairGrid(df, vars=['a','b']) # 指定特征

g = sns.PairGrid(df, palette='GnBu_d') # 指定渐变色
```

## 回归分析图

```python
sns.regplot(x='total_bill', y='tip', data=df) # 制定数据集和键，画回归图
sns.lmplot(x='total_bill', y='tip', data=df) # 同上

sns.regplot(x='x', y='y', data=df, x_jitter=.05) # 制定x指标抖动
```

## 多变量分析绘图

```python
sns.stripplot(x='x', y='y', data=df) # 散点图，不推荐使用，数据量大时数据分布不直观
sns.stripplot(x='x', y='y', data=df, jitter=True) # 不推荐使用，数据量大时不直观，设置数据抖动

sns.swarmplot(x='x', y='y', data=df) # 能够直观体现数据分布
sns.swarmplot(x='x', y='y', hue='z', data=df) # 根据新的指标区分散点颜色
```

## 盒图

可显示离群点

```python
sns.boxplot(x='x', y='y', hue='z', data=df) # 离群点显示为菱形
sns.boxplot(data=df, orient='h') # orient='h'制定横着画盒图

sns.violinplot(x='x', y='y', hue='z', data=df) # 小梯形图，可反应数据分布
sns.violinplot(x='x', y='y', hue='z', data=df, split=True) # split，根据z指标区分颜色

# 两种图重叠显示
sns.violinplot(x='x', y='y', data=df, inner=None)
sns.swarmplot(x='x', y='y', data=df, color='w', alpha=0.5)
```

## 条形图

```python
sns.barplot(x='x', y='y', hue='z', data=df)
```

## 点图

```python
sns.pointplot(x='x', y='y', hue='z', data=df) # 点图可能更好的描述变化差异
sns.pointplot(x='x', y='y', hue='z', data=df,
             palette={'z1':'g', 'z2':'m'}, # 设置不同z的颜色
             markers=['^','o'], # 设置点的形状
             linestyles=['-','--']) # 设置连线方式
```

## 多层面板分类图

Parameters:

- x,y,hue 数据集变量，变量名
- data 数据集名
- row, col 更多分类变量进行平铺显示，传入变量名
- col_wrap 每行的最高平铺数
- estimator 在每个分类中进行矢量到标量的映射 矢量
- ci 置信区间 浮点数或None
- n_boot 计算置信区间时使用的引导迭代次数 整数
- units 采样单元的标识符， 用于执行多级引导和重复测量设计 数据变量或向量数据
- order, hue_order 对应排序列表 字符串列表
- kind: 可选； point默认；bar柱形图；count频次；box盒图；violin提琴图；strip散点图；swarm分散图
- size每个面的高度
- aspect 纵横比
- orient 方向 v / h
- color 颜色
- legend hue 信息面板 True/False
- legend_out 是否扩展图形，并将信息框绘制在中心右边 True/False
- sharex sharey 共享轴线

```python
sns.factorplot(x='x', y='y', hue='h', data=df) # 默认点图
sns.factorplot(x='x', y='y', hue='h', data=df， kind='bar') # 条形图
sns.factorplot(x='x', y='y', hue='h', col='time', data=df，kind='swarm') # 指定4个指标的swarm图

sns.factorplot(x='x', y='y', hue='h', col='day', data=df, kind='box', size=4, aspect=.5)
# 盒图，size制定大小， aspect指定长宽比
```

## 展示多个子集

```python
g = sns.FacetGrid(df,col='a') # 实例画板
g.map(plt.hist, 'x') # 画条形图

g = sns.FacetGrid(df,col='a', hue='z') # 实例画板,指定列，指定区分类
g.map(plt.scatter, 'x','y', alpha=.7) # 画条形图
g.add_legend() # 显示图例

g = sns.FacetGrid(df,row='a', col='b', margin_titles=True) # 制定行列项，并设置图形间间隔
g.map(sns.regplot, 'x', 'y', color=0.3, fit_reg=False, x_jitter=.1) # color=0.3设置深浅，fit_reg设置是否显示回归线

g = sns.FacetGrid(df, col='a', size=4, aspect=0.5) #设置图形大小和长宽比
g.map(plt.scatter,'x','y',s=50, # s设置散点大小
      alpha=0.7, linewidth=0.5, edgecolor='white')

g = sns.FacetGrid(df, palette=pal) # 制定调色板

g = sns.FacetGrid(df,hue_kws={'marker':['^','v']}) # 制定maker

g.set_axis_labels('xx','yy') # 制定轴标签
g.set(xticks=[10,30,50], yticks=[2,6,10]) # 制定轴刻度
g.fig.subplots_adjust(wspace=0.02, hspace=0.02) # 设置子图间距
```

## 热度图

```python 
heatmap = sns.heatmap(2darray) # 热度图
ax = sns.heatmap(2darray, vmin=0.2, vmax=0.5) # 设置热度颜色对应的最大值和最小值

ax = sns.heatmap(df, center=0) # 设置中心值
ax = sns.heatmap(flight, annot=True, fmt='d') # 显示热度当前值，fmt='d'显示字体格式
ax = sns.heatmap(flight, linewidths=0.5) # 设置格子间距
ax = sns.heatmap(flight, cmap='YlBnBu') # 设置调色板
ax = sns.heatmap(flight, cbar=False) # 不显示调色板
```

