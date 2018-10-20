#!/usr/bin/python3
#-*- coding:utf-8 -*-

'''
测试决策树算法的正确性
author: GT
time: 2018/10/20 00:48
'''

'''
总结:
1. 加深了对决策树算法的理解
2. 完成了整个决策过程实现: 创建树->检查递归出口->计算香农熵->分割数据集->计算信息增益->生成子树
3. 学到了[x for x in iteratable]的语法
4. 学到了lambda匿名函数的写法和作用
'''

from decision_tree import DecisionTree
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# 先获取数据
file = './iris/iris.data'
df = pd.read_csv(file, header=None)
df = shuffle(df) # 打乱顺序
df = df.reset_index(drop=True) # 重新设置序号
# 保存当前的训练数据
with open('test.dat', 'w') as f:
	for item in df.values:
		s = ','.join([str(x) for x in item])
		f.write(s)
		f.write('\n')
	pass
# 提取样本数据喝样本分类
data = df.iloc[:100,:4] # 前4列, 取前100项数据
classes = df.iloc[:100, 4] # 第5列
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

dt = DecisionTree()
dt.create_tree(data=data.values, classes=classes.values, feature_names=feature_names)
dt.dump_tree("tree.dt")
content = dt.dotify()
print(content)

X = [5.5,3.5,1.3,0.2]
res = dt.classify(X)
print("direct result: ", res)