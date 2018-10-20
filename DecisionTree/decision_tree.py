#!/usr/bin/python3
#-*- coding:utf-8 -*-

'''
决策树算法实现
author: GT
time: 2018/10/19 19:13
'''

from collections import defaultdict, namedtuple
from math import log2
import numpy as np
import pickle
import uuid

class DecisionTree(object):
	'''决策树'''

	def __init__(self):
		pass

	def classify(self, X, tree=None, feature_names = None):
		'''
		使用已经训练好的决策树给新的数据分类
		@param X: 数据
		@param tree: 决策树
		@param feature_names: 特征名
		@return 数据所属分类
		'''
		if tree is None:
			tree = self.tree
			pass
		if feature_names is None:
			feature_names = self.feature_names
			pass
		if type(tree) is not dict:
			return tree
			pass

		feature = list(tree.keys())[0]
		value = X[feature_names.index(feature)] # 样本特征值
		child = tree[feature][value]
		return self.classify(X, child, feature_names)
		pass # end of classify function

	def create_tree(self, data, classes, feature_names):
		'''
		创建树, 用dict表示树 {feature_name: { feature_value: tree_node}}
		@param data: 样本数据集合
		@param classes: 样本数据对应分类
		@param feature_names: 样本数据中每一维对应的特征名称
		@return 树节点
		'''
		# 如果只剩一种类型的数据, 将剩下的数据作为叶子节点返回
		if len(set(classes)) == 1:
			return classes[0]
			pass
		# 如果特征集合为空, 则将以最多数据的分类作为叶子节点发挥
		if len(feature_names) == 0:
			return self.__max_class(classes)
			pass
		# 否则, 继续分割特征
		tree = {}
		feat_index = self.__choose_best_feature(data, classes)
		feat_name = feature_names[feat_index]
		new_dataset = self.__split_data(data, classes, feat_index)
		# sub_feature_names = feature_names[:]; sub_feature_names.pop(feat_name)
		sub_feature_names = feature_names[:feat_index]+feature_names[feat_index+1:]

		tree[feat_name] = {} # 以该特征为根
		# 将其他特征值添加到树节点中
		for feat_val, (sub_data, sub_cls) in new_dataset.items():
			tree[feat_name][feat_val] = self.create_tree(sub_data, sub_cls, sub_feature_names)
			pass
		self.tree = tree
		self.feature_names = feature_names
		return tree
		pass # end of create_tree function

	def __choose_best_feature(self, data, classes):
		'''
		选择信息增益最大的特征
		@param data: 样本数据
		@param classes: 样本数据对应类型
		@return 特征索引
		'''
		# 计算原特征集香农熵
		orig_entropy = self.__cal_shang_nong_entropy(classes)
		# 计算根据feat分类后系统的熵
		IG = []
		for index in range(len(data[0])):
			'''
			1. 分别计算在该特征有几种值
			2. 通过不同特征值对数据分成不同堆
			3. 分别计算不同的堆的香农熵
			4. 计算该特征下的香农熵
			'''
			new_dataset = self.__split_data(data, classes, index)
			new_entropy = sum([
				len(sub_cls)/len(classes) * self.__cal_shang_nong_entropy(sub_cls)
				for _, (_, sub_cls) in new_dataset.items()
			]) # 特征值概率乘以香农熵
			IG.append(orig_entropy-new_entropy) # 计算信息增益
			pass
		return IG.index(max(IG))
		pass # end of __choose_best_feature function

	def __split_data(self, data, classes, index):
		'''
		根据第index个特征分割样本数据
		@param data: 样本数据
		@param classes: 样本数据对应类别
		@param index: 特征索引
		@return dict{feat_value: [data, classes]}
		'''
		d = defaultdict(lambda:[[],[]]) # 或者在for...in里面写设置默认值d={}; d.setdefault(item[index], [[],[]])
		for item, cls in zip(data, classes):
			d[item[index]][0].append(np.append(item[:index], item[index+1:])) # 数据,除去index处的特征
			d[item[index]][1].append(cls) # 样本数据的类别不变, 直接添加
			pass
		return d
		pass # end of __split_data function


	def __cal_shang_nong_entropy(self, classes):
		'''
		计算香农熵 S = sum(-P_c*log(P_c))
		@param classes: 类别
		@return 
		'''
		d = defaultdict(lambda:0)
		for cls in classes:
			d[cls] += 1
			pass
		'''
		在github上的大神的代码:
		uniq_vals = set(classes)
		val_nums = {key: values.count(key) for key in uniq_vals}
		'''
		p = [v/len(classes)  for k, v in d.items()] # 计算每种类型的概率
		S = sum([-x*log2(x) for x in p]) # 计算香农熵
		return S
		pass # end of __cal_shang_nong_entropy function

	def __max_class(self, classes):
		'''
		获取最多的类别并返回
		@param classes: 样本数据对应类型
		@return 
		'''
		d = defaultdict(lambda:0) # 使用匿名函数使每个键都有初始值, 效果等同于defaultdict(int)
		for cls in classes: # 统计每一种类型的个数
			d[cls] += 1
			pass
		return max(d, key=d.get) # 使用max函数返回最多的类型 !!! 注意max的用法
		pass

	'''-------------------------------------------------
	以下代码为copy PytLab:
	https://github.com/PytLab/MLBox/blob/master/decision_tree/trees.py
	-------------------------------------------------'''

	def dump_tree(self, filename):
		'''
		存储树结构到文件中
		@param filename: 文件名
		@return
		'''
		with open(filename, 'wb') as f:
			pickle.dump(self.tree, f)
			pass
		pass # end of dump_tree function

	def load_tree(self, filename):
		'''
		从文件中加载树结构
		@param filename: 文件名
		@return 
		'''
		with open(filename, 'rb') as f:
			tree = pickle.load(f)
			self.tree = tree
			pass
		return tree
		pass # end of load_tree function

	def get_nodes_edges(self, tree=None, root_node=None):
		''' 返回树中所有节点和边
		'''
		Node = namedtuple('Node', ['id', 'label'])
		Edge = namedtuple('Edge', ['start', 'end', 'label'])

		if tree is None:
			tree = self.tree

		if type(tree) is not dict:
			return [], []

		nodes, edges = [], []

		if root_node is None:
			label = list(tree.keys())[0]
			root_node = Node._make([uuid.uuid4(), label])
			nodes.append(root_node)

		for edge_label, sub_tree in tree[root_node.label].items():
			node_label = list(sub_tree.keys())[0] if type(sub_tree) is dict else sub_tree
			sub_node = Node._make([uuid.uuid4(), node_label])
			nodes.append(sub_node)

			edge = Edge._make([root_node, sub_node, edge_label])
			edges.append(edge)

			sub_nodes, sub_edges = self.get_nodes_edges(sub_tree, root_node=sub_node)
			nodes.extend(sub_nodes)
			edges.extend(sub_edges)

		return nodes, edges
		pass # end of get_nodes_edges function

	def dotify(self, tree=None):
		''' 获取树的Graphviz Dot文件的内容
		'''
		if tree is None:
			tree = self.tree

		content = 'digraph decision_tree {\n'
		nodes, edges = self.get_nodes_edges(tree)

		for node in nodes:
			content += '    "{}" [label="{}"];\n'.format(node.id, node.label)

		for edge in edges:
			start, label, end = edge.start, edge.label, edge.end
			content += '    "{}" -> "{}" [label="{}"];\n'.format(start.id, end.id, label)
		content += '}'

		with open('test.dot', 'w') as f:
			f.write(content)
			pass

		return content
		pass # end of dotify function

	pass