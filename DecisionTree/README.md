# 决策树算法实践

.
├── decision_tree.py		 决策树算法
├── iris			Iris数据集
│   ├── bezdekIris.data
│   ├── iris.data
│   └── iris.names
├── __pycache__
│   └── decision_tree.cpython-36.pyc
├── README.md
├── test.dot		
└── test.py			算法测试代码

## 决策树算法

决策树算法属于贪心分治算法, 假设所有的特征都是类别类型,自顶向下递归建树, 根据信息增益(IG)选择最佳的特征进行分类.
递归出口:
1. 节点上所有训练样本都属于一个类别
2. 节点上没有训练样本
3. 没有更多的特征可供选择

### 香农熵(entropy)
给定一个随机变量X, 其取值为{P(X=V1)=P1, ..., P(X=Vc)=Pc }, X的熵为:
H(X) = -sum( Pi*log(Pi) ) i=1,2, ..., c. c为当前数据集中的类别数量
**熵越大, 说明系统越混乱**

条件熵(conditional entropy): 给定随机变量A(属性)后(根据属性A分割数据集),(数据集)X剩余的熵:
H(X|A) = sum(P(A=a) * H(X|A=a)) a=1,..,|A| a是属性A的属性值

### 信息增益(Information Gain)
IG 定义为 给定随机变量A(属性)后, 系统X所减少的不确定性
IG(X|A) = H(X) - H(X|A) = H(X) - sum(P(A=aj)*H(X|A=aj)

在决策树算法中, 选择的最佳的分割属性就是通过IG判断.
