from math import log
import operator

# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1] # 数据的最后一列
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  # 默认值为0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries #获取概率，也就是这个特征出现的次数/总样本
        shannonEnt -= prob * log(prob,2) # 以2为底求对数，香农熵的公式
    return shannonEnt


def createDataSet():
    dataSet = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    labels = ['no surfacing','flipppers']
    return dataSet,labels

# axis 表示第几列的纬度index ，value表示这个纬度的值作为分界
def splitDataSet(dataSet,axis,value):
    retDataSet  = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] # 从开头到axis列的数据
            reducedFeatVec.extend(featVec[axis + 1:])
            # 上面两行代码意思是排除了当前行数据中的value值,意义为何要排除,因为要递归做决策树，下次运算不能再出现
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1 # 随便挑一行数据，-1是因为最后一列是结果
    baseEntropy = calcShannonEnt(dataSet) # 计算当前的香农熵
    bestInfoGain = 0.0;bestFeature = -1
    for i in range(numFeatures): # i代表第i个特征
        # 将dataSet中的数据按行依次放入example中，然后取得example中的example[i]元素，放入列表featList中
        # example 是自定义的元素变量 可以拆开看 1. for example in dataSet 2. featList = [example[i]]
        featList = [example[i] for example in dataSet] # 其实就是获取dataset的第i列的数据
        uniqueVals = set(featList) # 去重 因为要判断以哪个value为分界
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            # 这里有点类似权重，subDataSet中，每部分数据占的比重
            prob = len(subDataSet)/float(len(dataSet)) 
            newEntropy += prob * calcShannonEnt(subDataSet) # subDataSet每部分数据加起来
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 多数表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount,key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


# 创建树的函数代码

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] # 取dataSet/subDataset最后一列,就是是否是鱼的
    if classList.count(classList[0]) == len(classList): # classList 列表里面都是同一个字符串，说明分类完全相同
        return classList[0]
    if len(dataSet[0]) == 1 : # dataSet剩下最后一列 example :[[x1],[x2]..]
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat]) # 解除数组中的变量对数据的引用，del语句作用在变量上，而不是数据对象上,数组=删除这个数据
    featValues = [example[bestFeat] for example in dataSet] # 取出当前best特征的列的值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:] # 复制
        # splitDataSet 的返回值就是参数dataSet中少一列（当前最优feature）的数据
        # 子调用的返回结果会附着在父的上面形成树状
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

    # result {'no surfacing': {0: 'no', 1: {'flipppers': {0: 'no', 1: 'yes'}}}}
    # no surfacing 就是是否上浮，flipppers是否有脚蹼
    # 先找出最优的特征，就是是否上浮，然后value有2中，0和1
    # 当为0时，通过分割数据集，发现样本分类一样，直接返回分类，就是'no'
    # 当为1时，通过分割数据集，发现样本分类不一样，而且并没有剩下最后一列（这个案例，剩下最后一列就是分类）
    # 继续选择最优的列，这里就只有是否有脚蹼了
    # 当按是否有脚蹼为0分割子集的时候，分类都是否，所以直接返回‘no’，同理，为1分割的时候，分类都是是，所以直接返回‘yes’
    # 这里隐藏了 if len(dataSet[0]) == 1 的情况，假设分割1的时候，分类不同，这时候就会走这行代码，分类不同，但是已经走完所有特征了，这里直接使用投票的方式解决
    
