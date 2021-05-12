import operator
from numpy import *
import os


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# inX 分类的输入向量，也就是需要预测的样本,比如[1.2,1.3]
# dataet 训练样本集
# labels 标签向量
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 矩阵.shape ，矩阵.shape[0]获取行大小，矩阵.shape[1]获取列大小
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile 把inX向量复制成和dataset行大小的矩阵，1表示一行由1个inX组成
    # 比如inX = [0,0] 那么tile(inX,(dataSetSize,1)) 为
    # array([[0, 0],
    #   [0, 0],
    #  [0, 0],
    #   [0, 0]])
    sqDiffMat = diffMat ** 2  # **代表乘方，会把矩阵中每个元素乘方
    sqDistances = sqDiffMat.sum(axis=1)  # axis=1表示矩阵所有的行中的元素相加
    distances = sqDistances ** 0.5  # 开方
    sortedDisIndicies = distances.argsort()  # 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y，这里获取到的是排序后的索引（index）
    classCount = {}  # 定义一个字典，就是map，key是标签，value是统计这个标签的数量
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 默认值给0
    # Python3.5中：iteritems变为items,1是tunple的index，这里表示对统计数量倒排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # sorted 默认升序 获取对象的第1个域的值
    # iteritems 将字典以迭代器的方式返回，而items方法则返回一个列表
    # sorted 返回新的数组
    return sortedClassCount[0][0]  # 返回最高统计数量的标签


# 读取文件
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))  # 创建numberOfLines 行，3列的矩阵（二维数组），用0填充
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # 矩阵每行填充为listFromLine分割的内容 ,是参数的分隔符
        # 最后一位是标签(分类)
        classLabelVector.append(int(listFromLine[-1]))  # -1表示倒数第一位，:-1表示从0开始，::-1表示从最后一位开始
        index += 1
    return returnMat, classLabelVector


# 归一化(oldValue - min) /(max - min)
# dataSet 是一个矩阵
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 每列最小值得到[col1_min,col2_min...]
    maxVals = dataSet.max(0)  # 每列最大值[col1_max,col2_max...]
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))  # shape 读取矩阵的长宽
    m = dataSet.shape[0]  # 0矩阵的行size 1 则表示获取列的size
    normDataSet = dataSet - tile(minVals, (m, 1))  # 矩阵减法
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 矩阵除法
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10  # 0.1表示90%用来训练，10%也就是0.1用来测试
    datingDataMat, datingLables = file2matrix(
        '/home/patinousward/workspace/python_code/machine-learning/machinelearninginaction/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)  # 得到测试的样本数量
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLables[numTestVecs:m], 3)
        # % 表示占位
        print("the classifier came back with : %d,the real answer is : %d" % (classifierResult, datingLables[i]))
        if (classifierResult != datingLables[i]): errorCount += 1.0
    print("the total error rate is:%f" % (errorCount / float(numTestVecs)))

def img2vector(filename):
    returnVec = zeros((1,1024)) # 1行1024列=32* 32
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0,32*i + j] = int(lineStr[j])
    return returnVec

# 标签数字_xx.txt
def handwritingClassTest():
    hwLabels = []
    # listDir 列出目录下的文件列表
    trainingFileList = os.listdir('/home/patinousward/workspace/digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024)) # 每个文件都创建一个向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0]) # 获取真实结果
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('/home/patinousward/workspace/digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('/home/patinousward/workspace/digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('/home/patinousward/workspace/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with:%d,the real answer is:%d" % (classifierResult,classNumStr))
        if(classifierResult != classNumStr):errorCount +=1.0
    # %s 字符串 %d 数字 %f 浮点数
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount /float(mTest)))

