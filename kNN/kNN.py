from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

# inX 分类的输入向量，也就是需要预测的样本,比如[1.2,1.3]
# dataet 训练样本集
# labels 标签向量
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] # 矩阵.shape ，矩阵.shape[0]获取行大小，矩阵.shape[1]获取列大小
    diffMat = tile(inX,(dataSetSize,1)) - dataSet # tile 把inX向量复制成和dataset行大小的矩阵，1表示一行由1个inX组成
    # 比如inX = [0,0] 那么tile(inX,(dataSetSize,1)) 为
    #array([[0, 0],
    #   [0, 0],
    #  [0, 0],
    #   [0, 0]])
    sqDiffMat = diffMat**2 # **代表乘方，会把矩阵中每个元素乘方
    sqDistances = sqDiffMat.sum(axis = 1) # axis=1表示矩阵所有的行中的元素相加
    distances = sqDistances**0.5 # 开方
    sortedDisIndicies = distances.argsort() # 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y，这里获取到的是排序后的索引（index）
    classCount = {}  # 定义一个字典，就是map，key是标签，value是统计这个标签的数量
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 # 默认值给0
    # Python3.5中：iteritems变为items,1是tunple的index，这里表示对统计数量倒排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)# sorted 默认升序 获取对象的第1个域的值
    # iteritems 将字典以迭代器的方式返回，而items方法则返回一个列表
    # sorted 返回新的数组
    return sortedClassCount[0][0] # 返回最高统计数量的标签


