import matplotlib.pyplot as plt

decisionNode = dict(boxstyle = "sawtooth",fc = "0.8") # 决策节点,定义字典 sawtooth 锯齿框
leafNode = dict(boxstyle = "round4",fc = "0.8") # round环形框
arriw_args = dict(arrowstyle="<-") # 箭头样式


def plotNode(nodeTxt,centerPt,parentPt,nodeType): # nodeType 就是上面定义的字典
    # plt.annotate(str, xy=data_point_position, xytext=annotate_position, 
    #              va="center",  ha="center", xycoords="axes fraction", 
    #              textcoords="axes fraction", bbox=annotate_box_type, arrowprops=arrow_style)
    # str是给数据点添加注释的内容，支持输入一个字符串
    # xy=是要添加注释的数据点的位置
    # xytext=是注释内容的位置
    # bbox=是注释框的风格和颜色深度，fc越小，注释框的颜色越深，支持输入一个字典
    # va="center",  ha="center"表示注释的坐标以注释框的正中心为准，而不是注释框的左下角(v代表垂直方向，h代表水平方向)
    # xycoords和textcoords可以指定数据点的坐标系和注释内容的坐标系，通常只需指定xycoords即可，textcoords默认和xycoords相同
    # arrowprops可以指定箭头的风格支持，输入一个字典
    # plt.annotate()的详细参数可用__doc__查看，如：print(plt.annotate.__doc__)
    createPlot.ax1.annotate(nodeTxt,xy = parentPt,xycoords = 'axes fraction',
    xytext = centerPt,textcoords = 'axes fraction',va = "center",ha = "center"
    ,bbox = nodeType,arrowprops = arriw_args
    )

def createPlot():
    # 这里1是图像名，如果是数字，显示为figure ${num} ,如果是string类型，则直接显示名字
    # facecolor 图表背景颜色
    fig = plt.figure(1,facecolor = 'white') 
    fig.clf() # 清除figure坐标轴
    createPlot.ax1 = plt.subplot(111,frameon = False)
    # u的意义在于用Unicode 格式 进行编码，防止乱码
    plotNode(U'decisionNode',(0.5,0.1),(0.1,0.5),decisionNode) # 好像默认没支持中文
    plotNode(U'leafNode',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

# 获取树的叶子节点
def getNumLeafs(myTrees):
    numLeafs = 0
    firstStr = list(myTrees.keys())[0] # 决策树一般第一层都是一个节点,加list转化，否则报错
    secondDict = myTrees[firstStr] # 获取第二层开始
    for key in secondDict.keys():
        if(type(secondDict[key])).__name__ == 'dict': # 如果这层是个dic，说明还有下一层，递归...
            numLeafs +=getNumLeafs(secondDict[key])
        else: numLeafs +=1 # 否则说明当前节点就是叶子节点，直接统计+1
    return numLeafs

# 获取树的最大深度
def getTreeDepth(myTrees):
    maxDepth = 0
    firstStr = list(myTrees.keys())[0]
    secondDict = myTrees[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else: thisDepth = 1
        if thisDepth > maxDepth:maxDepth = thisDepth # ？感觉这个恒成立？
    return maxDepth

# 检索树
def retrieveTree(i):
    listOfTrees = [{
        'no surfacint':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}
    },
    {
        'no surfacint':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'},1:'no'}}}}
    }]
    return listOfTrees[i]


