"""KNN 手写数字识别"""
# 将测试数据转换成只有一列的0-1矩阵形式
# 将所有（L个）训练数据也都用上方法转换成只有一列的0-1矩阵形式
# 把L个单列数据存入新矩阵A中——矩阵A每一列存储一个字的所有信息
# 用测试数据与矩阵A中的每一列求距离，求得的L个距离存入距离数组中
# 从距离数组中取出最小的K个距离所对应的训练集的索引
# 拥有最多索引的值就是预测值

import os  # 导入os内置库来读取文件名
import operator  # 导入time来测试效率
import numpy as np  # 导入科学计算库numpy         安装方法pip install numpy

trainingDigits = r'.\digits\trainingDigits'
testDigits = r'.\digits\testDigits'
# ↑数据路径
training = (os.listdir(trainingDigits))  # 读取训练集
test = (os.listdir(testDigits))  # 读取测试集


# print(len(training))  #1934个训练集
# print(len(test))  #945个测试集

def read_file(doc_name):  # 定义一个把32x32格式转为1行的函数
    data = np.zeros((1, 1024))  # 创建1个zero数组
    f = open(doc_name)  # 打开文件
    for i in range(32):  # 已知每个文件中有32行32列
        line = f.readline()  # 取行
        for j in range(32):  # 取每行中的每一列
            data[0, 32 * i + j] = int(line[j])  # 给data值
    # print(pd.DataFrame(data))   # 不要在这里转换成DataFrame
    return data  # 否则测试集效率会降低7倍
    # 读取训练集效率会降低12倍


def dict_list(dic: dict):  # 定义函数将字典转化为列表
    keys = dic.keys()  # dic.keys()就是字典的k
    values = dic.values()  # dic.values()就是字典的V
    lst = [(key, val) for key, val in zip(keys, values)]  # for k,v in zip(k,v)
    return lst  # zip是一个可迭代对象
    # 返回一个列表


def similarity(tests, trainings, labels, k):  # tests:测试集 # trainings:训练样本集 # labels:标签 # k: 邻近的个数
    data_line = trainings.shape[0]  # 获取训练集的行数data_line
    zu = np.tile(tests, (data_line, 1)) - trainings  # 用tile把测试集tests重构成一个 data_line行、1列的1维数组
    q = np.sqrt((zu ** 2).sum(axis=1)).argsort()  # 计算完距离后从低到高排序,arg_sort返回的是索引
    my_dict = {}  # 设置一个dict
    for i in range(k):  # 根据我们的k来统计出现频率，样本类别
        vote_label = labels[q[i]]  # q[i]是索引值,通过labels来获取对应标签
        my_dict[vote_label] = my_dict.get(vote_label, 0) + 1  # 统计每个标签的次数
    sort_class_count = sorted(dict_list(my_dict), key=operator.itemgetter(1), reverse=True)
    # 获取vote_label键对应的值，无返回默认
    return sort_class_count[0][0]  # 返回出现频次最高的类别


def distinguish():  # 定义一个识别手写数字的函数
    label_list = []  # 将训练集存储到一个矩阵并存储他的标签
    train_length = len(training)  # 直接一次获取训练集长度
    train_zero = np.zeros((train_length, 1024))  # 创建(训练集长度，1024)维度的zeros数组
    for i in range(train_length):  # 通过遍历训练集长度
        doc_name = training[i]  # 获取所有的文件名
        file_label = int(doc_name[0])  # 取文件名第一位文件的标签
        label_list.append(file_label)  # 将标签添加至label_list中
        train_zero[i, :] = read_file(r'%s\%s' % (trainingDigits, doc_name))  # 转成1024的数组
    # 下面是测试集
    errorNum = 0  # 记录error的初值
    testNum = len(test)  # 同上 获取测试集的长度
    errFile = []  # 定义一个空列表
    for i in range(testNum):  # 将每一个测试样本放入训练集中使用KNN进行测试
        test_doc_name = test[i]  # 通过i当作下标来获取测试集里面的文件
        test_label = int(test_doc_name[0])  # 拿到测试文件的名字 拿到我们的数字标签
        test_data_or = read_file(r'%s\%s' % (testDigits, test_doc_name))  # 调用read_file操作测试集
        result = similarity(test_data_or, train_zero, label_list, 3)  # 调用similarity返回了result
        print("正在测试 %d, 内容是 %d" % (test_label, result))  # 输出result和标签
        if result != test_label:  # 判断标签是否等于测试名
            errorNum += 1  # 不是则+1 记录次数
            errFile.append(test_doc_name)  # 并把错误的文件名加入错误列表
    print("错误数量有 :%d" % errorNum)  # 输出错误的数量
    print("错误的有 :%s" % [i for i in errFile])  # 输出错误的列表中的名字
    print("准确率 %.2f%%" % ((1 - (errorNum / float(testNum))) * 100))  # 计算准确率
