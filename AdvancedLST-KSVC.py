# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_TE_set(data_path,label_path):
    X = pd.read_csv(data_path,header=None)
    Y = pd.read_csv(label_path,header=None)
    return np.array(X).T,np.array(Y).squeeze()

def split_set(X,Y,ratio = 0.8):
    cls1 = np.argwhere(Y==0).squeeze()
    cls2 = np.argwhere(Y==1).squeeze()
    cls3 = np.argwhere(Y==2).squeeze()

    idx1 = int(len(cls1)*ratio)
    A_train = X[cls1[:idx1]]
    A_test = X[cls1[idx1:len(cls1)]]

    idx2 = int(len(cls2)*ratio)
    B_train = X[cls2[:idx2]]
    B_test = X[cls2[idx2:len(cls2)]]

    idx3 = int(len(cls3)*ratio)
    C_train = X[cls3[:idx3]]
    C_test = X[cls3[idx3:len(cls3)]]
    return A_train,A_test,B_train,B_test,C_train,C_test

def train_model(A,B,C):
    l1,m = A.shape
    l2,_ = B.shape
    l3,_ = C.shape

    e1 = np.ones((l1,1))
    e2 = np.ones((l2,1))
    e3 = np.ones((l3,1))

    E = np.hstack((A,e1))
    F = np.hstack((B,e2))
    G = np.hstack((C,e3))

    c5 = c3 = c1 = pow(2,5)
    c6 = c4 = c2 = pow(2,-5)

    w1b1 = np.dot(np.linalg.inv(c1*np.dot(F.T,F)+np.dot(E.T,E)+c2*np.dot(G.T,G)),\
                  (c1*np.dot(F.T,e2)+c2*np.dot(G.T,e3)))

    w2b2 = np.dot(np.linalg.inv(c3*np.dot(E.T,E)+np.dot(F.T,F)+c4*np.dot(G.T,G)),\
                  (c3*np.dot(E.T,e1)+c4*np.dot(G.T,e3)))

    w3b3 = np.dot(np.linalg.inv(c5*np.dot(E.T,E)+np.dot(G.T,G)+c6*np.dot(F.T,F)),\
                  (c5*np.dot(E.T,e1)+c6*np.dot(F.T,e2)))

    return w1b1,w2b2,w3b3

def cal_accuracy(label_test,result1,result2,result3):
    num = result1.shape[0]
    pre_test = np.zeros(num)
    i = 0
    for x1,x2,x3 in zip(result1,result2,result3):
        if x1<x2 and x1<x3:
            pre_test[i] = 1
        elif x2<x1 and x2<x3:
            pre_test[i] = 2
        elif x3<x1 and x3<x2:
            pre_test[i] = 3
        i = i+1
    count = 0
    for x,y in zip(pre_test,label_test):
        if x==y:
            count+=1
    return count/num

def test_model(data_test,label_test,w1b1,w2b2,w3b3):
    w1 = w1b1[:-1]
    b1 = w1b1[-1]
    w2 = w2b2[:-1]
    b2 = w2b2[-1]
    w3 = w3b3[:-1]
    b3 = w3b3[-1]
    e = np.ones((data_test.shape[0],1))

    # 决策函数
    num = data_test.shape[0]
    bound = np.ones(num)*0.8

    result1 = (np.dot(data_test,w1)+b1*e).squeeze()
    result2 = (np.dot(data_test,w2)+b2*e).squeeze()
    result3 = (np.dot(data_test,w3)+b3*e).squeeze()

    # 计算准确率
    acc = cal_accuracy(label_test, result1, result2, result3)
    print("the predict accuracy is :", acc)

    # 绘图显示结果
    x = np.arange(0,num)
    plt.plot(x,result1,color='b')
    plt.plot(x,result2,color='g')
    plt.plot(x,result3,color='k')

    plt.show()

if __name__=='__main__':
    # 读入数据集
    data_path = "C:\\Code_py\\LST-KSVC\\te_data\\tar\\X_tar.csv"
    label_path = "C:\\Code_py\\LST-KSVC\\te_data\\tar\\Y_tar.csv"
    X,Y = load_TE_set(data_path, label_path)

    # 切分数据集为训练集和测试集
    A_train, A_test, B_train, B_test, C_train, C_test = split_set(X, Y)
    print("train dataset shape is:",A_train.shape,B_train.shape,C_train.shape)

    # 计算模型参数
    w1b1, w2b2, w3b3 = train_model(A_train, B_train, C_train)

    # 获得训练集和测试集
    data_test = np.vstack((np.vstack((A_test,B_test)),C_test))
    label_test = np.hstack((np.hstack((np.ones(A_test.shape[0]),\
                    np.ones(B_test.shape[0])*2)),np.ones(C_test.shape[0])*3))

    # data_train = np.vstack((np.vstack((A_train,B_train)),C_train))
    # label_train = np.hstack((np.hstack((np.ones(A_train.shape[0]),\
    #                 np.ones(B_train.shape[0])*2)),np.ones(C_train.shape[0])*3))

    # 测试模型
    # test_model(data_train, label_train, w1b1, w2b2, w3b3)
    test_model(data_test, label_test, w1b1, w2b2, w3b3)