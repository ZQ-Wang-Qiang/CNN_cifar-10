#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import numpy as np
import os
from __future__ import print_function
from six.moves import cPickle as pickle
from scipy.misc import imread
import platform


# In[2]:


import numpy
import scipy.special
import csv
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import cifar_10_read   # 用来装载cifar-10


# In[4]:


[train_data,train_label,test_data,test_label]=cifar_10_read.load_CIFAR10("C:/Users/Administrator/cifar-10-batches-py/")


# In[5]:


# train_data,test_label,test_data,test_label 矩阵数据
train_data.shape


# In[6]:


# BP神经网络类
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # 权重矩阵，wih和who
        #从节点i至下一层节点j的权重为w_i_j
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        
        # 学习率
        self.lr = learningrate
        
        # sigmoid作为激活函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    # 神经网络的训练
    def train(self, inputs_list, targets_list):
        # 输入行向量特征，处理为列向量
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # 计算输入隐藏层的信号
        hidden_inputs = numpy.dot(self.wih, inputs)
        
        # 计算隐藏层输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 计算输出层输入信号
        final_inputs = numpy.dot(self.who, hidden_outputs)
        
        # 计算最终输出信号
        final_outputs = self.activation_function(final_inputs)
        
        # 输出层误差为 (target - actual)
        output_errors = targets - final_outputs
        
        # 计算隐藏层输出误差
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # 更新隐藏层至输出层的权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # 更新输入层至隐藏层的权重
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass

    # 运行神经网络
    def query(self, inputs_list):
    
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# In[7]:


'''# 将训练三维数据矩阵拉成列向量
for i in range(len(train_data)):      
    t1=train_data[i].reshape(32*32*3,)
    t1=t1.reshape(-1,1)
    if i==0:
        da_train=t1
    if i!=0:
        da_train=np.append(da_train,t1,axis=1)'''


# In[8]:


'''da_train=da_train.T'''


# In[36]:


# number of input, hidden and output nodes
input_nodes = 3072
hidden_nodes = 1000
output_nodes = 10
# learning rate
learning_rate = 0.1
# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)  #BP神经网络框架


# In[10]:


'''#读入训练集
epochs = 1
for e in range(epochs):
    for record in da_train:
        all_values=record
        inputs = all_values / 255.0 * 0.99 + 0.01   #所有数据归一化
        targets = numpy.zeros(output_nodes) + 0.01
        targets[train_label[jishu]] = 0.99
        n.train(inputs, targets)
        jishu=jishu+1
        pass
    pass'''


# In[11]:


da_train = numpy.loadtxt(open("C:/Users/Administrator/作业/featvector.csv","rb"), delimiter=",", skiprows=0)
da_test = numpy.loadtxt(open("C:/Users/Administrator/作业/featvector_test.csv","rb"), delimiter=",", skiprows=0)


# In[37]:


jishu=0
for record in da_train:
        all_values=record
        inputs = all_values / 255.0 * 0.99 + 0.01   #所有数据归一化
        targets = numpy.zeros(output_nodes) + 0.01
        targets[train_label[jishu]] = 0.99
        n.train(inputs, targets)
        jishu=jishu+1
        pass


# In[13]:


# 测试集
'''test_data.shape'''


# In[14]:


# 将测试三维数据矩阵拉成列向量
'''for i in range(len(test_data)):      
    t1=test_data[i].reshape(32*32*3,)
    t1=t1.reshape(-1,1)
    if i==0:
        da_test=t1
    if i!=0:
        da_test=np.append(da_test,t1,axis=1)'''


# In[15]:


# 获得了处理过的测试集
'''da_test=da_test.T;da_test.shape'''


# In[38]:


# 测试神经网络
# 记录判别状态用数组
jishu=0
scorecard = []
for record in da_test:
    all_values = record             
    correct_label = test_label[jishu]
    inputs = all_values / 255.0 * 0.99 + 0.01
    # 询问输出层
    outputs = n.query(inputs)
    # 取最大值为神经网络判断的类别
    label = numpy.argmax(outputs)
    # 判断正确增加1，否则增加0
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
    jishu=jishu+1


# In[39]:


# 计算正确率
scorecard_array = numpy.asarray(scorecard)
print ("正确率 = ", scorecard_array.sum() / scorecard_array.size)


# In[18]:


'''np.savetxt('C:/Users/Administrator/作业/featvector.csv',da_train,delimiter=',')'''
'''my_matrix = numpy.loadtxt(open("C:/Users/Administrator/作业/featvector.csv","rb"), delimiter=",", skiprows=0)'''
'''np.savetxt('C:/Users/Administrator/作业/featvector_test.csv',da_test,delimiter=',')'''
'''my_matrix = numpy.loadtxt(open("C:/Users/Administrator/作业/featvector_test.csv","rb"), delimiter=",", skiprows=0)'''

