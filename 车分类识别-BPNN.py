#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  读取图像所需
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


import numpy
import scipy.special
import csv
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


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


# In[4]:


# main
# 读图像文件函数
def load_Img(imgDir,imgFoldName):
    imgs = os.listdir(imgDir+imgFoldName)
    imgNum = len(imgs)      #文件夹中图片数量
    img0 = Image.open(imgDir+imgFoldName+"/"+imgs[0])
    arr0 = np.array(img0)
    [a1,a2,a3]=arr0.shape
    data = np.empty((imgNum,a1,a2,a3))  
    for i in range (imgNum):
        img = Image.open(imgDir+imgFoldName+"/"+imgs[i])
        arr = np.array(img)
        data[i,:,:,:] = arr   # 依次读取图片转换成像素图
    return data


# In[5]:


# 读图像
craterDir = "C:/Users/Administrator/Kaggle/选取的训练集/"
foldName = "bicycle"
d_bicycle=load_Img(craterDir,foldName)
craterDir = "C:/Users/Administrator/Kaggle/选取的训练集/"
foldName = "car"
d_car=load_Img(craterDir,foldName)
craterDir = "C:/Users/Administrator/Kaggle/选取的训练集/"
foldName = "motorbike"
d_motorbike=load_Img(craterDir,foldName)


# In[6]:


# 将三维数据矩阵拉成列向量
for i in range(len(d_bicycle)):      
    t1=d_bicycle[i].reshape(375*500*3,)
    t1=t1.reshape(-1,1)
    if i==0:
        da_bicycle=t1
    if i!=0:
        da_bicycle=np.append(da_bicycle,t1,axis=1)


# In[7]:


# 将三维数据矩阵拉成列向量
for i in range(len(d_car)):        
    t1=d_car[i].reshape(375*500*3,)
    t1=t1.reshape(-1,1)
    if i==0:
        da_car=t1
    if i!=0:
        da_car=np.append(da_car,t1,axis=1)


# In[8]:


# 将三维数据矩阵拉成列向量
for i in range(len(d_motorbike)):    
    t1=d_motorbike[i].reshape(375*500*3,)
    t1=t1.reshape(-1,1)
    if i==0:
        da_motorbike=t1
    if i!=0:
        da_motorbike=np.append(da_motorbike,t1,axis=1)


# In[9]:


print(d_bicycle.shape,d_car.shape,d_motorbike.shape)


# In[10]:


da_bicycle.shape


# In[11]:


da_bicycle=da_bicycle.T


# In[12]:


da_car=da_car.T;da_motorbike=da_motorbike.T


# In[13]:


da_bicycle.shape


# In[14]:


# number of input, hidden and output nodes
input_nodes = 562500
hidden_nodes = 800
output_nodes = 3
# learning rate
learning_rate = 0.3
# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)  #BP神经网络框架


# In[15]:


#读入bicycle训练
epochs = 1
for e in range(epochs):
    for record in da_bicycle:
        all_values=record
        inputs = all_values / 255.0 * 0.99 + 0.01   #所有数据归一化
        targets = numpy.zeros(output_nodes) + 0.01
        targets[0] = 0.99
        n.train(inputs, targets)
        pass
    pass


# In[ ]:


#读入car训练
epochs = 1
for e in range(epochs):
    for record in da_car:
        all_values=record
        inputs = all_values / 255.0 * 0.99 + 0.01   #所有数据归一化
        targets = numpy.zeros(output_nodes) + 0.01
        targets[1] = 0.99
        n.train(inputs, targets)
        pass
    pass


# In[ ]:


#读入motorbike训练
epochs = 1
for e in range(epochs):
    for record in da_motorbike:
        all_values=record
        inputs = all_values / 255.0 * 0.99 + 0.01   #所有数据归一化
        targets = numpy.zeros(output_nodes) + 0.01
        targets[2] = 0.99
        n.train(inputs, targets)
        pass
    pass


# In[ ]:


#  load the mnist test data CSV file into a list
#  转载测试集
craterDir = "C:/Users/Administrator/Kaggle/"
foldName = "选取的测试集"
d_test=load_Img(craterDir,foldName)


# In[ ]:


d_test.shape


# In[ ]:


for i in range(len(d_test)):    
    t1=d_test[i].reshape(375*500*3,)
    t1=t1.reshape(-1,1)
    if i==0:
        da_test=t1
    if i!=0:
        da_test=np.append(da_test,t1,axis=1)


# In[ ]:


da_test=da_test.T
da_test.shape


# In[ ]:


# 测试神经网络
# 记录判别状态用数 组
scorecard = []
for record in da_test:
    all_values = record             
    correct_label = que_ding_de_zhi
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


# In[ ]:


# 计算正确率
scorecard_array = numpy.asarray(scorecard)
print ("正确率 = ", scorecard_array.sum() / scorecard_array.size)

