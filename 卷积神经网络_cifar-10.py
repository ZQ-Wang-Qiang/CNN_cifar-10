#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.compat.v1 as tf     # 需要用tensorflow
tf.disable_v2_behavior()
import cifar_reader                   # 读取cifar-10的函数


# In[2]:


batch_size = 100      # 每次处理100个图片，分批处理
s = 0           
show_s = 10
train_iter = 50000    # 共50000张训练图片


# In[3]:


#  placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，
# 它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3])  # 给input_x占个位 ，None个数不确定
y_lab = tf.placeholder(dtype=tf.float32, shape=[None, 10])               # 输出分值
rember = tf.placeholder(tf.float32)     # 遗忘率
train_sign = tf.placeholder(tf.bool)


# In[4]:


#### 卷积层1 
## tf.Variable()
## tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。
# 这个函数产生正太分布，均值和标准差自己设定
W1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32, stddev=1e-2))  
# 随机初始化权重，使用64个卷积核
'''
  tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
  除去name参数用以指定该操作的name，与方法有关的一共五个参数：
  input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为
float32和float64其中之一
  filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]
这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个
地方需要注意，第三维in_channels，就是参数input的第四维
  strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
  padding：string类型的量，只能是"SAME","VALID"其中之一
  结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。
'''
conv1 = tf.nn.conv2d(input, W1, strides=(1, 1, 1, 1), padding="VALID") 
# padding="VALID"，窗口移动，数据不足时直接舍弃
# BN 批量规范化
bn1 = tf.layers.batch_normalization(conv1, training=train_sign)
# 这个函数tf.nn.relu()的作用是计算激活函数 relu，即 max(features, 0):将大于0的保持不变，小于0的数置为0。
relu1 = tf.nn.relu(bn1)
# 池化层，选用最大法池化
pool1 = tf.nn.max_pool(relu1, strides=[1, 2, 2, 1], padding="VALID", ksize=[1, 3, 3, 1])
'''tf.nn.max_pool(value, ksize, strides, padding, name=None)
参数是四个，和卷积很类似：
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是
[batch, height, width, channels]这样的shape
第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在
batch和channels上做池化，所以这两个维度设为了1
第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式'''


# In[5]:


#### 卷积层2
W2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 32, 64], dtype=tf.float32, stddev=1e-2))
conv2 = tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding="SAME")
bn2 = tf.layers.batch_normalization(conv2, training=train_sign)
relu2 = tf.nn.relu(bn2)
pool2 = tf.nn.max_pool(relu2, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1], padding="VALID")


# In[6]:


#### 卷积层3
W3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], dtype=tf.float32, stddev=1e-1))
conv3 = tf.nn.conv2d(pool2, W3, strides=[1, 1, 1, 1], padding="SAME")
bn3 = tf.layers.batch_normalization(conv3, training=train_sign)
relu3 = tf.nn.relu(bn3)
pool3 = tf.nn.max_pool(relu3, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1], padding="VALID")


# In[7]:


# 全连接层
dense_tmp = tf.reshape(pool3, shape=[-1, 2*2*128])   # 有 2*2*128 个列
print(dense_tmp)
fc1 = tf.Variable(tf.truncated_normal(shape=[2*2*128, 512], stddev=0.01))
# tf.matmul() 矩阵点乘
bn_fc1 = tf.layers.batch_normalization(tf.matmul(dense_tmp, fc1), training=train_sign)
dense1 = tf.nn.relu(bn_fc1)
dropout1 = tf.nn.dropout(dense1, rember)
'''tf.nn.dropout()是tensorflow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层
Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，
让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。但是它的权重得保留下来（只是
暂时不更新而已），因为下次样本输入时它可能又得工作了'''


# In[8]:


# 全连接层2
fc2 = tf.Variable(tf.truncated_normal(shape=[512, 10], stddev=0.01))
out = tf.matmul(dropout1, fc2)
print(out)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_lab))
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
dr = cifar_reader.Cifar10DataReader(cifar_folder="C:/Users/Administrator/cifar-10-batches-py/")


# In[9]:


# 测试网络
correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y_lab, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[10]:


# 初始化所有的共享变量
init = tf.initialize_all_variables()
saver = tf.train.Saver()


# In[11]:


# 开启一个训练
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while s * batch_size < train_iter:
        s += 1
        batch_xs, batch_ys = dr.next_train_data(batch_size)
        # 获取批数据,计算精度, 损失值
        opt, acc, loss = sess.run([optimizer, accuracy, cost],feed_dict={input: batch_xs, y_lab: batch_ys, rember: 0.6, train_sign: True})
        if s % show_s == 0:
            print ("训练进度 " + str(s*batch_size)+ " 张" + ", 损失值 = " + "{:.6f}".format(loss) + ", 训练准确率 = " + "{:.5f}".format(acc))
    print ("训练完成。")
    num_examples = 10000
    d,l = dr.next_test_data(num_examples)
    print ("测试集准确率:", sess.run(accuracy, feed_dict={input: d, y_lab: l, rember: 1.0, train_sign: True}))
    saver.save(sess, "model_tmp/cifar10_demo.ckpt")






