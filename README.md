# Deep-learning-course-design


**一个简单的课程设计，只求混个及格**

​		具体代码在main.py文件中；然后在study_1文件夹中是一个比较复杂的代码，在网上找的学习了一下。

是一个训练cifar100的设计

###### 1.训练集直接从tensorflow的从datasets中直接获取

train （50000，32，32，3）
test（10000，32，32，3）

###### 2.训练模型

采用的是一个卷积神经网络

网络设计：
 第一层
      卷积：32个filter、大小5*5、strides=1、padding="SAME"
      激活：Relu
      池化：大小2x2、strides2
 第二层
      卷积：64个filter、大小5*5、strides=1、padding="SAME"
      激活：Relu
      池化：大小2x2、strides2
 全连接层
