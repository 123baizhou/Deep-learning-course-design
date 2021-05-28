from tensorflow.python.keras.datasets import cifar100
from tensorflow.python import keras
import tensorflow as tf


class CNNMnist(object):
    #编写两层 + 两层全连接层网络模型
    model = keras.Sequential([
        # 卷积层1：32个5*5*3的filter,strides=1,padding='same'
        keras.layers.Conv2D(32, kernel_size=5, strides=1,
                            padding='same', data_format='channels_last', activation=tf.nn.relu),
        # 池化层1：2*2的窗口，strides=2
        keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        # 卷积层2：64个5*5*32的filter,strides=1,padding='same'
        keras.layers.Conv2D(64, kernel_size=5, strides=1,
                            padding='same', data_format='channels_last', activation=tf.nn.relu),
        # 池化层2：2*2的窗口，strides=2
        keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same'),
        # [None, 8, 8, 64]—— > [None, 8 * 8 * 64]
        keras.layers.Flatten(),
        # 全连接层神经网络
        # 1024个神经元神经网络层
        keras.layers.Dense(1024, activation=tf.nn.relu),
        # 100个神经元神经网络
        keras.layers.Dense(100, activation=tf.nn.softmax),
    ])

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar100.load_data()

        #数据归一化
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    # print(x_train.shape)
    # print(x_test.shape)

    def compile(self):

        CNNMnist.model.compile(optimizer=keras.optimizers.Adam(),
                               loss=tf.keras.losses.sparse_categorical_crossentropy,
                               metrics=['accuracy'])
        return None

    #训练
    def fit(self):

        CNNMnist.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)

        return None

    #评估
    def evaluate(self):

        test_loss, test_acc = CNNMnist.model.evaluate(self.x_test, self.y_test)

        print(test_loss, test_acc)
        return None


if __name__ == '__main__':
    cnn = CNNMnist()

    cnn.compile()

    cnn.fit()

    cnn.predict()

    print(CNNMnist.model.summary())

