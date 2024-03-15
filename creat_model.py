import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

import os



class LeNet(Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), activation="sigmoid", input_shape=(28, 28, 1))
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.c2 = Conv2D(filters=16, kernel_size=(5, 5), activation="sigmoid")
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(120, activation="sigmoid")
        self.f2 = Dense(84, activation="sigmoid")
        self.f3 = Dense(10, activation="softmax")
        # 配置训练方法
        self.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["sparse_categorical_accuracy"]
        )

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y

if __name__ == "__main__":

    # 加载数据集
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2],1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # y_train = tf.keras.utils.to_categorical(y_train, 10)
    # y_test = tf.keras.utils.to_categorical(y_test, 10)
    print(np.shape(x_train[0]),y_train.shape)

    # model = BaseLine()
    model = LeNet()




    # 断点续训，读取模型
    # checkpoint_save_path = "cifar10/BaseLine.ckpt"
    checkpoint_save_path = "./mnist/LeNet.ckpt"
    if os.path.exists(checkpoint_save_path + ".index"):
        print("*******load the model******")
        model.load_weights(checkpoint_save_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_weights_only=True,
        save_best_only=True
    )



    # 训练模型
    history = model.fit(x_train, y_train, batch_size=128, epochs=7,
                        validation_split=0.3,
                        validation_freq=1, callbacks=[cp_callback])

    # 打印网络结构和参数
    model.summary()
    # 写入参数
    with open("mnist_lenet_weights.txt", "w") as f:
        for v in model.trainable_variables:
            f.write(str(v.name) + "\n")
            f.write(str(v.shape) + "\n")
            f.write(str(v.numpy()) + "\n")

    # 显示训练和预测的acc、loss曲线
    acc = history.history["sparse_categorical_accuracy"]
    val_acc = history.history["val_sparse_categorical_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="train acc")
    plt.plot(val_acc, label="validation acc")
    plt.title("train & validation acc")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="train loss")
    plt.plot(val_loss, label="validation loss")
    plt.title("train & validation loss")
    plt.legend()
    plt.show()
