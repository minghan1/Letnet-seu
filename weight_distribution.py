import tensorflow as tf
import os
from creat_model import LeNet
import numpy as np
import matplotlib.pyplot as plt
import seu
import tqdm
import copy
import fault_injection
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # 禁用gpu

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = LeNet()
    checkpoint_save_path_noseu = "./mnist/LeNet.ckpt"
    checkpoint_save_path_seu = "./mnist_seu/LeNet.ckpt"
    checkpoint_save_path = checkpoint_save_path_noseu
    result_savepath_noseu = "./pic/weights_distri.png"
    result_savepath_seu = "./pic/seu_weights_distri.png"
    result_savepath = result_savepath_noseu
    sel = 0
    if sel == 0:
        checkpoint_save_path = checkpoint_save_path_noseu
        result_savepath = result_savepath_noseu
    elif sel == 1:
        checkpoint_save_path = checkpoint_save_path_seu
        result_savepath = result_savepath_seu
    print(checkpoint_save_path)
    if os.path.exists(checkpoint_save_path + ".index"):
        print("*******load the model******")
        model.load_weights(checkpoint_save_path)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    weights = model.get_weights()
    # print(test_loss, test_acc)
    # model.summary()
    print()
    y = []
    for i in range(len(weights)):
        if (i != 8):
            continue
        else:
            y = np.ndarray.flatten(weights[i])

    # example data
    mu = np.mean(y)  # mean of distribution
    sigma = np.sqrt(np.var(y))  # standard deviation of distribution
    print(mu, sigma, np.var(y))
    num_bins = 42

    fig, ax = plt.subplots()

    # the histogram of the data
    n, bins, patches = ax.hist(y, num_bins, density=True, range=(-1,1))

    # add a 'best fit' line
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
         np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
    ax.plot(bins, y, '--')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability density')
    ax.set_title('Histogram of lenet.layer[7]: '
                 fr'$\mu={mu:.3f}$, $\sigma={sigma:.3f}$')

    # Tweak spacing to prevent clipping of ylabel
    # fig.tight_layout()
    plt.savefig("./pic/weights_distri5.png")
    plt.show()