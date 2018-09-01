# python Record_BiRNN1.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import os


np.random.seed(0)
tf.set_random_seed(1234)

'''
モデルファイル用設定
'''
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model1')

if os.path.exists(MODEL_DIR) is False:
    os.mkdir(MODEL_DIR)


def inference(x, n_in=None, maxlen=None, n_hidden=None, n_out=None):
    def weight_variable(shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name=None):
        initial = tf.zeros(shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    # 時系列データの形式をAPIの仕様に合わせるため、最終的に
    # (ミニバッチサイズ, 入力次元数) が時間長分ある形に変形
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_in])
    x = tf.split(x, maxlen, 0)

    cell_forward = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    cell_backward = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _, _ = \
        rnn.static_bidirectional_rnn(cell_forward, cell_backward, x,
                                     dtype=tf.float32)

    W = weight_variable([n_hidden * 2, n_out])
    b = bias_variable([n_out])
    y = tf.nn.softmax(tf.matmul(outputs[-1], W) + b)

    return y


def loss(y, t):
    cross_entropy = \
        tf.reduce_mean(-tf.reduce_sum(
                       t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)),
                       reduction_indices=[1]))
    return cross_entropy


def training(loss):
    optimizer = \
        tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999)
    train_step = optimizer.minimize(loss)
    return train_step


def accuracy(y, t):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False


if __name__ == '__main__':
    '''
    データの生成
    '''

    ###  学習が可能なデータセット
    Musle_data = pd.read_csv('TeachingData/Musle.csv', encoding='utf-8', header=None)
    AccX_data = pd.read_csv('TeachingData/AccX.csv', encoding='utf-8', header=None)
    AccY_data = pd.read_csv('TeachingData/AccY.csv', encoding='utf-8', header=None)
    AccZ_data = pd.read_csv('TeachingData/AccZ.csv', encoding='utf-8', header=None)
    target = pd.read_csv('TeachingData/Y.csv', encoding='utf-8', header=None)

    Musle_data = Musle_data.values
    AccX_data = AccX_data.values
    AccY_data = AccY_data.values
    AccZ_data = AccZ_data.values
    target = target.values

    maxlen = len(Musle_data[0])  # ひとつの時系列データの長さ
    target = np.reshape(target, len(Musle_data))

    X1 = np.array(Musle_data).reshape(len(Musle_data), maxlen, 1)
    X2 = np.array(AccX_data).reshape(len(AccX_data), maxlen, 1)
    X3 = np.array(AccY_data).reshape(len(AccY_data), maxlen, 1)
    X4 = np.array(AccZ_data).reshape(len(AccZ_data), maxlen, 1)
    Y = np.eye(3)[target.astype(int)] # one-hotベクトルに変換

    X =[]
    for i in range(len(X1)): # 180
        k = np.concatenate((X1[i],X2[i],X3[i],X4[i]), axis = 1)
        X.append(k)

    X = np.array(X).reshape(len(X1), maxlen, 4)
    Y = np.eye(3)[target.astype(int)] # one-hotベクトルに変換

    # データ設定
    N_train = int(len(Musle_data) * 0.8)
    N_validation = len(Musle_data) - N_train

    X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X, Y, test_size=N_validation)

    '''
    モデル設定
    '''

    n_in = len(X[0][0])  # 4
    n_hidden = 300
    n_out = len(Y[0])  # 3

    x = tf.placeholder(tf.float32, shape=[None, maxlen, n_in])
    t = tf.placeholder(tf.float32, shape=[None, n_out])

    y = inference(x, n_in=n_in, maxlen=maxlen, n_hidden=n_hidden, n_out=n_out)
    loss = loss(y, t)
    train_step = training(loss)

    accuracy = accuracy(y, t)

    early_stopping = EarlyStopping(patience=10, verbose=1)
    history = {
        'val_loss': [],
        'val_acc': []
    }

    '''
    モデル学習
    '''
    epochs =  100
    batch_size = 36

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()  # モデル保存用
    sess = tf.Session()
    sess.run(init)

    n_batches = N_train // batch_size

    for epoch in range(epochs):
        X_, Y_ = shuffle(X_train, Y_train)

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            sess.run(train_step, feed_dict={
                x: X_[start:end],
                t: Y_[start:end]
            })

        val_loss = loss.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation
        })
        val_acc = accuracy.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation
        })

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print('epoch:', epoch,
              ' validation loss:', val_loss,
              ' validation accuracy:', val_acc)

        if early_stopping.validate(val_loss):
            break

    # モデル保存
    model_path = saver.save(sess, MODEL_DIR + '/model.ckpt')
    print('Model saved to:', model_path)

    '''
    最終的な予測精度の評価
    '''
    accuracy_rate = accuracy.eval(session=sess, feed_dict={
        x: X_validation,
        t: Y_validation
    })
    print('accuracy : ', accuracy_rate)

    '''
    学習の進み具合をグラフで可視化
    '''
    accuracy = history['val_acc']
    loss = history['val_loss']

    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(9,6))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8)
    # 損失のプロット
    ax_acc = fig.add_subplot(111)
    ax_acc.plot(range(len(accuracy)), accuracy,
             label='Accuracy', color='red', lw=0.7)
    # 正解率のプロット
    ax_loss = ax_acc.twinx()
    ax_loss.plot(range(len(loss)), loss,
             label='Loss', color='blue', lw=0.7)

    ax_acc.set_xlabel('Loss & Accuracy Map')
    ax_acc.grid(True)
    ax_acc.set_xlabel('epochs')
    ax_acc.set_ylabel('accuracy')
    ax_loss.set_ylabel('loss')
    ax_acc.legend(bbox_to_anchor=(0, 1.17), loc='upper left', borderaxespad=0.2, fontsize=10)
    ax_loss.legend(bbox_to_anchor=(0, 1.1), loc='upper left', borderaxespad=0.2, fontsize=10)
    plt.show()
