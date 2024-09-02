import numpy as np
import pandas as pd

#导入 train_test_split 函数，实现数据集的划分，将其分为训练集和测试集
from sklearn.model_selection import train_test_split

#导入 r2_score 函数，在测试集上计算 R2（决定系数）
from sklearn.metrics import r2_score

import seaborn as sns

import matplotlib.pyplot as plt

import random
import math


# 数据处理部分

class Data_Process():
    def __init__(self):

        # 读取目标数据集，将其储存在 self.data 中
        self.data = pd.read_csv('Student_Performance.csv')

        # 将每一列，按照索引名划分
        self.data.columns = {'Hours Studied': self.data.columns[0], 'Previous Scores': self.data.columns[1],
                             'Extracurricular Activities': self.data.columns[2], 'Sleep Hours': self.data.columns[3],
                             'Sample Question Papers Practiced': self.data.columns[4], 'Performance Index': self.data.columns[5]}
        # 数据去重
        self.data.drop_duplicates()

        # 过滤数据集，将其中的 Extracurricular Activities 去掉
        self.data = self.data.drop('Extracurricular Activities', axis=1)


        #计算线性相关系数 皮尔逊相关系数 观察4个特征与目标值之间的关系
        data = np.array(self.data)
        data_R = np.corrcoef(data.T)
        print(f'皮尔逊相关系数为: \n{data_R}')

        data_frame = pd.DataFrame(data, columns=['Hours Studied', 'Previous Scores', 'Sleep Hours',
                             'Sample Question Papers Practiced', 'Performance Index'])


        # 热力图
        # sns.heatmap(data_R, annot=True, cmap='coolwarm')
        # plt.savefig('heatmap.pdf', format='pdf')
        # plt.show()



        # 散点矩阵图
        # sns.pairplot(data_frame)
        # plt.savefig('scatterplot_matrix.pdf', format='pdf')
        # plt.show()




        # 存储归一化后的新数据
        self.new_data = np.array([[0.0 for _ in range(self.data.shape[1])] for __ in range(self.data.shape[0])])
        # new_data[i]即为第i??数据的一行数??

        # 存储mean max min
        self.mmm = [[0.0 for _ in range(3)] for __ in range(self.data.shape[1] - 1)]

        # mmm[i]即为第i个参数，分别为mean,max,min

        # 归一化算法采用x'=(x-mean(x))/(max-min)
        i = 0
        while i < self.data.shape[1] - 1:
            # ??遍历x进???归一化，不用??y
            arr = self.data.iloc[:, i]
            self.mmm[i][0] = arr.mean()
            self.mmm[i][1] = arr.max()
            self.mmm[i][2] = arr.min()
            j = 0
            while j < self.data.shape[0]:
                self.new_data[j][i] = (arr[j] - self.mmm[i][0]) / (self.mmm[i][1] - self.mmm[i][2])
                j = j + 1
            i = i + 1

        i = 0
        while i < self.data.shape[0]:
            # 遍历y,复制入新数组
            self.new_data[i][self.data.shape[1] - 1] = self.data.iloc[i, self.data.shape[1] - 1]
            i = i + 1

        # 数据集随机分为训练集和测试集
        self.data_train, self.data_test = train_test_split(self.new_data, test_size=0.1)

    # 获得训练集数据
    def get_data_train(self):
        return self.data_train

    # 获得测试集数据
    def get_data_test(self):
        return self.data_test


# 计算均方损失
def MSE(y_hat, y):
    return (y_hat - y) ** 2


# 线性回归模型

class Liner():
    # inout_x 为训练集上的 X 数据，inout_y 为训练集上的 Y 数据
    def __init__(self, data_train):
        input_x = data_train[:, 0:4]
        input_y = data_train[:, -1]

        self.X = input_x  # np矩阵，形状为（数据个数 ，参数个数）
        self.num = input_x.shape[0]  # 数据个数
        self.Y = input_y  # 形状（n,1） n为数据个数

        # self.W = np.zeros((input_x.shape[1], 1))  权重初始化，全为0
        self.W = np.zeros((self.X.shape[1], 1))  # 权重初始化，全为0
        self.B = np.random.randn()  # 偏差初始化，为随机值

        # loss_set 储存训练过程中的 loss 损失值
        self.loss_set = []

        # test_loss 储存测试过程中的 loss 损失值
        self.test_loss = []

        # test_loss_per 储存测试过程中的 loss 损失值的偏差百分比
        self.test_loss_per = []


    def get_loss_set(self):
        return self.loss_set

    def get_test_loss(self):
        return self.test_loss

    def get_loss_per(self):
        return self.test_loss_per


    # 前向预测
    def forward(self, x):
        prediction = np.dot(x, self.W) + self.B
        return prediction


    def SGD(self, epochs, mini_batch_size, Alpha):  # 随机梯度下降（训练周期，最小批次数据量，学习率）
        # 将数据打乱分批进行训练
        print(f'epoch0:  parameter:{self.W.T}, bias:{self.B}')
        # 打印出当前的所有参数值，以便实时观测更新

        for epoch in range(epochs):
            training_data = list(zip(self.X, self.Y))
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, self.num, mini_batch_size)]
            for mini_batch in mini_batches:
                loss = self.updata_mini_batch(mini_batch, Alpha)
            self.loss_set.append(loss)
            if epoch % 1 == 0:
                print(f'epoch{epoch + 1}:  parameter:{self.W.T}, bias:{self.B},loss:{loss:.3f}')

            # 如果采用学习率衰减的训练方法，下面的内容要使用
            # if (epoch+1) % 20 == 0:
            #     Alpha *= 0.5

        print(f'训练后的参数:  parameter:{self.W.T}, bias:{self.B}')

    # 更新权重与偏差，返回损失函数的平均值
    def updata_mini_batch(self, a_batch_of_data, Alpha):
        loss = np.array([])
        i = 0
        # 权重和偏差的更新值
        delta_w = np.zeros((self.X.shape[1], 1))
        delta_b = 0

        while i < len(a_batch_of_data):
            x = a_batch_of_data[i][0] # x的训练值
            y = a_batch_of_data[i][1] # y的真实值


            y_hat = self.forward(x)   # y的预测值
            loss = np.append(loss, MSE(y_hat, y))

            # 如果你 使用 L1 正则化来修改损失函数
            # Lambda_L1 = 0.1
            # delta_w += Alpha * ((y_hat - y) * x.reshape(-1, 1) + Lambda_L1 * np.sign(self.W))

            # 如果你 使用 L2 正则化来修改损失函数
            # Lambda_L2 = 0.1
            # if (math.sqrt(np.sum(self.W**2)) ) == 0:
            #     delta_w += Alpha * ((y_hat - y) * x.reshape(-1, 1))
            # else :
            #     delta_w += Alpha * ((y_hat - y) * x.reshape(-1, 1) + Lambda_L2 * self.W / math.sqrt(np.sum(self.W**2)))

            # 如果你 不使用 正则化方法 来修改损失函数
            delta_w += Alpha * (y_hat - y) * x.reshape(-1, 1)

            delta_b += Alpha * (y_hat - y)
            i = i+1

        self.W -= delta_w / len(a_batch_of_data)
        self.B -= delta_b / len(a_batch_of_data)
        return loss.mean()


    def Test(self, data_test):
        test_x = data_test[:, 0:4]
        test_y = data_test[:, -1]

        # y_hat_set 储存测试集上预测值集合， y_set 储存测试集上真实值集合
        y_hat_set = []
        y_set = []

        test_loss_value = 0

        i=0;
        while i<len(test_x):
            x = test_x[i]
            y = test_y[i]

            y_hat = self.forward(x)

            y_set.append(y)
            y_hat_set.append(y_hat)

            test_loss_value = MSE(y_hat, y)
            self.test_loss.append(test_loss_value)

            self.test_loss_per.append(abs(y_hat-y)/y)

            test_loss_value = 0
            i = i + 1

        r2 = r2_score(y_set, y_hat_set)
        print(f'测试集上的R2 相关系数为:{r2}')


if __name__ == "__main__":
    # 数据预处理
    dataset = Data_Process()

    # 获得训练集和测试集数据
    data_train = dataset.get_data_train()
    data_test = dataset.get_data_test()

    # 线性模型建立
    model = Liner(data_train)

    # 设置打印浮点数组的精度
    np.set_printoptions(precision=3)
    # 在训练集上训练数据，三个参数分别是 训练轮次、每轮中的最小批次、学习率
    model.SGD(200, 20, 0.001)

    # 如果采用学习率递减的方法，请使用下面的训练函数
    # model.SGD(200, 20, 0.01)

    # 在测试集上进行检验
    model.Test(data_test)


    # 打印结果
    range_1 = range(0,200)
    loss_set = model.get_loss_set()

    figure1, = plt.plot(range_1, loss_set)
    plt.xlabel('Iteration')
    plt.ylabel('Loss Sqrt')
    plt.xticks(np.linspace(0, 200, 11))
    plt.legend(handles=[figure1], labels=['train_loss'], loc='best')
    plt.savefig('train_loss.pdf', format='pdf')
    plt.show()


    range_2 = range(0, len(data_test))
    test_loss = model.get_test_loss()

    figure2, = plt.plot(range_2, test_loss)
    plt.xlabel('Iteration')
    plt.ylabel('Test Loss')
    plt.xticks(np.linspace(0, len(data_test), 11))
    plt.legend(handles=[figure2], labels=['test_loss'], loc='best')
    plt.savefig('test_loss.pdf', format='pdf')
    plt.show()

    print(f'测试集上的均方损失值为: {np.mean(test_loss)}')





    bin_1_width = 0.2
    bins_1 = np.arange(0, np.max(test_loss) + bin_1_width, bin_1_width)
    hist_1, _ = np.histogram(test_loss, bins=bins_1, density=True)
    bin_1_centers = (bins_1[:-1] + bins_1[1:]) / 2

    hist_1 = hist_1*100

    # 绘制折线图
    plt.plot(bin_1_centers, hist_1, marker='none')
    plt.xlabel('test_loss_value')
    plt.ylabel('Persent')
    plt.title('Probability Distribution')
    plt.savefig('test_loss_persent.pdf', format='pdf')
    plt.show()



    range_3 = range(0, len(data_test))
    test_loss_per = model.get_loss_per()
    figure3, = plt.plot(range_3, test_loss_per)
    plt.xlabel('Iteration')
    plt.ylabel('Test Loss percent')
    plt.xticks(np.linspace(0, len(data_test), 11))
    plt.legend(handles=[figure3], labels=['test_loss'], loc='best')
    plt.savefig('test_loss_per.pdf', format='pdf')
    plt.show()

    print(f'测试集上的平均偏差百分比为: {np.mean(test_loss_per)}')

    bin_2_width = 0.01
    bins_2 = np.arange(0, np.max(test_loss_per) + bin_2_width, bin_2_width)
    hist_2, _ = np.histogram(test_loss_per, bins=bins_2, density=True)
    bin_2_centers = (bins_2[:-1] + bins_2[1:]) / 2


    # 绘制折线图
    plt.plot(bin_2_centers, hist_2, marker='o')
    plt.xlabel('test_loss_per_value')
    plt.ylabel('Persent')
    plt.title('Probability Distribution')
    plt.savefig('test_loss_per_persent.pdf', format='pdf')
    plt.show()