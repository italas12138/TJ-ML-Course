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


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc



import os
# 获取脚本文件的目录
script_dir = os.path.dirname(os.path.realpath(__file__))

# 将工作目录切换到脚本文件的目录
os.chdir(script_dir)

class Logistic:
    def __init__(self, Alpha=0.001, epochs=200):
        self.Alpha = Alpha
        self.epochs = epochs

        self.train_num = None
        self.feature = None

        self.weights = None
        self.bias = None

        self.accuracy_list = []
        self.y_percent_list = []
        self.y_pred_list = []
    

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def get_accracy(self):
        return self.accuracy_list
    
    def SGD(self, X, y, mini_batch_size, optimization = "None"):
        self.train_num, self.feature_num = X.shape

        # 初始化模型参数
        self.weights = np.zeros(self.feature_num)
        self.bias = 1

        print(f'epoch0:  parameter:{self.weights}, bias:{self.bias:.3f}')

        for epoch in range(self.epochs):
            # 将数据打乱分批进行训练
            training_data = list(zip(X, y))
            random.shuffle(training_data)

            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, self.train_num, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, optimization)
            
            print(f'epoch{epoch + 1}:  parameter:{self.weights}, bias:{self.bias:.3f}')
            self.accuracy_list.append(model.predict(X_test, y_test))

            # 如果采用学习率衰减的训练方法，下面的内容要使用
            # if (epoch+1) % 20 == 0:
            #     self.Alpha *= 0.5

        print(f'训练后的参数:  parameter:{self.weights}, bias:{self.bias:.3f}')

    def update_mini_batch(self, a_batch_of_data, optimization):

        # 权重和偏差的更新值初始化
        delta_w = np.zeros(self.feature_num)
        delta_b = 0

        for i in range(len(a_batch_of_data)):
            x = a_batch_of_data[i][0] # x的训练值
            y = a_batch_of_data[i][1] # y的真实值

            linear_output = np.dot(x, self.weights) + self.bias
            y_hat = self.sigmoid(linear_output)

            if optimization == "L1":
                # 如果你 使用 L1 正则化来修改损失函数
                Lambda_L1 = 0.01
                delta_w += self.Alpha * ((y_hat - y) * x + Lambda_L1 * np.sign(self.weights))

            elif optimization == "L2":
                # 如果你 使用 L2 正则化来修改损失函数
                Lambda_L2 = 0.1
                if np.sum(self.weights**2) == 0:
                    delta_w += self.Alpha * ((y_hat - y) * x)
                else :
                    delta_w += self.Alpha * ((y_hat - y) * x + Lambda_L2 * self.weights / math.sqrt(np.sum(self.weights**2)))

            else:
                # 如果你 不使用 正则化方法 来修改损失函数
                delta_w += self.Alpha * (y_hat - y) * x

            delta_b += self.Alpha * (y_hat - y)

        self.weights -= delta_w / len(a_batch_of_data)
        self.bias -= delta_b / len(a_batch_of_data)



    def predict(self, X, y):
        test_num = X.shape[0]
        accuracy_num = 0
        self.y_percent_list = []
        self.y_pred_list = []
        for i in range(test_num):
            linear_output = np.dot(X[i], self.weights) + self.bias
            y_pred = self.sigmoid(linear_output)
            y_pred_classes = np.where(y_pred > 0.5, 1, 0)

            if y_pred_classes == y[i]:
                accuracy_num = accuracy_num + 1
            
            self.y_percent_list.append(y_pred)
            self.y_pred_list.append(y_pred_classes)
            
        accuracy = accuracy_num / test_num
        print("正确率：{:.3f}".format(accuracy))
        return accuracy

    def get_y_pred_list(self):
        return self.y_pred_list
    
    def get_y_percent_list(self):
        return self.y_percent_list
    
    def get_weights(self):
        return self.weights
    


def result_display(model, model_feature, y_test):
    # 打印训练过程的正确率
    range_1 = range(0,200)
    accuracy_set = model.get_accracy()

    figure1, = plt.plot(range_1, accuracy_set)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.xticks(np.linspace(0, 200, 11))
    plt.legend(handles=[figure1], labels=['Accuracy'], loc='best')
    plt.savefig('accuracy_epochs.pdf', format='pdf')
    plt.show()


    # 打印各特征的重要性条形图
    # 获取特征系数
    feature_coefficients = model.get_weights()
    # 计算特征重要性（系数的绝对值）
    feature_importance = np.abs(feature_coefficients)

    # 绘制条形图
    plt.bar(model_feature, feature_importance)
    plt.xlabel('feature')
    plt.ylabel('importance')
    plt.title('feature_importance')
    plt.xticks(rotation=90)
    plt.savefig('weights_importance.pdf', format='pdf')
    plt.show()

    


    # 打印预测概率曲线
    # 排序预测概率
    y_percent = model.get_y_percent_list()
    sorted_scores = sorted(y_percent, reverse=True)

    # 计算累积概率
    cumulative_probs = [float(i+1) / len(sorted_scores) for i in range(len(sorted_scores))]

    # 绘制预测概率曲线
    plt.plot(cumulative_probs, sorted_scores, color='b', label='Prediction Probability')
    # 累积概率
    plt.xlabel('Cumulative Probability')
    # 预测概率
    plt.ylabel('Prediction Probability')
    plt.title('Model Prediction Probability Curve')
    plt.legend(loc="lower right")
    plt.savefig('percent_prediction.pdf', format='pdf')
    plt.show()


    
    # 打印混淆矩阵
    y_pred = model.get_y_pred_list()
    cm = confusion_matrix(y_test, y_pred)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    # 计算精确率
    precision = precision_score(y_test, y_pred)
    # 计算召回率
    recall = recall_score(y_test, y_pred)
    # 计算F1分数
    f1 = f1_score(y_test, y_pred)   

    # 绘制混淆矩阵
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['True', 'False']  # 类别标签
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # 在每个方格中显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('data_matrix.pdf', format='pdf')
    plt.show()

    # 打印指标
    print("正确率 Accuracy: {:.3f}%".format(accuracy * 100))
    print("精确率 Precision: {:.3f}%".format(precision * 100))
    print("召回率 Recall: {:.3f}%".format(recall * 100))
    print("F1分数 F1 Score: {:.3f}%".format(f1 * 100))


    # 打印ROC曲线，并计算AUC
    # 计算ROC曲线的假正率（FPR）、真正率（TPR）和阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_percent)

    # 计算AUC值
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='b', label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('ROC.pdf', format='pdf')
    plt.show()

    # 打印AUC值
    print("AUC: {:.3f}".format(roc_auc))




if __name__ == "__main__":

    # 数据获取
    train_df = pd.read_csv('processed_train.csv')
    test_df = pd.read_csv('processed_test.csv')

    # 将原始数据分为训练集和验证集
    data_train, data_test = train_test_split(train_df, test_size=0.2)

    # 选择四个特征来训练模型
    model_one_feature = ['CryoSleep', 'RoomService', 'Spa', 'VRDeck']
    X_train = data_train[model_one_feature].values
    X_test = data_test[model_one_feature].values
    y_train = data_train['Transported'].values
    y_test = data_test['Transported'].values    

    # 线性模型建立
    model = Logistic()

    # 设置打印浮点数组的精度
    np.set_printoptions(precision=3)
    
    # 在训练集上训练数据，三个参数分别是 训练轮次、每轮中的最小批次、学习率
    model.SGD(X_train, y_train, 20)

    # 在测试集上进行检验
    print("四特征的训练结果：")
    model.predict(X_test, y_test)

    result_display(model, model_one_feature, y_test)








    # # 选择八个特征来训练模型
    # model_two_feature = ['CryoSleep', 'RoomService', 'Spa', 'VRDeck', 'Side', 'HomePlanet_num', 'Destination_num', 'Deck_num']
    # X_train = data_train[model_two_feature].values
    # X_test = data_test[model_two_feature].values
    # y_train = data_train['Transported'].values
    # y_test = data_test['Transported'].values    

    # model = Logistic()
    # np.set_printoptions(precision=3)
    # model.SGD(X_train, y_train, 20)
    # print("八特征的训练结果：")
    # model.predict(X_test, y_test)

    # result_display(model, model_two_feature, y_test)


    # # 选择全特征来训练模型
    # y_train = data_train['Transported'].values
    # y_test = data_test['Transported'].values    
    # data_three_train = data_train.drop('Transported', axis=1)
    # data_three_test = data_test.drop('Transported', axis=1)
    # X_train = data_three_train.values
    # X_test = data_three_test.values

    # model = Logistic()
    # np.set_printoptions(precision=3)
    # model.SGD(X_train, y_train, 20)
    # print("全特征的训练结果：")
    # model.predict(X_test, y_test)

    # model_three_feature = data_three_train.columns.tolist()
    # result_display(model, model_three_feature, y_test)






