import numpy as np
import pandas as pd

class Linear_Regression_Multi_Variable(object):
    def __init__(self):
        # 用来标记是否在x的最后一列补上过1
        self.mark = 0

    def Dataloader(self, filepath):
        # 加载数据，代码可以根据数据集适当调整
        self.data = pd.read_csv(filepath)
        self.data_x = self.data[self.data.columns[1:-1]]
        self.data_y = self.data[self.data.columns[-1]]

    def Train_test_split(self, test_ratio=0.25, seed=None):
        # test_ratio是训练集和测试集的划分百分比，定义为：测试集/总数据集合
        # 默认为：25%
        # 设置随机数种子，保证每次生成的结果都是一样的
        if seed:
            np.random.seed(seed)
        # permutation随机生成0-len(data)随机序列
        shuffled_indices = np.random.permutation(len(self.data_x))
        test_set_size = int(len(self.data_x) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        self.train_x, self.train_y = np.array(self.data_x.iloc[train_indices]), \
                                     np.array(self.data_y[train_indices])
        self.test_x, self.test_y = np.array(self.data_x.iloc[test_indices]), \
                                   np.array(self.data_y.iloc[test_indices])
        self.column_one_train = np.array([[1] * len(self.train_x)]).reshape(len(self.train_x), 1)
        self.column_one_test = np.array([[1] * len(self.test_x)]).reshape(len(self.test_x), 1)
        # 在x的最后一列补上1，构成X
        if self.mark == 0:
            self.train_x = np.hstack((self.train_x, self.column_one_train))
            self.test_x = np.hstack((self.test_x, self.column_one_test))
            self.mark = 1

    def regression_coefs(self): 
        self.H = np.matmul(self.train_x.T, self.train_x)
        # 判断矩阵X'X是否是满秩矩阵
        if np.all(abs(np.linalg.eigvals(self.H)) > 1e-4):
            self.coef_ = np.linalg.inv(self.H).dot(self.train_x.T).dot(self.train_y)
        else:
            print("X'X不是可逆矩阵，无法使用正规方程求解！")

    def predict(self, input_x):
        # 预测
        pred_y = (input_x).dot(self.coef_)
        return pred_y

    def statistical_magnitude(self):
        # 训练集上的RMSE、r2
        # 计算数据集的各个回归指标
        y_hat = (self.train_x).dot(self.coef_)
        self.rmse = np.sqrt(np.sum((self.train_y - y_hat) ** 2) / len(self.train_x))
        self.r2_score = 1 - np.sum((self.train_y - y_hat) ** 2) / \
                        np.sum((self.train_y - y_hat.mean()) ** 2)