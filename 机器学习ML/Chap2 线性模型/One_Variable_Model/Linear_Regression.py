import numpy as np
import pandas as pd

class Linear_Regression_One_Variable(object):
    def __init__(self):
        pass

    def dataloader(self, filepath):
        # 数据加载
        data = pd.read_csv(filepath)
        self.train_X = np.array(data[data.columns[0]])
        self.train_Y = np.array(data[data.columns[1]])

    def statistical_magnitude(self):
        # 计算数据集的各个统计量指标
        self.Sxx = np.sum((self.train_X - self.train_X.mean()) ** 2)
        self.Syy = np.sum((self.train_Y - self.train_Y.mean()) ** 2)
        self.Sxy = np.sum((self.train_X - self.train_X.mean()) * (self.train_Y - self.train_Y.mean()))
        self.Tss = self.Syy
        self.Ess = self.Sxy ** 2 / self.Sxx
        self.Rss = self.Tss - self.Ess
        self.r2_Score = 1 - self.Rss / self.Tss
    
    def regression_coefs(self):
        # 计算回归系数
        self.w = self.Sxy / self.Sxx
        self.b = self.train_Y.mean() - self.train_X.mean() * self.w
        return (self.w, self.b)