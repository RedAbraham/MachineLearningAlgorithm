import numpy as np

# create the data for train
datas = np.array([[1, 3], [4, 6], [7, 9]]) # [x, y]

# define the model（it's up to data）: y = ax + b
class UnaryLinearRegression:
    def __init__(self, datas):
        self.datas = datas

        # create the paramater
        self.W = np.array([0, 0])

    def Predict(self, x):  # x is singal variable
        return self.W[0] + self.W[1] * x

    def Loss(self, x, y): # use Mean Squared Error as loss
        return (self.Predict(x) - y)**2

    def Cost(self):
        coss = 0
        for data in self.datas:
            coss += self.Loss(data[0], data[1])
        return coss



