import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# create the data
datas = np.array([[1, 1, 0], [2, 4, 1], [5, 2, 0], [1, 2, 0], [4, 7, 1], [7, 10, 1]], dtype=float) # [x_1, x_2, y]
val_datas = np.array([[2, 1, 0], [3, 9, 1]], dtype=float)

class LogisticRegression:
    def __init__(self, datas):
        self.datas = datas
        # create the paramater
        self.W = np.array([0, 0, 0], dtype=float)

    def predict(self, x): # x is vector, like [1, 1]、 [4, 4]
        x = np.append(x, 1.0)
        return sigmoid(x@(self.W.reshape(-1, 1)))


    def loss(self, x, y): # x is vector, like [1, 1]、 [4, 4]
        return -(y*np.log(self.predict(x))+(1-y)*np.log(1-self.predict(x)))

    def cost(self):
        cost = 0
        for data in self.datas:
            x = data[:2]
            y = data[2]
            cost += self.loss(x, y)
        return cost/self.datas.shape[0]

    def GD(self, k):
        X = self.datas[:, :2]
        ones_column = np.ones((X.shape[0], 1), dtype=float)
        X = np.hstack((X, ones_column)) # horizontal stack
        Y = self.datas[:, 2:3]
        self.W = (self.W.reshape(-1, 1) - k*(1/self.datas.shape[0])*X.T@(sigmoid(X@self.W.T).reshape(-1, 1) - Y)).T


if __name__ == '__main__':
    logisticRegression = LogisticRegression(datas)
    for epoch in range(30000):
        logisticRegression.GD(0.1)
        print("epoch:", epoch, " Cost:", logisticRegression.cost())

