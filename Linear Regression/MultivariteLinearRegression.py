import numpy as np
import random


# create the data
datas = np.array([[1, 1, 3], [4, 4, 9], [7, 7, 15]], dtype=float) # [x_1, x_2, y]
val_datas = np.array([[2, 2, 5], [3, 3, 10]], dtype=float)

# define the model（it's up to your data）: y = ax_1 + bx_2 + c
class MultivaariteLinearRegression():
    def __init__(self, datas):
        self.datas = datas
        # create the paramater
        self.W = np.array([0, 0, 0], dtype=float)
        pass

    def predict(self, x): # x is vector, like [1, 1]、 [4, 4]
        x = np.append(x, 1.0)
        return x@(self.W.reshape(-1, 1))  # output a scalar

    def Loss(self, x, y):
        return (1/2)*((self.predict(x)-y)**2)

    def Cost(self):
        cost = 0
        for data in self.datas:
            x = data[:2]
            y = data[2]
            cost += self.Loss(x, y)
        return cost/self.datas.shape[0]

    def SGD(self, k):  # k is learnig rate
        i = random.randint(0, len(self.datas)-1) # the scope of generate random number contains the maximum
        data = self.datas[i]
        x = np.append(data[:2], 1.0)
        y = data[2]
        self.W -= k*x*(x@self.W.T - y)

if __name__ == "__main__":
    multivaariteLinearRegression = MultivaariteLinearRegression(datas)
    for epoch in range(10000):
        multivaariteLinearRegression.SGD(0.01)
        print("epoch:", epoch, " Cost:",multivaariteLinearRegression.Cost())
