import numpy as np

# create the data for train
datas = np.array([[1, 3], [4, 9], [7, 15]], dtype=float) # [x, y]
val_datas = np.array([[2, 5], [3, 10]], dtype=float)

# define the model（it's up to your data）: y = ax + b
class UnaryLinearRegression:
    def __init__(self, datas):
        self.datas = datas

        # create the paramater
        self.W = np.array([0, 0], dtype=float)

    def Predict(self, x):  # x is singal variable
        return self.W[0] + self.W[1] * x   # W[0]=b W[1]=a y=b+ax

    def Loss(self, x, y): # use Mean Squared Error as loss
        return (1/2)*((self.Predict(x) - y)**2)  # ** is

    def Cost(self):
        coss = 0
        for data in self.datas:
            coss += self.Loss(data[0], data[1])
        return coss/datas.shape[0]

    def GD(self, k):
        temp1 = lambda datas:sum(self.W[0]+self.W[1]*data[0]-data[1] for data in datas)
        temp = self.W[0]
        self.W[0] = self.W[0] - k*temp1(self.datas)/datas.shape[0]  # b = b-E(b+ax-y)
        temp2 = lambda datas:sum((temp+self.W[1]*data[0]-data[1])*data[0] for data in datas)
        self.W[1] = self.W[1] - k*temp2(self.datas) / datas.shape[0]  # b = b-E((b+ax-y)x)


if __name__ == "__main__":
    # print(datas.shape[0])
    unaryLinearRegression = UnaryLinearRegression(datas)

    for i in range(10000):
        unaryLinearRegression.GD(0.01)
        print(i, "Cost: ", unaryLinearRegression.Cost())

    # Model Evaluation: Accuracy Precision Recall F1_score Confusion_Matrix Receiver_Operating_Characteristic


    pass
