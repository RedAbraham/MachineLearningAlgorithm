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

    # Model Evaluation: MSE MAE R^2 MAPE
    # ①MSE 1/N*(y-y_predict)^2
    total = 0
    for val in val_datas:
        y_predict = unaryLinearRegression.Predict(val[0])
        total += (val[1]-y_predict)**2
    print("MES: ", total/val_datas.shape[0])

    #②MAE 1/N*|y-y_predict|
    total = 0
    for val in datas:
        y_predict = unaryLinearRegression.Predict(val[0])
        total += abs(val[1]-y_predict)
    print("MAS: ", total/val_datas.shape[0])

    # ③ R^2
    total1 , total2 = 0, 0
    y_mean = val_datas.mean(axis=0)[1]
    for val in val_datas:
        y_predict = unaryLinearRegression.Predict(val[0])
        total1 += (val[1]-y_predict)**2
        total2 += (y_predict-y_mean)**2
    print("R^2: ", 1-(total1/total2))

    # ④MAPE
    total = 0
    for val in datas:
        y_predict = unaryLinearRegression.Predict(val[0])
        total += abs((val[1]-y_predict)/val[1])
    print("MAPE: ", total/val_datas.shape[0]*100, "%")

    pass
