import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, alpha=0.001, max_iter=1000, min_error=0.001):
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_error = min_error
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        # correction of the size of the data
        if len(X.shape) == 1: 
            X = X.reshape((len(X), 1))

        n_samples, n_features = X.shape

        # initialization of the weights and bias
        self.weights = np.zeros(n_features)        
        self.bias = 0

        for _ in range(self.max_iter):
            
            # prediction with the actual weights and bias
            y_pred = np.dot(X, self.weights) + self.bias

            # derivates of the cost function
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)

            # update of the actual weights
            self.weights -= self.alpha*dw
            self.bias -= self.alpha*db

            # error check to avoid unecessary iterations
            if np.linalg.norm(y - y_pred) <= self.min_error: break

    def predict(self, x):
        return np.dot(x,self.weights) + self.bias

if __name__ == "__main__":
    
    #------------------------------
    # One weight regression test
    #------------------------------

    # y = 2x

    # x,y = np.arange(0,10,0.5), 2*np.arange(0,10,0.5)

    # noise = np.random.normal(0,1,y.shape) 
    # y += noise

    # lg = LinearRegression(alpha=0.02, min_error=0.001)
    # lg.fit(x,y)

    # plt.scatter(x, y, c='b', label='Data')
    # plt.plot(x, lg.weights*x+lg.bias, c='r', label='Fit')
    # plt.legend()
    # plt.show()

    #------------------------------
    # Two weights regression test
    #------------------------------

    # y = 5x1 - 10x2 + 4

    # x = np.array([[i,j] for i in range(5) for j in range(5)])
    # y = np.array([5*i - 10*j + 4 for i,j in x])


    # lg = LinearRegression(alpha=0.01, max_iter=5000)
    # lg.fit(x,y)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(x[:,0], x[:,1], y, label='points')
    # ax.plot_trisurf(x[:,0], x[:,1], np.dot(lg.weights, x.T) + lg.bias)
    # ax.legend()
    # plt.show()

    pass