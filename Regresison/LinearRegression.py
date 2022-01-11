
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


class LinearRegression:
    '''
    An implimentation of linear regression. 
    
    Parameters
    ----------
    learning_rate: integar
        Learning rate for gradient descent 

    n_steps: integar
        Number of steps for gradient descent 

    l2_penalty: boolean
        If you want to use an l2 penalty

    lambda_val: integar
        The learning rate for the l2 penalty
    
    '''
    def __init__(self, learning_rate = 1e-3, n_steps = 100, l2_penalty = True, lambda_val = 0.1):

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.l2_penalty = l2_penalty
        self.lambda_val = lambda_val
    
    def calculate_gradient(self, X, y):
        '''
        Derivative of the cost function. 

        Extended to include the derivative of the cost function with an l2 penalty. 
        '''
        if self.l2_penalty:
            gradient = 2/X.shape[0] * np.dot(X.T, (np.dot(X, self.W) - y)) + (2 * self.lambda_val * self.W)
        else:
            gradient = 2/X.shape[0] * np.dot(X.T, (np.dot(X, self.W) - y))

        return gradient 
        
    def gradient_descent(self, X, y):
        '''
        Iteratively update the weights for n-steps via the 
        Gradient Descent algorithm. 
        '''
        
        for i in range(self.n_steps):
            self.W -= self.learning_rate * self.calculate_gradient(X, y)
       
        return self.W
        
    def fit(self, X, y):
        '''
        Learn the optimal parameters of the model.
        '''
        
        # add bias term to X matrix 
        Xtrain = np.c_[np.ones(X.shape[0]), X]
        
        # randomly initialise the weights
        self.W = np.random.rand((Xtrain.shape[1])) 
        
        # Update the weights
        self.W = self.gradient_descent(Xtrain, y)
    
    def predict(self, X):
        '''
        Predict y for unseen X. 
        '''    
        
        # add bias term to X matrix 
        Xpred = np.c_[np.ones(X.shape[0]), X]
        
        return np.dot(Xpred, self.W)
    
    def coef_(self):
        '''
        Return model weights
        '''
        return self.W[1:]
    
    def intercept_(self):
        '''
        Return model bias term
        '''
        return self.W[0]
        

if __name__ == "__main__":

    # make data 
    data = datasets.make_regression(n_samples=10000, n_features=10, n_targets=1, 
                                    bias=0.0)
    X = data[0]
    y = data[1]

    # split into train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, 
                                                        random_state=1)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # instantiate object and fit model  
    model = LinearRegression(n_steps=10000, l2_penalty=True, lambda_val=0.001)
    model.fit(X_train, y_train)     

    # training loss
    y_pred = model.predict(X_train)
    print(f"Training loss: {mean_squared_error(y_train, y_pred)}")

    # testing loss 
    y_pred = model.predict(X_test)
    print(f"Testing loss: {mean_squared_error(y_test, y_pred)}")

    # model weights
    print(f"Model weights: {model.coef_()}")

    # model bias 
    print(f"Model bias: {model.intercept_()}")