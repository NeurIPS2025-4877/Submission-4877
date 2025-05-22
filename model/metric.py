import numpy as np

def MSE(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def MAE(y, y_pred):
    return np.mean(np.abs(y - y_pred))


