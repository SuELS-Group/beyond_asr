import numpy as np

def exp(step_count):
    return (1/(np.e**step_count))

def log(step_count):
    return np.log(step_count + 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
