import numpy as np

def exp_step_count(step_count):
    return (1/(2**step_count))

def log_step_count(step_count):
    return -np.log2(step_count + 1)
