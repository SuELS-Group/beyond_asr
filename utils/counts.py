import numpy as np

def basic_step_count(step_count):
    return (1/step_count)

def exp_step_count(step_count):
    return (1/(2**step_count))

def log_step_count(step_count):
    return -np.log2(step_count + 1)

def square_step_count(step_count):
    return (1/(step_count**2))
