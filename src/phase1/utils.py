import numpy as np 

# Small activation/utility helpers used by the BinaryNN model.

def tanh(x):
    """Elementwise tanh activation."""
    return np.tanh(x)

def tanh_prime(x):
    """Derivative of tanh(x) = 1 - tanh^2(x), used in backprop."""
    return 1.0 - np.tanh(x)**2

def softmax(z):
    """
    Numerically stable softmax over the last dimension.
    Subtracting max avoids overflow when exponentiating large values.
    """
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z/ np.sum(exp_z, axis=1, keepdims=True)
