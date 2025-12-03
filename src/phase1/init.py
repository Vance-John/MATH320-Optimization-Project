import numpy as np

def xavier_init(size_in, size_out):
    """
    Xavier/Glorot uniform initializer.
    Draws weights from U(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out)))
    so that forward activations and backpropagated gradients have similar scale,
    reducing early-layer saturation.
    """
    bound = np.sqrt(6 / (size_in + size_out))
    return np.random.uniform(-bound, bound, (size_out, size_in))

def init_parameters(model):
    """
    Initialize all network parameters using Xavier for weights and zeros for biases,
    then flatten into the single vector format expected by BinaryNN.
    """
    W1 = xavier_init(model.input_dim, model.hidden_dim)
    b1 = np.zeros(model.hidden_dim)

    W2 = xavier_init(model.hidden_dim, model.output_dim)
    b2 = np.zeros(model.output_dim)

    return model.flatten(W1, b1, W2, b2)
