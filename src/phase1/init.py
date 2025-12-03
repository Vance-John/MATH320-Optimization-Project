import numpy as np

def xavier_init(size_in, size_out):
    bound = np.sqrt(6 / (size_in + size_out))
    return np.random.uniform(-bound, bound, (size_out, size_in))

def init_parameters(model):
    W1 = xavier_init(model.input_dim, model.hidden_dim)
    b1 = np.zeros(model.hidden_dim)

    W2 = xavier_init(model.hidden_dim, model.output_dim)
    b2 = np.zeros(model.output_dim)

    return model.flatten(W1, b1, W2, b2)