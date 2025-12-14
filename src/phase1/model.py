import numpy as np
from .utils import tanh, tanh_prime, softmax

class BinaryNN:
    """
    Two-layer neural network for binary classification on flattened MNIST images.
    All parameters are stored as a single vector `w` so optimizers can work in
    vector space while we manually reshape inside the model.
    """

    def __init__(self, input_dim=784, hidden_dim=H, output_dim=2):
        """
        Args:
            input_dim: Number of input features (784 for 28x28 MNIST).
            hidden_dim: Width of the hidden layer (caller must supply H).
            output_dim: Number of classes (2 for binary).
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Parameter block shapes when unflattened from the single vector.
        self.shapes = {
            "W1": (hidden_dim, input_dim),
            "b1": (hidden_dim),
            "W2": (output_dim, hidden_dim),
            "b2": (output_dim)
        }           
        self.num_params = {
            hidden_dim * input_dim
            + hidden_dim
            +output_dim * hidden_dim
            +output_dim
        }

    def unflatten(self, w):
        """Split flat parameter vector into weight and bias tensors."""
        H, D = self.hidden_dim, self.input_dim
        out = self.output_dim

        p = 0
        W1 = w[p:p + H*D].reshape(H, D); p += H*D
        b1 = w[p:p + H]; p += H
        W2 = w[p:p + out*H].reshape(out, H); p += out*H
        b2 = w[p:p + out]

        return W1, b1, W2, b2

    def flatten(self, W1, b1, W2, b2):
        """Concatenate parameters back into a single vector matching unflatten order."""
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])
    
    def forward(self, X, w):
        """
        Compute network outputs for a batch.

        Args:
            X: Input batch shaped (N, input_dim).
            w: Flat parameter vector.
        Returns:
            h: Hidden activations after tanh.
            z1: Hidden pre-activations.
            yhat: Softmax probabilities for each class.
            z2: Output pre-activations.
        """
        W1, b1, W2, b2 = self.unflatten(w)

        z1 = X @ W1.T + b1        # (N, H)
        h = tanh(z1)              # (N, H)
        z2 = h @ W2.T + b2        # (N, 2)
        yhat = softmax(z2)        # (N, 2)

        return h, z1, yhat, z2
    
    def loss(self, X, Y, w):
        """Average cross-entropy loss for one-hot labels Y and predictions."""
        # Y is one-hot (N, 2)
        _, _, yhat, _ = self.forward(X, w)
        eps = 1e-12
        loss = -np.sum(Y * np.log(yhat + eps)) / X.shape[0]
        return loss
    
    def grad(self, X, Y, w):
        """
        Analytic gradient via backprop, returned as a flat vector matching flatten().
        """
        N = X.shape[0]

        W1, b1, W2, b2 = self.unflatten(w)
        h, z1, yhat, z2 = self.forward(X, w)

        # Output layer gradient
        dz2 = (yhat - Y) / N                  # (N,2)

        dW2 = dz2.T @ h                       # (2,H)
        db2 = np.sum(dz2, axis=0)             # (2,)

        # Hidden layer backprop
        dh = dz2 @ W2                         # (N,H)
        dz1 = dh * tanh_prime(z1)             # (N,H)

        dW1 = dz1.T @ X                       # (H,784)
        db1 = np.sum(dz1, axis=0)             # (H,)

        return self.flatten(dW1, db1, dW2, db2)
