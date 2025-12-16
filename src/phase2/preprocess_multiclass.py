import numpy as np
import os
import struct

def one_hot_encode(y, num_classes=5):
    """
    Convert labels (0-4) to one-hot vectors.
    """
    N = y.shape[0]
    y_onehot = np.zeros((N, num_classes))
    y_onehot[np.arange(N), y] = 1.0
    return y_onehot

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
    return data

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.uint8)
    return data

def preprocess_multiclass(data_dir, output_dir, seed=42):
    print("Loading raw data for Multiclass (0-4)...")
    # Adjust paths if necessary
    train_img = load_mnist_images(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    train_lbl = load_mnist_labels(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    test_img = load_mnist_images(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    test_lbl = load_mnist_labels(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))

    # Combine
    X_all = np.concatenate([train_img, test_img], axis=0)
    y_all = np.concatenate([train_lbl, test_lbl], axis=0)

    # Filter: Keep Digits 0, 1, 2, 3, 4
    mask = y_all <= 4
    X_filtered = X_all[mask]
    y_filtered = y_all[mask]

    print(f"Filtered (0-4) size: {X_filtered.shape[0]}")

    # Normalize
    X_filtered = X_filtered.astype(np.float32) / 255.0

    # One-Hot Encode (5 Classes)
    Y_filtered = one_hot_encode(y_filtered, num_classes=5)

    # Split
    np.random.seed(seed)
    indices = np.random.permutation(X_filtered.shape[0])
    X_shuffled = X_filtered[indices]
    Y_shuffled = Y_filtered[indices]

    N_train = 15000 # Increased slightly since we have more data
    N_test = X_shuffled.shape[0] - N_train

    X_train = X_shuffled[:N_train]
    Y_train = Y_shuffled[:N_train]
    X_test = X_shuffled[N_train:]
    Y_test = Y_shuffled[N_train:]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Save with NEW filenames
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train_04.npy'), X_train)
    np.save(os.path.join(output_dir, 'Y_train_04.npy'), Y_train)
    np.save(os.path.join(output_dir, 'X_test_04.npy'), X_test)
    np.save(os.path.join(output_dir, 'Y_test_04.npy'), Y_test)
    print("Saved 0-4 multiclass data.")

if __name__ == "__main__":
    preprocess_multiclass("./MNIST_Data", "./data/processed")