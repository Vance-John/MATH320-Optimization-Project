import numpy as np
import os
import struct

def load_mnist_images(filename):
    """
    Load MNIST images from the raw IDX3-ubyte binary file.
    Returns: numpy array of shape (N, 784)
    """
    with open(filename, 'rb') as f:
        # Read magic number, number of images, rows, cols
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        
        # Read the pixel data
        data = np.fromfile(f, dtype=np.uint8)
        
        # Reshape to (N, 784)
        data = data.reshape(num, rows * cols)
        
    return data

def load_mnist_labels(filename):
    """
    Load MNIST labels from the raw IDX1-ubyte binary file.
    Returns: numpy array of shape (N,)
    """
    with open(filename, 'rb') as f:
        # Read magic number, number of labels
        magic, num = struct.unpack(">II", f.read(8))
        
        # Read the label data
        data = np.fromfile(f, dtype=np.uint8)
        
    return data

def one_hot_encode(y, num_classes=2):
    """
    Convert binary labels (0/1) to one-hot vectors.
    0 -> [1, 0]
    1 -> [0, 1]
    """
    N = y.shape[0]
    y_onehot = np.zeros((N, num_classes))
    
    # y is a vector of 0s and 1s. 
    # If y[i] is 0, we want index 0 set to 1.
    # If y[i] is 1, we want index 1 set to 1.
    # We can use np.arange(N) and y to index directly.
    y_onehot[np.arange(N), y] = 1.0
    
    return y_onehot

def preprocess_and_save(data_dir, output_dir, seed=42):
    """
    Loads raw MNIST, filters to 0/1, splits, and saves to .npy format.
    """
    print("Loading raw data...")
    # Paths to raw files (adjust filenames if yours differ after unzipping)
    train_img_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_lbl_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_img_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_lbl_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

    X_train_all = load_mnist_images(train_img_path)
    y_train_all = load_mnist_labels(train_lbl_path)
    X_test_all = load_mnist_images(test_img_path)
    y_test_all = load_mnist_labels(test_lbl_path)

    # 1. Combine train and test temporarily to filter and shuffle uniformly
    # (Or keep separate as per standard MNIST practice, but your plan mentions 
    # a specific split strategy. We'll follow the plan: Filter -> Shuffle -> Split)
    X_all = np.concatenate([X_train_all, X_test_all], axis=0)
    y_all = np.concatenate([y_train_all, y_test_all], axis=0)

    print(f"Total MNIST size: {X_all.shape[0]}")

    # 2. Filter to Digits 0 and 1
    mask = (y_all == 0) | (y_all == 1)
    X_filtered = X_all[mask]
    y_filtered = y_all[mask]

    print(f"Filtered 0/1 size: {X_filtered.shape[0]}")

    # 3. Normalization (0-255 -> 0-1)
    X_filtered = X_filtered.astype(np.float32) / 255.0

    # 4. One-Hot Encoding
    Y_filtered = one_hot_encode(y_filtered)

    # 5. Deterministic Splitting
    np.random.seed(seed)
    indices = np.random.permutation(X_filtered.shape[0])
    
    X_shuffled = X_filtered[indices]
    Y_shuffled = Y_filtered[indices]

    # Defined in Project Plan: N_train ~ 10000, N_val = 2000
    N_train = 10000
    N_val = 2000
    # Remaining goes to test
    
    X_train = X_shuffled[:N_train]
    Y_train = Y_shuffled[:N_train]
    
    X_val = X_shuffled[N_train:N_train+N_val]
    Y_val = Y_shuffled[N_train:N_train+N_val]
    
    X_test = X_shuffled[N_train+N_val:]
    Y_test = Y_shuffled[N_train+N_val:]

    print(f"Final Splits:\n Train: {X_train.shape}\n Val:   {X_val.shape}\n Test:  {X_test.shape}")

    # 6. Save to disk
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train_01.npy'), X_train)
    np.save(os.path.join(output_dir, 'Y_train_01.npy'), Y_train)
    np.save(os.path.join(output_dir, 'X_val_01.npy'), X_val)
    np.save(os.path.join(output_dir, 'Y_val_01.npy'), Y_val)
    np.save(os.path.join(output_dir, 'X_test_01.npy'), X_test)
    np.save(os.path.join(output_dir, 'Y_test_01.npy'), Y_test)
    
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    # Adjust these paths to where your files actually are
    DATA_DIR = "MNIST_Data" 
    OUTPUT_DIR = "./data/processed"
    preprocess_and_save(DATA_DIR, OUTPUT_DIR)