import sys
import os
import time
import json
import numpy as np

# --- PATH SETUP 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
phase1_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'phase1'))
phase3_dir = os.path.abspath(os.path.join(current_script_dir, '..', 'phase3'))

for path in [phase1_dir, phase3_dir]:
    if path not in sys.path:
        sys.path.append(path)
# ----------------------------------------------------

from model import BinaryNN
from init import init_parameters
from optimizers import BFGS, LBFGS, SGD

# Configuration
# Reduced H sizes and Iterations to ensure runtime feasibility
SMALL_DIMS = [2, 3, 4, 5, 10] 
LARGE_DIMS = [20, 30, 50, 100]
MAX_ITER = 50 
RESULTS_FILE = os.path.join(current_script_dir, "multi_class_experiment_results.json")

def load_data():
    """Locate and load the preprocessed full dataset."""
    # Try to find data relative to the script location
    project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'data', 'processed')
    
    print(f"Loading data from: {data_dir}")
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train_04.npy'))
        Y_train = np.load(os.path.join(data_dir, 'Y_train_04.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test_04.npy'))
        Y_test = np.load(os.path.join(data_dir, 'Y_test_04.npy'))
        return X_train, Y_train, X_test, Y_test
    except FileNotFoundError:
        print("CRITICAL: Data files not found. Did you run Phase 2 preprocess.py?")
        sys.exit(1)

def compute_accuracy(model, X, Y, w):
    """Computes classification accuracy: (Correct / Total)"""
    # Forward pass
    _, _, yhat, _ = model.forward(X, w)
    # yhat is (N, 2) probabilities
    # Y is (N, 2) one-hot
    
    preds = np.argmax(yhat, axis=1)
    targets = np.argmax(Y, axis=1)
    
    correct = np.sum(preds == targets)
    return float(correct) / X.shape[0]

def run_experiments():
    X_train, Y_train, X_test, Y_test = load_data()
    print(f"Data Loaded. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    results = {}

    ALL_DIMS = sorted(LARGE_DIMS + SMALL_DIMS)

    for H in ALL_DIMS:
        print(f"\n{'='*40}")
        print(f"STARTING EXPERIMENTS FOR HIDDEN DIMENSION H={H}")
        print(f"{'='*40}")
            # 1. Initialize Model & Weights
            # We use the SAME w0 for all optimizers to ensure fair starting point
        model = BinaryNN(hidden_dim=H, output_dim = 5)
        w0 = init_parameters(model)
        
        results[H] = {}
        
            # 2. Define Optimizers to Test
        optimizers = {
            "SGD": SGD(model, learning_rate=0.1, max_iter=MAX_ITER),
                
            "L-BFGS": LBFGS(model, m=5, max_iter=MAX_ITER)
        }
        if H in SMALL_DIMS: 
            optimizers["BFGS"] = BFGS(model, max_iter=MAX_ITER)
        
        for name, opt in optimizers.items():
            print(f"\n>> Running {name} (H={H})...")
            
            # Run Optimization
            res = opt.optimize(X_train, Y_train, w0)
            
            # Compute Final Test Accuracy
            test_acc = compute_accuracy(model, X_test, Y_test, res.w_final)
            
            print(f"   Done! Time: {res.runtime:.2f}s | Final Loss: {res.loss_history[-1]:.6f} | Test Acc: {test_acc:.4f}")
            
            # Store Results
            results[H][name] = {
                "runtime": res.runtime,
                "iterations": res.iterations,
                "final_loss": res.loss_history[-1],
                "test_accuracy": test_acc,
                "loss_history": res.loss_history,
                "grad_norm_history": res.grad_norm_history,
                "status": res.status
            }

    # 3. Save to JSON
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n\nAll experiments complete. Results saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    run_experiments()