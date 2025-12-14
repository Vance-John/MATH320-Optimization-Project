import sys
import os
import numpy as np

# --- PATH SETUP START ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Calculate the path to 'src/phase1' relative to this script
phase1_dir = os.path.join(current_script_dir, '..', 'phase1')

# 3. Resolve it to an absolute path
phase1_dir = os.path.abspath(phase1_dir)

# 4. Add it to sys.path if it's not already there
if phase1_dir not in sys.path:
    sys.path.append(phase1_dir)
    print(f"DEBUG: Added to python path: {phase1_dir}")

# 5. Also add the current directory (phase3) just in case
if current_script_dir not in sys.path:
    sys.path.append(current_script_dir)
# --- PATH SETUP END ---

# Now we can import
try:
    from model import BinaryNN
    # Import the correct initialization function from phase1/init.py
    from init import init_parameters 
    print("SUCCESS: Successfully imported BinaryNN and init_parameters")
except ImportError as e:
    print("\nCRITICAL ERROR: Could not import modules.")
    print(f"Python is searching in these folders:\n{sys.path}")
    print(f"Specific Error: {e}")
    sys.exit(1)

# Import optimizers from the local folder
from optimizers import BFGS, LBFGS

# --- Sanity Check Script ---
H = 10 # Standardize on the hidden dimension used in your project plan

def run_sanity_check(data_dir="./data/processed"):
    print("--- Phase 3 Sanity Check Initiated ---")
    
    # 1. Load Data
    try:
        # Load a small batch for speed
        # Note: We need to go up two levels from src/phase3 to find data/processed
        # Or you can provide the absolute path. 
        # Assuming run from project root or src/phase3, let's try to locate it relative to script.
        project_root = os.path.join(current_script_dir, '..', '..')
        data_path_abs = os.path.abspath(os.path.join(project_root, 'data', 'processed'))
        
        X_train_full = np.load(os.path.join(data_path_abs, 'X_train_01.npy'))
        Y_train_full = np.load(os.path.join(data_path_abs, 'Y_train_01.npy'))
        
        # Use a small subset to ensure quick run time
        SUBSET_SIZE = 100 
        X_subset = X_train_full[:SUBSET_SIZE]
        Y_subset = Y_train_full[:SUBSET_SIZE]
        print(f"Loaded a subset of {SUBSET_SIZE} samples for testing.")

    except FileNotFoundError as e:
        print(f"Error: Could not find processed data.")
        print(f"Looked in: {data_path_abs}")
        print(f"Details: {e}")
        return

    # 2. Setup Model and Initial Weights
    # Input is 784, Hidden is H, Output is 2
    nn_model = BinaryNN(hidden_dim=H)
    
    # Use the function from init.py
    w0 = init_parameters(nn_model)
    
    print(f"Model initialized with H={H}. Total parameters: {len(w0)}")
    
    # 3. Run BFGS
    print("\n--- Testing BFGS (Full-Memory) ---")
    bfgs_optimizer = BFGS(model=nn_model, max_iter=20, tol_grad=1e-8)
    bfgs_result = bfgs_optimizer.optimize(X_subset, Y_subset, w0)
    
    print(f"\nBFGS Final Status: {bfgs_result.status}")
    print(f"Final Loss: {bfgs_result.loss_history[-1]:.6f}")
    print(f"Total Iterations: {bfgs_result.iterations}")
    
    # 4. Run L-BFGS
    print("\n--- Testing L-BFGS (Limited-Memory) ---")
    lbfgs_optimizer = LBFGS(model=nn_model, m=5, max_iter=20, tol_grad=1e-8)
    lbfgs_result = lbfgs_optimizer.optimize(X_subset, Y_subset, w0)

    print(f"\nL-BFGS Final Status: {lbfgs_result.status}")
    print(f"Final Loss: {lbfgs_result.loss_history[-1]:.6f}")
    print(f"Total Iterations: {lbfgs_result.iterations}")

    print("\n--- Sanity Check Complete ---")

if __name__ == "__main__":
    run_sanity_check()