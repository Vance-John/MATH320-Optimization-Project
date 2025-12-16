import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes json is in src/phase4/ or similar. Adjust path if needed.
results_path = os.path.join(current_dir, '../phase4_and_5/multi_class_experiment_results.json')

def plot_results():
    if not os.path.exists(results_path):
        print(f"Error: Could not find {results_path}")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    # Sort dimensions numerically (keys are strings in JSON)
    dims = sorted([int(k) for k in data.keys()])
    
    # Extract Data
    runtimes = {'SGD': [], 'BFGS': [], 'L-BFGS': []}
    accuracies = {'SGD': [], 'BFGS': [], 'L-BFGS': []}
    final_losses = {'SGD': [], 'BFGS': [], 'L-BFGS': []}
    
    # We need to track which dims have BFGS data (since we skipped it for large H)
    bfgs_dims = []

    for h in dims:
        h_str = str(h)
        res = data[h_str]
        
        # SGD and L-BFGS are present for all
        runtimes['SGD'].append(res['SGD']['runtime'])
        accuracies['SGD'].append(res['SGD']['test_accuracy'])
        final_losses['SGD'].append(res['SGD']['final_loss'])
        
        runtimes['L-BFGS'].append(res['L-BFGS']['runtime'])
        accuracies['L-BFGS'].append(res['L-BFGS']['test_accuracy'])
        final_losses['L-BFGS'].append(res['L-BFGS']['final_loss'])
        
        # BFGS might be missing
        if 'BFGS' in res:
            runtimes['BFGS'].append(res['BFGS']['runtime'])
            accuracies['BFGS'].append(res['BFGS']['test_accuracy'])
            final_losses['BFGS'].append(res['BFGS']['final_loss'])
            bfgs_dims.append(h)

    # --- FIGURE 1: Runtime Comparison (The "Cliff") ---
    plt.figure(figsize=(10, 6))
    plt.plot(dims, runtimes['SGD'], 'o-', label='SGD', color='green')
    plt.plot(dims, runtimes['L-BFGS'], 's-', label='L-BFGS (m=10)', color='blue')
    plt.plot(bfgs_dims, runtimes['BFGS'], '^-', label='BFGS (Full)', color='red')
    
    plt.title('Optimizer Runtime vs. Hidden Layer Dimension', fontsize=14)
    plt.xlabel('Hidden Dimension (H)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.yscale('log') # Log scale is crucial here!
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('MCfigure1_runtime_log.png')
    print("Saved MCfigure1_runtime_log.png")

    # --- FIGURE 2: Test Accuracy ---
    plt.figure(figsize=(10, 6))
    plt.plot(dims, accuracies['SGD'], 'o--', label='SGD', color='green', alpha=0.7)
    plt.plot(dims, accuracies['L-BFGS'], 's-', label='L-BFGS', color='blue')
    plt.plot(bfgs_dims, accuracies['BFGS'], '^-', label='BFGS', color='red')
    
    plt.title('Test Accuracy vs. Hidden Dimension', fontsize=14)
    plt.xlabel('Hidden Dimension (H)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.99, 1.0005) # Zoom in on the top 1% since results are so good
    plt.grid(True, alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('MCfigure2_accuracy.png')
    print("Saved MCfigure2_accuracy.png")

    # --- FIGURE 3: Convergence (H=20 Case Study) ---
    if '20' in data and 'BFGS' in data['20']:
        h20 = data['20']
        plt.figure(figsize=(10, 6))
        plt.plot(h20['SGD']['loss_history'], label='SGD', color='green')
        plt.plot(h20['L-BFGS']['loss_history'], label='L-BFGS', color='blue', linewidth=2)
        plt.plot(h20['BFGS']['loss_history'], label='BFGS', color='red', linestyle='--')
        
        plt.title('Training Loss Trajectory (H=20)', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Binary Cross Entropy Loss', fontsize=12)
        plt.yscale('log')
        plt.grid(True, which="both", alpha=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('MCfigure3_convergence_h20.png')
        print("Saved MCfigure3_convergence_h20.png")

if __name__ == "__main__":
    plot_results()