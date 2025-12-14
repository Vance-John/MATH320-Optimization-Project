import numpy as np
import time

class OptimizationResult:
    """Container for optimization results."""
    def __init__(self, w_final, loss_history, grad_norm_history, iterations, runtime, status):
        self.w_final = w_final
        self.loss_history = loss_history
        self.grad_norm_history = grad_norm_history
        self.iterations = iterations
        self.runtime = runtime
        self.status = status # 'converged', 'max_iter', 'error'

def backtracking_line_search(loss_fn, w, p, current_loss, current_grad, c=1e-4, rho=0.5, max_ls_iter=20):
    """
    Backtracking line search satisfying the Armijo condition (Sufficient Decrease).
    
    Condition: f(w + alpha * p) <= f(w) + c * alpha * grad(w)^T p
    """
    alpha = 1.0
    dir_deriv = np.dot(current_grad, p) # grad^T p
    
    # If direction is not a descent direction (shouldn't happen with BFGS/L-BFGS if theoretically sound), 
    # we might have issues.
    if dir_deriv > 0:
        # Fallback: reverse direction or warn? 
        # For quasi-Newton, p = -H g. If H is positive definite, -g^T H g < 0.
        pass

    for i in range(max_ls_iter):
        w_next = w + alpha * p
        loss_next = loss_fn(w_next)
        
        if loss_next <= current_loss + c * alpha * dir_deriv:
            return alpha, loss_next, w_next
        
        alpha *= rho # reduce step size
    
    # If line search fails, return small step
    return alpha, loss_fn(w + alpha * p), w + alpha * p

class BFGS:
    """
    Full-Memory BFGS Optimizer.
    Maintains an approximation of the Inverse Hessian (H).
    """
    def __init__(self, model, max_iter=100, tol_grad=1e-5, tol_change=1e-9):
        self.model = model
        self.max_iter = max_iter
        self.tol_grad = tol_grad
        self.tol_change = tol_change
        
    def optimize(self, X, Y, w0):
        start_time = time.time()
        
        # 1. Initialization
        w = w0.copy()
        N = len(w)
        I = np.eye(N)
        H = I.copy() # Initial Inverse Hessian approximation
        
        loss_fn = lambda w_: self.model.loss(X, Y, w_)
        grad_fn = lambda w_: self.model.grad(X, Y, w_)
        
        current_loss = loss_fn(w)
        current_grad = grad_fn(w)
        
        loss_history = [current_loss]
        grad_norm_history = [np.linalg.norm(current_grad)]
        
        print(f"[BFGS] Init Loss: {current_loss:.6f} | Grad Norm: {grad_norm_history[-1]:.6f}")

        status = "max_iter"
        
        for k in range(self.max_iter):
            # Check convergence
            if np.linalg.norm(current_grad) < self.tol_grad:
                status = "converged_grad"
                break
            
            # 2. Compute search direction p = -H * g
            p = -H @ current_grad
            
            # 3. Line Search
            alpha, next_loss, w_next = backtracking_line_search(loss_fn, w, p, current_loss, current_grad)
            
            # Check small change in w
            s = w_next - w
            if np.linalg.norm(s) < self.tol_change:
                status = "converged_step"
                w = w_next
                break
                
            # 4. Update
            next_grad = grad_fn(w_next)
            y = next_grad - current_grad
            
            # 5. Update Hessian Approximation (BFGS Update)
            rho = 1.0 / (np.dot(y, s) + 1e-10) # Safety term
            
            # (I - rho * s * y^T)
            V = I - rho * np.outer(s, y)
            
            # H_next = V^T * H * V + rho * s * s^T
            H = V.T @ H @ V + rho * np.outer(s, s)
            
            # Move to next step
            w = w_next
            current_loss = next_loss
            current_grad = next_grad
            
            # Logging
            loss_history.append(current_loss)
            grad_norm_history.append(np.linalg.norm(current_grad))
            
            if k % 10 == 0:
                print(f"[BFGS] Iter {k}: Loss={current_loss:.6f}, |g|={grad_norm_history[-1]:.6f}")

        runtime = time.time() - start_time
        return OptimizationResult(w, loss_history, grad_norm_history, k, runtime, status)


class LBFGS:
    """
    Limited-Memory BFGS (L-BFGS) Optimizer.
    Does not store H. Uses two-loop recursion with history of size m.
    """
    def __init__(self, model, m=10, max_iter=100, tol_grad=1e-5, tol_change=1e-9):
        self.model = model
        self.m = m
        self.max_iter = max_iter
        self.tol_grad = tol_grad
        self.tol_change = tol_change
        
    def _two_loop_recursion(self, grad, s_list, y_list):
        """
        Computes H * g using the stored (s, y) pairs.
        """
        q = grad.copy()
        alphas = []
        k = len(s_list)
        
        # First loop (backward)
        for i in range(k - 1, -1, -1):
            s_i = s_list[i]
            y_i = y_list[i]
            rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
            
            alpha_i = rho_i * np.dot(s_i, q)
            alphas.append(alpha_i)
            q = q - alpha_i * y_i
            
        # Initial Hessian approximation gamma * I
        if k > 0:
            s_last = s_list[-1]
            y_last = y_list[-1]
            gamma = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-10)
        else:
            gamma = 1.0
            
        r = gamma * q
        
        # Second loop (forward)
        # We stored alphas in reverse order (k-1 down to 0), so we reverse them back 
        # to match iterating from 0 to k-1
        alphas = alphas[::-1] 
        
        for i in range(k):
            s_i = s_list[i]
            y_i = y_list[i]
            rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
            alpha_i = alphas[i]
            
            beta = rho_i * np.dot(y_i, r)
            r = r + s_i * (alpha_i - beta)
            
        return r

    def optimize(self, X, Y, w0):
        start_time = time.time()
        
        w = w0.copy()
        
        loss_fn = lambda w_: self.model.loss(X, Y, w_)
        grad_fn = lambda w_: self.model.grad(X, Y, w_)
        
        current_loss = loss_fn(w)
        current_grad = grad_fn(w)
        
        # History buffers
        s_list = [] # steps
        y_list = [] # grad differences
        
        loss_history = [current_loss]
        grad_norm_history = [np.linalg.norm(current_grad)]
        
        print(f"[L-BFGS m={self.m}] Init Loss: {current_loss:.6f} | Grad Norm: {grad_norm_history[-1]:.6f}")

        status = "max_iter"

        for k in range(self.max_iter):
            if np.linalg.norm(current_grad) < self.tol_grad:
                status = "converged_grad"
                break
            
            # 1. Compute search direction p = - H * g using Two-Loop
            # Note: two_loop returns H*g, so we neglect the minus sign there and add it here
            Hg = self._two_loop_recursion(current_grad, s_list, y_list)
            p = -Hg
            
            # 2. Line Search
            alpha, next_loss, w_next = backtracking_line_search(loss_fn, w, p, current_loss, current_grad)
            
             # Check small change
            s = w_next - w
            if np.linalg.norm(s) < self.tol_change:
                status = "converged_step"
                w = w_next
                break

            # 3. Update History
            next_grad = grad_fn(w_next)
            y = next_grad - current_grad
            
            # Store new pair
            if np.dot(y, s) > 1e-10: # Only update if curvature condition holds
                s_list.append(s)
                y_list.append(y)
                # Prune history if > m
                if len(s_list) > self.m:
                    s_list.pop(0)
                    y_list.pop(0)
            
            # Move step
            w = w_next
            current_loss = next_loss
            current_grad = next_grad
            
            loss_history.append(current_loss)
            grad_norm_history.append(np.linalg.norm(current_grad))
            
            if k % 10 == 0:
                print(f"[L-BFGS] Iter {k}: Loss={current_loss:.6f}, |g|={grad_norm_history[-1]:.6f}")

        runtime = time.time() - start_time
        return OptimizationResult(w, loss_history, grad_norm_history, k, runtime, status)
    
class SGD:
        """
        Stochastic Gradient Descent (Full-Batch for this baseline).
        Updates parameters using: w = w - lr * grad
        """
        def __init__(self, model, learning_rate=0.1, max_iter=100, tol_grad=1e-5):
            self.model = model
            self.lr = learning_rate
            self.max_iter = max_iter
            self.tol_grad = tol_grad

        def optimize(self, X, Y, w0):
            start_time = time.time()
            w = w0.copy()
        
            # Initial Logging
            current_loss = self.model.loss(X, Y, w)
            current_grad = self.model.grad(X, Y, w)
        
            loss_history = [current_loss]
            grad_norm_history = [np.linalg.norm(current_grad)]
            status = "max_iter"
        
            print(f"[SGD lr={self.lr}] Init Loss: {current_loss:.6f}")

            for k in range(self.max_iter):
                # Check convergence
                if grad_norm_history[-1] < self.tol_grad:
                    status = "converged_grad"
                    break
            
                # Update Step
                # Recalculate grad (standard SGD)
                grad = self.model.grad(X, Y, w)
                w = w - self.lr * grad
            
                # Logging
                # Note: Computing loss every step is expensive but needed for the plot
                loss = self.model.loss(X, Y, w)
                grad_norm = np.linalg.norm(grad)
            
                loss_history.append(loss)
                grad_norm_history.append(grad_norm)
            
                if k % 10 == 0:
                    print(f"[SGD] Iter {k}: Loss={loss:.6f}")
                
            runtime = time.time() - start_time
            return OptimizationResult(w, loss_history, grad_norm_history, k, runtime, status)