import numpy as np

def numerical_grad(f, w, eps=1e-6):
    g_num = np.zeros_like(w)
    for j in range(len(w)):
        w_pos = w.copy(); w_pos[j] += eps
        w_neg = w.copy(); w_neg[j] -= eps
        g_num[j] = (f(w_pos) - f(w_neg)) / (2*eps)
    return g_num

def check_gradient(model, X, Y, w, tol=1e-4):
    analytic = model.grad(X, Y, w)
    f = lambda w_: model.loss(X, Y, w_)
    num = numerical_grad(f, w)

    rel_err = np.linalg.norm(analytic - num) / (np.linalg.norm(analytic) + np.linalg.norm(num))

    print("Gradient check:")
    print("‣ ‖analytic - numeric‖ =", np.linalg.norm(analytic - num))
    print("‣ Relative error =", rel_err)

    return rel_err < tol