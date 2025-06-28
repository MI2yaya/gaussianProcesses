import numpy as np
import matplotlib.pyplot as plt

X_train = np.linspace(-5, 5, 10).reshape(-1, 1)
Y_train = np.sin(X_train) + 0.1 * np.random.randn(*X_train.shape)

X_test = np.linspace(-6, 6, 200).reshape(-1, 1)

# RBF kernel function
def rbf_kernel(x1, x2, lengthscale=1.0, variance=1.0):
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return variance * np.exp(-0.5 / lengthscale**2 * sqdist)

# Posterior prediction function
def gp_posterior_predict(X_train, Y_train, X_test, kernel, noise_var=1e-4):
    K = kernel(X_train, X_train) + noise_var * np.eye(len(X_train))
    K_s = kernel(X_train, X_test)
    K_ss = kernel(X_test, X_test) + 1e-8 * np.eye(len(X_test))

    K_inv = np.linalg.inv(K)

    mu_s = K_s.T @ K_inv @ Y_train
    cov_s = K_ss - K_s.T @ K_inv @ K_s
    return mu_s.ravel(), cov_s

# Compute posterior
mu_s, cov_s = gp_posterior_predict(X_train, Y_train, X_test, rbf_kernel, noise_var=0.01)

# Compute standard deviation for confidence intervals
std_s = np.sqrt(np.diag(cov_s))

# Plotting
plt.figure(figsize=(10, 6))

# Plot training data
plt.plot(X_train, Y_train, 'ro', label="Training data")

# Plot GP mean prediction
plt.plot(X_test, mu_s, 'b-', label="GP mean prediction")

# Plot confidence interval (mean ± 2*std)
plt.fill_between(X_test.ravel(), mu_s - 2 * std_s, mu_s + 2 * std_s, color='blue', alpha=0.2, label="95% confidence interval")

plt.title("Gaussian Process Regression Posterior")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
