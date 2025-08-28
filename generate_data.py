import matplotlib.pyplot as plt
import numpy as np
import math

def get_data(true_sigma1, true_sigma2, n=1000, odd=0, true_e1=0, true_e2=0):
    np.random.seed(42)
    x_train = np.random.uniform(0, 10, size=(n, 1))
    x_test = np.random.uniform(10, 15, size=(odd, 1))
    eps1 = np.random.normal(true_e1, true_sigma1, size=(n, 1))
    eps2 = np.random.normal(true_e2, true_sigma2, size=(n, 1))
    y_train = x_train * np.sin(x_train) + eps1 * x_train + eps2
    if odd > 0:   
        eps3 = np.random.normal(true_e1, true_sigma1, size=(odd, 1))
        eps4 = np.random.normal(true_e2, true_sigma2, size=(odd, 1))
        y_test = x_test * np.sin(x_test) + eps3 * x_test + eps4
        return x_train, x_test, y_train, y_test

    return x_train, y_train #float64, ndarray

# # Plot data
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, s=10, alpha=0.6, label='Noisy data')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Generated Data: y = x * sin(x) + noise")
# plt.legend()
# plt.grid(True)
# plt.show()