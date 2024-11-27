import numpy as np
from numpy.random import default_rng as rng
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_fun
import mpmath as mp

N = 10000


def gamma_dist(a, b):
    print(f'Gamma distribution with mean={a/b:.3} and variance={a/(b*b):.3}')
    gamma_draws = rng().gamma(a, 1/b, N)
    _, x, _ = plt.hist(gamma_draws, 50, density=True)
    y = b**a * x**(a-1) * (np.exp(-b*x) / gamma_fun(a))
    plt.plot(x, y, linewidth=2, color='r')
    plt.show()


# Define parameters
mean = 1  # Fixed mean of the gamma distribution
shape_parameters = np.linspace(0.5, 4, 10)  # Shape parameters
scale = mean  # Scale parameter is fixed to 1 since mean = shape * scale

# Create x values for plotting
x = np.linspace(0, 5, 500)

# Plot gamma distribution for different shape parameters
for a in shape_parameters:
    b = a / mean
    y = b ** a * x ** (a - 1) * (np.exp(-b * x) / gamma_fun(a))
    plt.plot(x, y, label=f"Shape={a:.2f}")

# Customize plot
plt.title("Gamma Distribution with Mean=1 and Varying Shape Parameter", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("Probability Density Function", fontsize=12)
plt.legend(title="Shape Parameter", fontsize=10)
plt.grid(alpha=0.4)
plt.show()
