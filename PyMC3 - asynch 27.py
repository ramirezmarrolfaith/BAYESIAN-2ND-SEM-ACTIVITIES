import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data for a linear relationship
np.random.seed(42)  # Set seed for reproducibility

# Generate x values from 0 to 10
x = np.linspace(0, 10, 100)

# True parameters for the linear relationship
true_slope = 1.5
true_intercept = 2.0

# Generate y values with a linear relationship (y = true_slope * x + true_intercept + noise)
noise = np.random.normal(loc=0, scale=1.5, size=len(x))
y = true_slope * x + true_intercept + noise

# Calculate the regression coefficients using numpy
X = np.vstack([np.ones_like(x), x]).T
coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

# Extract slope (beta_1) and intercept (beta_0)
beta_0, beta_1 = coefficients

# Create the scatter plot and plot the regression line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.7, edgecolors='w', label='Data Points')
plt.plot(x, beta_0 + beta_1 * x, color='red', label='Regression Line')

# Add labels and title
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Linear Regression Model')

# Show legend
plt.legend()

plt.grid(True)
plt.show()

# Print the regression coefficients
print(f"Slope (beta_1): {beta_1:.2f}")
print(f"Intercept (beta_0): {beta_0:.2f}")
