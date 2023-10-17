import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# Define the two-carrier model function
def two_carrier_model(x, n1, n2, u1, u2):
    e = 1.602e-19  # elementary charge
    B = x  # magnetic field
    
    R_H = ((((n1 * u1**2) - (n2 * u2**2)) * B) + (((n1 - n2) * (u1**2) * (u2**2) * (B**3))))/ (e * (((n1 * u1) + (n2 * u2))**2 + ((n1 - n2)**2 * (u1**2) * (u2**2) * (B**2))))
    
    return R_H

# Read the Hall voltage data from a file
data = np.loadtxt('250KRxy.txt')
x = data[:, 0]  # Magnetic Field values
y = data[:, 1]  # Hall Voltage values

def objective_function(params):
    n1, n2, u1, u2 = params
    y_pred = two_carrier_model(x, n1, n2, u1, u2)
    error = np.sum((y_pred - y)**2)  
    return error

bounds = [(8e20, 5e26), (8e20, 5e26), (1, 1.1), (1, 1.1)]  
result = differential_evolution(objective_function, bounds)
fitted_params = result.x
x_fit = np.linspace(min(x), max(x), 100)
y_fit = two_carrier_model(x_fit, *fitted_params)
y_pred = two_carrier_model(x, *fitted_params)
residuals = y - y_pred
mse = np.mean(residuals**2)
rmse = np.sqrt(mse)
r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
num_runs = 100
fitted_params_runs = []

for _ in range(num_runs):
    result = differential_evolution(objective_function, bounds)
    fitted_params_runs.append(result.x)


fitted_params_runs = np.array(fitted_params_runs)
fitted_params_mean = np.mean(fitted_params_runs, axis=0)
fitted_params_std = np.std(fitted_params_runs, axis=0)
x_fit = np.linspace(min(x), max(x), 100)
y_fit = two_carrier_model(x_fit, *fitted_params_mean)
y_pred = two_carrier_model(x, *fitted_params_mean)
residuals = y - y_pred
mse = np.mean(residuals**2)
rmse = np.sqrt(mse)
r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
print("Fit Statistics:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r_squared)
plt.scatter(x, y, label='Data', color='red')
plt.plot(x_fit, y_fit, label='Fitted Curve')
plt.xlabel('Magnetic Field')
plt.ylabel('Hall Voltage')
plt.title('Curve Fitting with Two-Carrier Model')
plt.legend()
plt.grid(True)
plt.show()
print("Fitted Parameters:")
parameter_names = ['n1', 'n2', 'u1', 'u2']
for param, std, name in zip(fitted_params_mean, fitted_params_std, parameter_names):
    print(name + ":", param, "+/-", std , "(" , std*100 , "%)")