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

# Diff ev
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
print("Fit Statistics:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared:", r_squared)
plt.scatter(x, y, label='Data', color = 'red')
plt.plot(x_fit, y_fit, label='Fitted Curve')
plt.xlabel('Magnetic Field')
plt.ylabel('Hall Voltage')
plt.title('Curve Fitting with Two-Carrier Model')
plt.legend()
plt.grid(True)
plt.show()
print("Fitted Parameters:")
print("n1 =", fitted_params[0])
print("n2 =", fitted_params[1])
print("u1 =", fitted_params[2])
print("u2 =", fitted_params[3])


if result.success:

    fitted_params = result.x
    fun_min = result.fun
    dof = len(x) - len(fitted_params)  
    param_std = np.sqrt(fun_min / dof) * np.ones_like(fitted_params)
    fitted_data = np.column_stack((x_fit, y_fit))
    np.savetxt('fitted_data_250.txt', fitted_data, header='Magnetic Field (B)  Hall Voltage (R_H)', fmt='%.8f')

else:
    
    fitted_params = np.zeros_like(initial_guess)
    param_std = np.zeros_like(initial_guess)


print("Fitted Parameters:")
for param, std in zip(fitted_params, param_std):
    print(param, "+/-", std)
