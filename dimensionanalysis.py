import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

url = "https://raw.githubusercontent.com/nakshatra-nitt/Nakshatra-Inductions-2025/main/experimental_Planck_Temp_data.csv"
data = pd.read_csv(url)

log_h = np.log(data['h'])
log_c = np.log(data['c'])
log_G = np.log(data['G'])
log_kB = np.log(data['k_B'])  
log_TP = np.log(data['T_P']) 

# Prepare the feature matrix (log of independent variables)
X = np.column_stack((log_h, log_c, log_G, log_kB))

# Target variable (log of dependent variable)
y = log_TP

# linear regression
model = LinearRegression()
model.fit(X, y)

# coefficients and intercept
w, x, y, z = model.coef_
intercept = model.intercept_


print("Estimated exponents:")
print(f"w (h exponent): {w:.4f}")
print(f"x (c exponent): {x:.4f}")
print(f"y (G exponent): {y:.4f}")
print(f"z (k_B exponent): {z:.4f}")
print(f"\nIntercept (log of proportionality constant): {intercept:.4f}")

print("\nValues from dimensional analysis")
print("w = 0.5, x = 2.5, y = -0.5, z = -1")