import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


df = pd.read_csv("data.csv")

def col_data():
    x = df['Chieucao(cm)']
    y = df['Cannang(kg)']
    return x, y

x_raw, y_raw = col_data()


m = len(x_raw)

#chuan hoa du lieu
x_norm = (x_raw - x_raw.min())/(x_raw.max() - x_raw.min())
y_norm = (y_raw - y_raw.min())/(y_raw.max() - y_raw.min())


#thiet lap ma tran X
one_col = np.ones((m,1))
X = np.c_[one_col, x_norm, x_norm**2]


y_norm = y_norm.to_numpy().reshape(m,1)


#thiet lap cac tham so
theta = np.array([[0], [0], [0]])


learing_rate = 0.1
eps = 1e-4


#thuat toan gradient descent
while True:
    
    grad = (1/m) * np.dot(X.T, (np.dot(X, theta) - y_norm))

    if np.all(np.abs(grad) < eps):
        break  

    theta = theta - learing_rate * grad


#thuat toan nghien cuu phuong phap tinh toan truc tiep
theta = (np.linalg.pinv(X.T @ X) @ (X.T @ y_norm))

print(f"He so hoi quy: theta = {theta}")


plt.figure(figsize=(10, 6))
plt.plot(x_raw , y_raw, "o", label = "Du lieu thuc te", alpha=0.7, markersize=5) 


Y_pred_norm = np.dot(X, theta)

plt.plot(x_raw, Y_pred_norm * (y_raw.max() - y_raw.min()) + y_raw.min(), "r-", label = "Duong hoi quy", linewidth=2)

plt.xlabel("Chiều cao (cm)")
plt.ylabel("Cân nặng (kg)")
plt.title("Du lieu va duong hoi quy")

plt.grid(True)

plt.legend()
plt.show()


