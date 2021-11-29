import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# Veri Yükleme
veriler = pd.read_csv('dataset/maaslar.csv')

# data frame dilimleme (slice)
x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]
X = x.values
Y = y.values

# Linear Regression
# doğrusal model oluşturma.
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x, y)

# polynomial regression
# Doğrusal olmayan model oluşturma.
from sklearn.preprocessing import PolynomialFeatures

poly_reg2 = PolynomialFeatures(degree=2)
x_poly = poly_reg2.fit_transform(x)
lr2 = LinearRegression()
lr2.fit(x_poly, y)

# 10. dereceden polinom
poly_reg3 = PolynomialFeatures(degree=10)
x_poly = poly_reg3.fit_transform(x)
lr3 = LinearRegression()
lr3.fit(x_poly, y)

# Görselleştirme.
# linear regresion görselleştirmesi.
plt.scatter(x, y, color='red')
plt.plot(x, lr.predict(x), color='black')
plt.show()

# 2.dereceden polinomal regresyon görselleştirmesi.
plt.scatter(x, y, color='red')
plt.plot(x, lr2.predict(poly_reg2.fit_transform(x)), color='black')
plt.show()

# 10.dereceden polinomal regresyon görselleştirmesi.
plt.scatter(x, y, color='red')
plt.plot(x, lr3.predict(poly_reg3.fit_transform(x)), color='black')
plt.show()


# Ölçeklendirme.
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_scaler = sc1.fit_transform(x)
sc2 = StandardScaler()
y_scaler = sc2.fit_transform(y)

# Destek Vector
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scaler, y_scaler)
plt.scatter(x_scaler, y_scaler, color='red')
plt.plot(x_scaler, svr_reg.predict(x_scaler), color='black')
plt.show()




# Decision Tree Regression (Her değişken için aynı aralığı verir.)
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X, Y)
Z = X + 0.5
K = X - 0.4
plt.scatter(X, Y, color='red')
plt.plot(X, dt_reg.predict(X), color='black')
plt.show()

plt.plot(x, dt_reg.predict(Z), color='orange')
plt.plot(x, dt_reg.predict(K), color='purple')
plt.show()



print(dt_reg.predict([[11]]))
print(dt_reg.predict([[6.6]]))

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X, Y.ravel())
print(rf_reg.predict([[6.6]]))

plt.scatter(X, Y, color='red')
plt.plot(X, rf_reg.predict(X), color='black')
plt.plot(X, rf_reg.predict(Z), color='gray')
plt.plot(x, dt_reg.predict(K), color='orange')
plt.show()


print("----------------------------------------")
print("Linear Regression  R-Square değeri")
print(r2_score(Y, lr.predict(X)))
print(("\n"))
print("Polynomial Regression R-Square değeri")
print(r2_score(Y, lr2.predict(poly_reg2.fit_transform(X))))
print("\n")
print("Support Vector Regression R-Square değeri")
print(r2_score(y_scaler, svr_reg.predict(x_scaler)))
print("\n")
print("Decision Tree R-Square değeri")
print(r2_score(Y, dt_reg.predict(X)))
print("\n")
print("Random Forest R-Square değeri")
print(r2_score(Y, rf_reg.predict(X)))



