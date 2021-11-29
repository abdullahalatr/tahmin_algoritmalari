import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Karar Ağacı
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)
Z=X+0.5
K=X-0.5
plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='black')
plt.show()

plt.plot(x,r_dt.predict(Z),color='orange')
plt.plot(x,r_dt.predict(K),color='purple')
plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))