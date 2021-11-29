import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Veri Yükleme
veriler = pd.read_csv('dataset/maaslar.csv')

# data frame dilimleme (slice)
x = veriler.iloc[:, 1:2]
y = veriler.iloc[:, 2:]

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

#Görselleştirme.
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


from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_scaler=sc1.fit_transform(x)
sc2=StandardScaler()
y_scaler=sc2.fit_transform(y)

from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_scaler,y_scaler)
plt.scatter(x_scaler,y_scaler,color='red')
plt.plot(x_scaler,svr_reg.predict(x_scaler),color='black')
plt.show()

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))
