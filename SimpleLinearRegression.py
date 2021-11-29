# 1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Veri Onisleme

# 2.1. Veri Yukleme
veriler = pd.read_csv('dataset/satislar.csv')
# pd.read_csv("veriler.csv")


# 2.2.veri on isleme
aylar = veriler[['Aylar']]
# test
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

#2.3.verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

#2.4.verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

#2.5.model inşası (linear regression)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))

plt.title("Aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
plt.show()

















