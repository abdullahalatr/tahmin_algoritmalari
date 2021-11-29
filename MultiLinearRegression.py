#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm


#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('dataset/veriler.csv')

print(veriler)

#2.2.encoder: Kategorik -> Numeric
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

ulke = veriler.iloc[:,0:1].values
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#2.3.encoder: Kategorik -> Numeric
cinsiyet = veriler.iloc[:,-1:].values
cinsiyet[:,-1] = le.fit_transform(veriler.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(cinsiyet).toarray()
print(cinsiyet)

#2.4.numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

yas=veriler.iloc[:,1:4]
sonuc2 = pd.DataFrame(data=yas, index = range(22), columns = ['boy','kilo','yas'])
print("sonuc2=\n",sonuc2)

sonuc3 = pd.DataFrame(data = cinsiyet[:,:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)


#2.5.dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

#2.6.verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)
print('x_train =\n',x_train)
print('y_train =\n',y_train)

#2.7.model oluşturma
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)

boy=s2.iloc[:,3:4].values
sol=s2.iloc[:,:3]
sag=s2.iloc[:,4:]

veri=pd.concat([sol,sag],axis=1)
print(veri)
x_train,x_test,y_train,y_test= train_test_split(veri,boy,test_size=0.33, random_state=0)
print("x_train \n",x_train)
print("y_train \n",y_train)

regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print("x_test\n",x_test)
print("y_test\n",y_test)
print("y_pred\n",y_pred)

#2.8.Geri eleme metodu
X=np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)

X_l=veri.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(boy,X_l).fit()
print(model.summary())

X_l=veri.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(boy,X_l).fit()
print(model.summary())

X_l=veri.iloc[:,[0,1,2,3]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(boy,X_l).fit()
print(model.summary())
