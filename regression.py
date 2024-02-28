import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
datalar=pd.read_csv("C:/Users/Kerem/Desktop/Regresson/reklam.csv")

print(datalar)
veri=datalar.copy() 


print(veri.isnull().sum()) #eksik gözlem değerleri kontrol edilir



sns.pairplot(veri,kind="reg") #korelasyon grafik yapısını gösterir
plt.show()     

#boxplot ile aykırı gözlem değerlerine bakılır
sns.boxplot(veri["TV"]) 
sns.boxplot(veri["Radio"])
sns.boxplot(veri["Newspaper"])
plt.show()


#newspaper üzerindeki aykırı gözlem değerlerlerini baskılama yöntemi ile aşağıya doğru çekilir
q1=veri["Newspaper"].quantile(0.25)
q3=veri["Newspaper"].quantile(0.75)
iqr=q3-q1
ustsinir=q3+1.5*iqr
aykiri=veri["Newspaper"]>ustsinir
veri.loc[aykiri,["Newspaper"]]=ustsinir
sns.boxplot(veri["Newspaper"])
plt.show()

y=veri["Sales"] 
x=veri[["Newspaper","TV","Radio"]] 


sabit=sm.add_constant(x)
model=sm.OLS(y,sabit).fit()
print(model.summary()) #model hakkında genel bilgiye sahip olmak için kullanılır


y=veri["Sales"]
x=veri[["TV","Radio"]] 


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) #x ve y için eğitim ve test yapısı oluşturcaz, yüzde 20 lik bir test yapısı ayırdık
print(x_train)

lr=LinearRegression()

lr.fit(x_train,y_train) 
print(lr.coef_)   

tahmin=lr.predict(x_test) 
print(tahmin)

y_test=y_test.sort_index()

df = pd.DataFrame({"gerçek": y_test, "tahmin": tahmin})
df.plot(kind="line")
plt.show()










