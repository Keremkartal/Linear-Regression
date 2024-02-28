import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=sns.load_dataset("tips")
veri=data.copy()
print(veri.dtypes)
katagori=[]
kategorik=veri.select_dtypes(include=["category"]) 
for i in kategorik.columns:
    katagori.append(i) 
veri=pd.get_dummies(veri,columns=katagori,drop_first=True)
veri= veri.replace({False: 0, True: 1})

y=veri["tip"]
x=veri.drop(columns="tip",axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) 
lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)

y_test=y_test.sort_index() 
df=pd.DataFrame({"gercek:":y_test,"tahmin:":tahmin})
df.plot(kind="line")
plt.show()
