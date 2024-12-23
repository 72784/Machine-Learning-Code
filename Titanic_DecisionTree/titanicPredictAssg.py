# %%
import pandas as pd 
import matplotlib.pyplot as plt

# %%
df=pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\CSUS\CSC215\CSC215_Assg6_Ayush\titanic.csv')
df

# %%
#checking null values
print(df.isnull().sum()
)

# %%
data = df[['Pclass', 'Sex','Age', 'Fare','Survived']]
data

# %%
median_age= data.Age.median()
data.Age= data.Age.fillna(median_age)

# %%
data

# %%
print(data.isnull().sum()
)

# %%
data.Sex.replace({'male':'0','female':'1'},inplace=True)
data

# %%
pred_y= data['Survived']

# %%
from sklearn.model_selection import train_test_split
data_train, data_test, pred_train, pred_test = train_test_split(data[['Pclass','Sex','Age','Fare']], pred_y, test_size=0.3)


# %%
from sklearn import tree

# %%
model= tree.DecisionTreeClassifier()

# %%
model.fit(data_train,pred_train)

# %%
survived_Prediction = model.predict(data_test)

# %%
survived_Prediction

# %%
model.score(data_train,pred_train)

# %%
model.score(data_test,pred_test)

# %%
survived = model.predict([['3','1','40','9.222']])
survived

# %%
survived = model.predict([['1','1','25','7.222']])
survived

# %%



