# %%
import pandas as pd 
import matplotlib.pyplot as plt


# %%
df=pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\CSUS\CSC215\CSC215_Assg5_Ayush\IRIS.csv')
df

# %%
df['species'].unique()

# %%
df['species'].replace({'Iris-setosa':'0','Iris-versicolor':'1','Iris-virginica':'2'}, inplace=True)

# %%
df

# %%
data = df[['sepal_length', 'sepal_width','petal_length', 'petal_width']]
data

# %%
pred_y = df['species']

# %%
from sklearn.model_selection import train_test_split
main_data_train, main_data_test, pred_data_train, pred_data_test = train_test_split(data, pred_y, test_size=0.2)
from sklearn.linear_model import LogisticRegression

# %%
model = LogisticRegression()

# %%
model.fit(main_data_train, pred_data_train)

# %%
modelPrediction= model.predict(main_data_test)


# %%
model.score(main_data_test,pred_data_test)

# %%
model.score(main_data_train,pred_data_train)

# %%
specie = model.predict([[6.0,4.0,5.0,4.0]])
specie

# %%
from sklearn.metrics import confusion_matrix
#Creating Confusion Matrix for the model
conf_matrix= confusion_matrix(pred_data_test,modelPrediction)
conf_matrix

# %%
#printing Confusion Metrix
import seaborn as sn

plt.figure(figsize=(5,2))
sn.heatmap(conf_matrix, annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

# %%



