import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

df=pd.read_csv('wdbc.data.txt', header= 0)
col_names= ['id_number', 'diagnosis', 'radius_mean', 
         'texture_mean', 'perimeter_mean', 'area_mean', 
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave_points_mean', 'symmetry_mean', 
         'fractal_dimension_mean', 'radius_se', 'texture_se', 
         'perimeter_se', 'area_se', 'smoothness_se', 
         'compactness_se', 'concavity_se', 'concave_points_se', 
         'symmetry_se', 'fractal_dimension_se', 
         'radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst', 
         'compactness_worst', 'concavity_worst', 
         'concave_points_worst', 'symmetry_worst', 
         'fractal_dimension_worst'] 
df.columns= col_names

df.set_index(['id_number'], inplace= True)
print(df.head())
print(df.columns)
print(df.describe())

sns.countplot(df['diagnosis'],palette="Blues")
plt.plot()
mean1=df[['radius_mean', 
         'texture_mean', 'perimeter_mean', 'area_mean', 
         'smoothness_mean', 'compactness_mean', 'concavity_mean',
         'concave_points_mean', 'symmetry_mean', 
         'fractal_dimension_mean']]
se1=df[['radius_se', 'texture_se', 
         'perimeter_se', 'area_se', 'smoothness_se', 
         'compactness_se', 'concavity_se', 'concave_points_se', 
         'symmetry_se', 'fractal_dimension_se']]
worst1=df[['radius_worst', 'texture_worst', 'perimeter_worst',
         'area_worst', 'smoothness_worst', 
         'compactness_worst', 'concavity_worst', 
         'concave_points_worst', 'symmetry_worst', 
         'fractal_dimension_worst']]
'''
plt.figure(figsize=(10,10))
sns.heatmap(mean1.corr(),annot=True,cmap='Oranges')
plt.show()
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.swarmplot(x='diagnosis',y='radius_mean',data=df)
plt.subplot(2,2,2)
sns.swarmplot(x='diagnosis',y='area_mean',data=df)
plt.subplot(2,2,3)
sns.swarmplot(x='diagnosis',y='perimeter_mean',data=df)
plt.show()
'''
df.drop(['area_mean','perimeter_mean'],axis=1,inplace=True)
'''
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.violinplot(x='diagnosis',y='compactness_mean',palette="Oranges",inner='quartile',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='diagnosis',y='concavity_mean',palette="Oranges",inner="quartile",data=df)
plt.subplot(2,2,3)
sns.violinplot(x='diagnosis',y='concave_points_mean',palette="Oranges",inner="quartile",data=df)
plt.show()
'''
df.drop(['compactness_mean','concavity_mean'],axis=1,inplace=True)
'''
df1=df[['texture_mean','diagnosis','symmetry_mean', 
         'fractal_dimension_mean','smoothness_mean']]
sns.heatmap(df1.corr(),annot=True)
plt.show()
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.swarmplot(x='diagnosis',y='symmetry_mean',data=df)
plt.subplot(2,2,2)
sns.swarmplot(x='diagnosis',y='fractal_dimension_mean',data=df)
plt.subplot(2,2,3)
sns.swarmplot(x='diagnosis',y='smoothness_mean',data=df)
plt.show()
'''
df.drop(['fractal_dimension_mean'],axis=1,inplace=True)
'''
print(df.columns)


plt.figure(figsize=(10,10))
sns.heatmap(se1.corr(),annot=True,cmap='Blues')
plt.plot()

plt.subplot(2,2,1)
sns.swarmplot(x='diagnosis',y='radius_se',data=df)
plt.subplot(2,2,2)
sns.swarmplot(x='diagnosis',y='area_se',data=df)
plt.subplot(2,2,3)
sns.swarmplot(x='diagnosis',y='perimeter_se',data=df)
plt.show()

plt.subplot(2,2,1)
sns.violinplot(x='diagnosis',y='radius_se',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='diagnosis',y='area_se',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='diagnosis',y='perimeter_se',data=df)
plt.show()
'''
df.drop(['perimeter_se','radius_se'],axis=1,inplace=True)
'''
plt.subplot(2,2,1)
sns.violinplot(x='diagnosis',y='compactness_se',palette="Blues",inner='quartile',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='diagnosis',y='concavity_se',palette="Blues",inner="quartile",data=df)
plt.subplot(2,2,3)
sns.violinplot(x='diagnosis',y='concave_points_se',palette="Blues",inner="quartile",data=df)
plt.show()
'''
df.drop(['compactness_se','concavity_se'],axis=1,inplace=True)
'''
df1=df[['texture_se','diagnosis','symmetry_se', 
         'fractal_dimension_se','smoothness_se']]
sns.heatmap(df1.corr(),annot=True)
plt.plot()
plt.subplot(2,2,1)
sns.swarmplot(x='diagnosis',y='symmetry_se',data=df)
plt.subplot(2,2,2)
sns.swarmplot(x='diagnosis',y='fractal_dimension_se',data=df)
plt.subplot(2,2,3)
sns.swarmplot(x='diagnosis',y='smoothness_se',data=df)
plt.show()

plt.subplot(2,2,1)
sns.boxplot(x='diagnosis',y='symmetry_se',data=df)
plt.subplot(2,2,2)
sns.boxplot(x='diagnosis',y='fractal_dimension_se',data=df)
plt.subplot(2,2,3)
sns.boxplot(x='diagnosis',y='smoothness_se',data=df)
plt.show()
'''
df.drop(['smoothness_se','symmetry_se'],axis=1,inplace=True)
'''
print(df.columns)

plt.figure(figsize=(10,10))
sns.heatmap(worst1.corr(),annot=True,cmap='YlOrBr')
plt.plot()

plt.subplot(2,2,1)
sns.swarmplot(x='diagnosis',y='radius_worst',data=df)
plt.subplot(2,2,2)
sns.swarmplot(x='diagnosis',y='area_worst',data=df)
plt.subplot(2,2,3)
sns.swarmplot(x='diagnosis',y='perimeter_worst',data=df)
plt.show()

'''
df.drop(['area_worst','radius_worst'],axis=1,inplace=True)
'''
plt.subplot(2,2,1)
sns.violinplot(x='diagnosis',y='compactness_worst',palette="YlOrBr",inner='quartile',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='diagnosis',y='concavity_worst',palette="YlOrBr",inner="quartile",data=df)
plt.subplot(2,2,3)
sns.violinplot(x='diagnosis',y='concave_points_worst',palette="YlOrBr",inner="quartile",data=df)
plt.show()
'''
print(df.columns)
df.drop(['compactness_worst','concavity_worst'],axis=1,inplace=True)
'''
df1=df[['texture_worst','diagnosis','symmetry_worst', 'fractal_dimension_worst','smoothness_worst']]
sns.heatmap(df1.corr(),annot=True)
plt.show()
print(df.columns)
'''
X= np.array(df.drop(['diagnosis'], axis=1))
y= np.array(df['diagnosis'])
x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state= 14)
'''
dt= DecisionTreeClassifier(random_state= 14)
tuned_params= [{'max_depth': [i+5 for i in range(5, 20) if i%5==0]}]
gs= GridSearchCV(estimator= dt, param_grid= tuned_params, cv= 10)
gs= gs.fit(x_train, y_train)
y_pred= gs.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(gs.best_score_)
print(gs.best_estimator_.max_depth)
'''
#10
'''
dt.fit(x_train,y_train)
y_pred= dt.predict(x_test)
print(accuracy_score(y_test,y_pred))
'''
'''
rf= RandomForestClassifier(n_estimators= 100, random_state= 14)
ti=[{'max_depth':[i+5 for i in range(5,31) if i%5==0]}]
gs= GridSearchCV(estimator= rf, param_grid= ti, cv= 10)
gs= gs.fit(x_train, y_train)
y_pred= gs.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(gs.best_score_)
print(gs.best_estimator_.max_depth)
'''
#10
'''
rf.fit(x_train,y_train)
y_pred= rf.predict(x_test)
print(accuracy_score(y_test,y_pred))
'''

sv= SVC(random_state= 14)
'''
tuned_params= [{'C': [0.1,1,10,100],'gamma':[0,0.1,1,10,100]}]
gs= GridSearchCV(estimator= sv, param_grid= tuned_params, cv= 10)
gs= gs.fit(x_train, y_train)
y_pred= gs.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(gs.best_score_)
print(gs.best_estimator_.gamma)
print(gs.best_estimator_.C)
'''
'''
sv.fit(x_train,y_train)
y_pred= sv.predict(x_test)
print(accuracy_score(y_test,y_pred))
'''
knn= KNeighborsClassifier()
nm=[{'n_neighbors':[i for i in range(1,26) ]}]
g=GridSearchCV(estimator=knn,param_grid=nm, cv=5)
g= g.fit(x_train, y_train)
y_pred= g.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(g.best_score_)
print(g.best_estimator_.n_neighbors)
#8
'''
knn.fit(x_train,y_train)
y_pred= knn.predict(x_test)
print(accuracy_score(y_test,y_pred))
'''