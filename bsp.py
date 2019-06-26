# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 22:13:45 2019

@author: mishr
"""

# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization


# import dataset from sklearn datasets
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

# Visualizing dataset
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'] )
sns.countplot(df_cancer['target'], label = "Count") 
# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True) 

#Importing Dataset to X and y 
X = df_cancer.iloc[:, :-1].values
y = df_cancer.iloc[:, 30].values

#Spliting Dataset to Test and Train Sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 5)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#fitting Classifier to the dataset
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',)
classifier.fit(X_train, y_train)

#predicting test Set
y_pred = classifier.predict(X_test)

#making confusion matrix to get no. of correct or incorrect value
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, y_pred))

