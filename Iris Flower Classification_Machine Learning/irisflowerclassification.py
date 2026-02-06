#Iris flower Classification

##loads iris dataset
##Splits into train/test sets.
##Trains a Logistic Regression classifier.
##Predicts species on unseen test data.
##Prints accuracy and classification report.


import pandas as pd #to handle tabular data
from sklearn.datasets import load_iris #iris dataset
from sklearn.model_selection import train_test_split #splits data into training and testing sets
from sklearn.linear_model import LogisticRegression #Simple Classification Model
from sklearn.metrics import accuracy_score, classification_report #Metrics to evaluate classification performance

#loads iris dataset
iris=load_iris()

#Splits into train/test sets.
data=pd.DataFrame(iris.data, columns=iris.feature_names)
data['species']=iris.target


x=data.drop('species',axis=1)
y=data['species']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=LogisticRegression(max_iter=200)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))




