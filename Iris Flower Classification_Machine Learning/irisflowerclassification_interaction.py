#Classification Model - Labelled data with input and output given [Check the dataset]

import pandas as pd
#from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
data=pd.read_csv("dataset/Iris.csv")

#Drop the Id column, Species column during training
#This ensures the model is trained only on the 4 numeric measurement features.
X = data.drop(['Id','Species'], axis=1)#axis=0 → rows, axis=1 → Columns  (tells Pandas: drop these labels from the columns.)
y = data['Species']  #extracts the labels (the thing you’re predicting) from your dataset.

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#X → your features (sepal length, sepal width, petal length, petal width).
#y → your labels (species of the flower).
#test_size=0.2 → 20% of the data goes into the test set, 80% into the training set.
#random_state=42 → ensures reproducibility. If run the code again, we'll get the same split every time.


# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Custom Prediction Section
print("\nEnter custom flower measurements:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width  = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width  = float(input("Petal width (cm): "))

# Create input array
sample = pd.DataFrame( [[sepal_length, sepal_width, petal_length, petal_width]], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

prediction = model.predict(sample)[0]
print(f"\nPredicted species: {prediction}")





#Note:


##X → your features (sepal length, sepal width, petal length, petal width).

##y → your labels (species of the flower).

##test_size=0.2 → 20% of the data goes into the test set, 80% into the training set.

##random_state=42 → ensures reproducibility. If you run the code again, you’ll get the same split every time.


##Suppose the dataset has 150 rows (Iris dataset size):

##Training set → 120 rows (80%)

##Testing set → 30 rows (20%)

##So:

##X_train → 120 rows of features

##y_train → 120 species labels

##X_test → 30 rows of features

##y_test → 30 species labels



##LogisticRegression → A classification algorithm that models probabilities using the logistic (sigmoid) function. It’s widely used for binary and multiclass classification.

##Default solver → 'lbfgs' (a quasi‑Newton optimization method).

##Regularization → By default, L2 regularization is applied to prevent overfitting.



##Binary classification is simpler: the model decides between two outcomes.

##Multiclass classification requires the model to handle multiple categories, often using strategies like one-vs-rest (train one classifier per class) or softmax (predict probabilities across all classes).



##The max_iter Parameter:

##Definition: Maximum number of iterations the solver will run to find the optimal weights.

##Default value: 100 iterations.

##Why increase it?

##On small, clean datasets (like Iris), 100 iterations is usually enough.

##On larger or more complex datasets, the solver may not converge within 100 iterations, leading to a ConvergenceWarning.

##Setting max_iter=200 gives the solver more time to converge.



##Key Takeaways:
##--------------

##max_iter=200 ensures the optimization process has enough steps to converge.

##If you still see convergence warnings, you can increase it further (e.g., 500 or 1000).

##Too high a value doesn’t harm accuracy but may slow training slightly.

##If convergence is slow, you can also try different solvers ('saga', 'liblinear', 'newton-cg') depending on dataset size and type.


