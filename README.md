This program is a complete supervised learning workflow using the Iris dataset Multi class.
It’s a machine learning classifier that takes flower measurements as input and outputs the predicted species, then tells you how well it performed.

1. Dataset Preparation
•	Loads the Iris dataset (classic dataset with 150 samples of iris flowers, each described by 4 features: sepal length, sepal width, petal length, petal width).
•	Adds a target column called species (numeric labels: 0 = setosa, 1 = versicolor, 2 = virginica).
•	Splits the data into features (X) and labels (y).

2. Train/Test Split
•	Divides the dataset into: 
o	Training set (80%) → used to teach the model.
o	Test set (20%) → used to evaluate how well the model generalizes to unseen data.

3. Model Training
•	Creates a Logistic Regression classifier.
•	Fits it on the training data (x_train, y_train).
•	The model learns mathematical relationships between flower measurements and species labels.

4. Prediction
•	Uses the trained model to predict species for the test set (y_pred).
•	Each prediction is one of the three classes (setosa, versicolor, virginica).

5. Evaluation
•	Compares predictions (y_pred) against the true labels (y_test).
•	Prints: 
o	Accuracy → proportion of correct predictions.
o	Classification report → precision, recall, f1-score for each species.

So the program:
•	Learning patterns in flower measurements.
•	Predicting species of iris flowers it hasn’t seen before.
•	Evaluating performance with metrics to show how accurate the predictions are.
