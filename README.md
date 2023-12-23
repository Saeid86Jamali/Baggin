# Comparing Rotation Forest and Random Forest on Breast Cancer Dataset

This code trains both Rotation Forest and Random Forest models on the breast cancer dataset. It performs a grid search to find the best hyperparameters for each model and then evaluates their performance on a testing set. The results show that the Random Forest model has better performance than the Rotation Forest model.

Step 1: Import libraries

The code starts by importing the following libraries:

RotationForestClassifier: This library is used to load the RotationForestClassifier algorithm.

numpy: This library is used for numerical operations, such as array manipulation and calculations.

sklearn.ensemble: This library is used for ensemble learning algorithms, including Rotation Forest and Random Forest.

sklearn.datasets: This library is used to load datasets, such as the breast cancer dataset.

sklearn.linear_model: This library is used for linear regression models, such as Logistic Regression.

sklearn.model_selection: This library is used for data splitting and model evaluation.

sklearn.preprocessing: This library is used for data pre-processing tasks, such as scaling.

sklearn.metrics: This library is used for performance evaluation metrics, such as accuracy, precision, recall, and F1 score.

Step 2: Load data

The code loads the breast cancer dataset using the load_breast_cancer() function from the sklearn.datasets library. This function returns a dictionary containing the features (X) and target labels (y) of the dataset.

Step 3: Split data into training and testing sets

The code splits the data into training and testing sets using the train_test_split() function from the sklearn.model_selection library. This function randomly divides the data into two sets: a training set (80% of the data) and a testing set (20% of the data). The training set is used to train the models, and the testing set is used to evaluate their performance.

Step 4: Normalize data (Optional)

The code normalizes the data using the MinMaxScaler() function from the sklearn.preprocessing library. This step is optional, but it is often helpful to normalize the data to ensure that all features have a similar scale.

Step 5: Train Rotation Forest model

The code trains a Rotation Forest model using the GridSearchCV() function from the sklearn.model_selection library. The GridSearchCV() function performs a grid search over the hyperparameters of the Rotation Forest model, which in this case are the number of estimators (n_estimators) and the criterion (gini or entropy). The function finds the best combination of hyperparameters that maximizes the performance of the model on the training set.

Step 6: Evaluate Rotation Forest model

The code evaluates the performance of the trained Rotation Forest model on the testing set using the accuracy_score(), precision_score(), recall_score(), f1_score(), and confusion_matrix() functions from the sklearn.metrics library. These functions calculate various metrics that measure the performance of the model, such as accuracy, precision, recall, and F1 score. The confusion matrix shows the distribution of the predicted labels compared to the actual labels.

Step 7: Train Random Forest model

The code trains a Random Forest model using the GridSearchCV() function from the sklearn.model_selection library. The GridSearchCV() function performs a grid search over the hyperparameters of the Random Forest model, which in this case are the number of estimators (n_estimators) and the criterion (gini or entropy). The function finds the best combination of hyperparameters that maximizes the performance of the model on the training set.

Step 8: Evaluate Random Forest model

The code evaluates the performance of the trained Random Forest model on the testing set using the same metrics as the Rotation Forest model.

Step 9: Compare the results

The code compares the performance of the Rotation Forest and Random Forest models by examining the values of the evaluation metrics. In this case, the Random Forest model has better performance than the Rotation Forest model, achieving an accuracy score of 0.947368 (94.74%) compared to the Rotation Forest's 0.953216 (95.32%).
