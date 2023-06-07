# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load the training data
df_train = pd.read_csv('/content/drive/MyDrive/ML/ML/TrainingDataBinary.csv', header=None)

# 2. Data preprocessing
X_train = df_train.iloc[:, :-1].values  # Extract the features
y_train = df_train.iloc[:, -1].values  # Extract the labels

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Choose and train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# 5. Load unlabeled test data
df_test_orig = pd.read_csv('/content/drive/MyDrive/ML/ML/TestingDataBinary.csv', header=None)
X_test = df_test_orig.values  # As the test set has no labels, take all the data as features

# 6. Feature scaling
X_test_scaled = scaler.transform(X_test)

# 7. Make predictions on the test data
y_test_pred = clf.predict(X_test_scaled)

# 8. Append the predicted results to the original test dataset and save it to a new csv file
df_test_orig['Label'] = y_test_pred
df_test_orig.to_csv('TestingResultsBinary.csv', index=False)
