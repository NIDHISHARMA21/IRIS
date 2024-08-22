# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:11:54 2024

@author: nidhi
"""

# 1. Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import pickle

# 2. Load and Explore the Dataset
file_path = r'E:\2_iris_project\IRIS.csv'
iris_df = pd.read_csv(file_path)

# Check the first few rows and basic statistics
print(iris_df.head(), "\n", iris_df.describe(), "\n", iris_df.isnull().sum())

# 3. Exploratory Data Analysis (EDA)
sns.pairplot(iris_df, hue='species')
plt.show()

sns.heatmap(iris_df.corr(), annot=True, cmap='coolwarm')
plt.show()

# 4. Data Preprocessing
X = iris_df.drop('species', axis=1)
y = iris_df['species']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 5. Model Building and Evaluation (SVM)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 6. Model Optimization (GridSearchCV)
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, verbose=1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# 7. Model Comparison (Random Forest)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")

# 8. Dimensionality Reduction (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA of Iris Dataset')
plt.show()

# 9. Save Models with Pickle
with open('svm_iris_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Test the loaded model
with open('svm_iris_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

y_pred_loaded = loaded_model.predict(X_test)
print(f"Accuracy of loaded model: {accuracy_score(y_test, y_pred_loaded):.2f}")

# Confusion Matrix Visualization
cm_svm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=label_encoder.classes_).plot(cmap='Blues')
plt.title("Confusion Matrix - SVM")
plt.show()

cm_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=label_encoder.classes_).plot(cmap='Greens')
plt.title("Confusion Matrix - Random Forest")
plt.show()
