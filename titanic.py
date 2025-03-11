import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
#Experiment constants
N_NEIGHBORS = 12
N_ESTIMATORS = 100
TEST_SIZE = 0.2
ITERATIONS = 500

# Setting up data
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "Downloads", "Titanic-Dataset.csv")
df = pd.read_csv(file_path)
def preprocess_data(df):
    df = df.drop(columns=["Cabin", "Ticket", "PassengerId"])
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
    "Mr": 1,
    "Mrs": 6,
    "Miss": 5,
    "Master": 5,
    "Don": 3,
    "Rev": 2,
    "Dr": 2,
    "Mme": 6,
    "Ms": 1,
    "Major": 2,
    "Mlle": 5,
    "Col": 2,
    "Capt": 2,
    "Countess": 6,
    "Jonkheer": 3
    }
    df["Title"] = df["Title"].map(title_map)
    df["Title"] = df["Title"].fillna(4)
    df["Embarked"] = df["Embarked"].map({"C": 3, "Q": 1, "S": 0})
    df["Embarked"] = df["Embarked"].fillna(0)
    df["Pclass"] = df["Pclass"].map({3: 3, 2: 1, 1: 0})
    df = df.drop(columns=["Name"])
    return df
# Linear regression of ages to fill in blanks
def impute_missing_ages_cv(X):
    X = X.copy()
    known_ages = X[X["Age"].notna()]
    unknown_ages = X[X["Age"].isna()]
    
    if not known_ages.empty and not unknown_ages.empty:  # Make sure there is training data
        lin_reg = LinearRegression()
        features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Title"]
        lin_reg.fit(known_ages[features], known_ages["Age"])
        X.loc[X["Age"].isna(), "Age"] = lin_reg.predict(unknown_ages[features])
    
    X["Age"] = X["Age"].clip(lower=0.42)  # Removing overly low ages
    return X

# Convert imputer into a FunctionTransformer to be used in pipeline
imputer_transformer = FunctionTransformer(impute_missing_ages_cv)
df = preprocess_data(df)
x = df.drop(columns=["Survived"])
y = df["Survived"]
# Optional - uncomment for KNN hyperparameter tuning
"""
def impute_missing_ages(train, test, features):
    
    # Split into known and unknown ages
    train_known_ages = train[train["Age"].notna()]
    train_unknown_ages = train[train["Age"].isna()]
    test_unknown_ages = test[test["Age"].isna()]

    # Train Linear Regression model on training set
    lin_reg = LinearRegression()
    lin_reg.fit(train_known_ages[features], train_known_ages["Age"])

    # Predict missing ages
    train.loc[train["Age"].isna(), "Age"] = lin_reg.predict(train_unknown_ages[features])
    test.loc[test["Age"].isna(), "Age"] = lin_reg.predict(test_unknown_ages[features])

    # Ensure ages are not negative or unrealistically low
    train["Age"] = train["Age"].clip(lower=0.42)
    test["Age"] = test["Age"].clip(lower=0.42)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=TEST_SIZE, stratify=y)
#Impute missing ages
impute_missing_ages(X_train, X_test, ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Title"])

# Apply StandardScaler
features = ["Age", "SibSp", "Parch", "Fare"]
scaler = StandardScaler()
X_train[features] = scaler.fit_transform(X_train[features])
X_test[features] = scaler.transform(X_test[features])

train_accuracies = []
test_accuracies = []
n = range(1, 21)
for i in n:
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, Y_train)
    train_accuracies.append(accuracy_score(Y_train, knn.predict(X_train)))
    test_accuracies.append(accuracy_score(Y_test, knn.predict(X_test)))

fig, ax = plt.subplots()             
plt.xlabel("Number of Neighbors (k)", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title("KNN Accuracy vs. Number of Neighbors", fontsize=16)
plt.xticks(n)  # Ensure all k-values are visible
plt.grid(True, linestyle="--", alpha=0.6)  # Light grid for better readability
plt.plot(n, train_accuracies, marker='o', linestyle='-', label="Train Accuracy", color="blue")
plt.plot(n, test_accuracies, marker='o', linestyle='-', label="Test Accuracy", color="red")
plt.show()                           
print(X_train.describe())
"""
# Convert to FunctionTransformer for pipeline usage
imputer_transformer = FunctionTransformer(impute_missing_ages_cv)

# Add models to pipeline for cross-validation
models = {
    "Logistic Regression": Pipeline([
        ("imputer", imputer_transformer),  
        ("scaler", StandardScaler()),      
        ("classifier", LogisticRegression(max_iter=1000))
    ]),
    "KNN": Pipeline([
        ("imputer", imputer_transformer),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=12))
    ]),
    "Random Forest": Pipeline([
        ("imputer", imputer_transformer),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

# Cross-validation:
for name, model in models.items():
    scores = cross_val_score(model, x, y, cv=50)  
    print(f"{name} Averaged Accuracy: {scores.mean():.4f}")

for _, model in models.items(): # Re-fitting models to extract feature importance
    model.fit(x, y)
# Extract feature importance from all models
for name, pipeline in models.items():
    classifier = pipeline.named_steps["classifier"]  # Extract the classifier from the pipeline

    if hasattr(classifier, "feature_importances_"):  # For the models with feature importance:
        importances = classifier.feature_importances_
        feature_names = x.columns  

        sorted_idx = np.argsort(importances)[::-1]
        
        print(f"\nFeature Importances for {name}:")
        for i in sorted_idx:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
    else:
        print(f"\n{name} does not support feature importance.") # Logistic Regression and KNN currently do not support feature importance.