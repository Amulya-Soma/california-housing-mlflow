# -*- coding: utf-8 -*-

# # src/train.py
# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score


# def load_data(path="data/iris.csv"):
#     return pd.read_csv(path)


# def train_and_log(model, model_name, X_train, X_test, y_train, y_test):
#     with mlflow.start_run(run_name=model_name):
#         # Train model
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         acc = accuracy_score(y_test, preds)

#         # Log parameters, metrics
#         if hasattr(model, "C"):
#             mlflow.log_param("C", model.C)
#         if hasattr(model, "n_estimators"):
#             mlflow.log_param("n_estimators", model.n_estimators)

#         mlflow.log_metric("accuracy", acc)

#         # Log model
#         # mlflow.sklearn.log_model(model, model_name)
#         mlflow.sklearn.log_model(model, name=model_name)

#         print(f"{model_name} Accuracy: {acc:.4f}")

#         return acc


# if __name__ == "__main__":
#     # Load data
#     df = load_data()
#     X = df.drop(columns=["target"])
#     y = df["target"]

#     # Train/test split
#     # X_train, X_test, y_train, y_test = train_test_split(
#     #     X, y, test_size=0.2, random_state=42
#     # )
#     X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, stratify=y, random_state=42
#     )


#     # Set MLflow experiment
#     mlflow.set_experiment("iris-classification")

#     # Train Logistic Regression
#     lr_acc = train_and_log(LogisticRegression(max_iter=200), "LogisticRegression",
#                            X_train, X_test, y_train, y_test)

#     # Train Random Forest
#     rf_acc = train_and_log(RandomForestClassifier(n_estimators=100), "RandomForest",
#                            X_train, X_test, y_train, y_test)

#     # Print best model
#     if rf_acc > lr_acc:
#         print("RandomForest is the best model ✅")
#     else:
#         print("LogisticRegression is the best model ✅")



# import os
# import mlflow
# import mlflow.sklearn
# import pandas as pd
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # -------------------------
# # 1. Load data
# # -------------------------
# housing = fetch_california_housing(as_frame=True)
# X = housing.data
# y = housing.target
# feature_names = X.columns

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # -------------------------
# # 2. Experiment Setup
# # -------------------------
# mlflow.set_experiment("california-housing")

# # -------------------------
# # 3. Models to train
# # -------------------------
# models = {
#     "LinearRegression": LinearRegression(),
#     "DecisionTree": DecisionTreeRegressor(random_state=42),
#     "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
#     "GradientBoosting": GradientBoostingRegressor(random_state=42)
# }

# # Scaling for models that benefit from it
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # -------------------------
# # 4. Train and log each model
# # -------------------------
# best_model = None
# best_score = -float("inf")

# for name, model in models.items():
#     with mlflow.start_run(run_name=name):
        
#         # Scale only for LinearRegression
#         if name == "LinearRegression":
#             model.fit(X_train_scaled, y_train)
#             y_pred = model.predict(X_test_scaled)
#         else:
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
        
#         # Metrics
#         rmse = mean_squared_error(y_test, y_pred, squared=False)  # no warning
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
        
#         # Log metrics
#         mlflow.log_metric("rmse", rmse)
#         mlflow.log_metric("mae", mae)
#         mlflow.log_metric("r2", r2)
        
#         # Log model with input example to avoid warnings
#         input_example = pd.DataFrame([X_test.iloc[0]], columns=feature_names)
#         mlflow.sklearn.log_model(model, name="model", input_example=input_example)
        
#         print(f"{name} → RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
#         # Track best model
#         if r2 > best_score:
#             best_score = r2
#             best_model = name

# print(f"\n✅ Best model: {best_model} with R² = {best_score:.4f}")


import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.data import from_pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

# ==== Load Dataset ====
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# ==== Split Data ====
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== Start MLflow Experiment ====
mlflow.set_experiment("california-housing")

# ==== Log Dataset in a Named Run ====
with mlflow.start_run(run_name="Dataset-Logging"):
    dataset = from_pandas(X_train, source="California Housing", name="california-housing")
    mlflow.log_input(dataset, context="training")

# ==== Models to Try ====
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model_name = None
best_r2 = -1
best_model = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate R²
        r2 = r2_score(y_test, y_pred)

        # Log parameters & metrics
        mlflow.log_param("model_name", name)
        mlflow.log_metric("r2_score", r2)

        # Log model with input example
        input_example = X_train.iloc[:1]
        mlflow.sklearn.log_model(model, name, input_example=input_example)

        print(f"{name} R²: {r2:.4f}")

        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model

print(f"✅ Best model: {best_model_name} with R² = {best_r2:.4f}")
