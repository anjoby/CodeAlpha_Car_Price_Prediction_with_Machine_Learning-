import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv(r"C:\Users\anjoj\Downloads\car data.csv")
print("First 5 rows of dataset:")
print(data.head())

# 2. Basic Info
print("\nDataset Info:")
print(data.info())

# 3. Handle categorical features
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# 4. Define features (X) and target (y)
X = data.drop("Selling_Price", axis=1)  # assuming 'Selling_Price' is target
y = data["Selling_Price"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\nLinear Regression Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

# 8. Train Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Performance:")
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))

# 9. Plot Actual vs Predicted
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred_rf, color='blue', label='Predicted vs Actual')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Car Price Prediction (Random Forest)")
plt.legend()
plt.show()

# 10. Feature Importance from Random Forest
feat_importances = pd.Series(rf_model.feature_importances_, index=data.drop("Selling_Price", axis=1).columns)
plt.figure(figsize=(10,5))
feat_importances.sort_values().plot(kind='barh')
plt.title("Feature Importance in Car Price Prediction")
plt.show()
