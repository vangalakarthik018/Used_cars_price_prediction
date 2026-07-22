import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("Loading Dataset...")
df = pd.read_csv("used_cars.csv")
print(df.head())

print("\n Missing Values")
print(df.isnull().sum())
df = df.dropna()

categorical_columns = [
    "Brand",
    "Fuel",
    "Transmission",
    "Owner"
]

label_encoders = {}
for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder
print("\n Categorical Encoding Completed")

X = df.drop("Price",axis=1)
y = df["Price"]

X_train, X_test, y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

print("\nTraining Model...")
model.fit(X_train, y_train)
print("Training Completed")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print("\n==============================")
print("Model Performance")
print("================================")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RSME: {rmse:.2f}")
print(f"R2 Score : {r2:.4f}")

with open("car_price_model.pkl","wb") as file:
    pickle.dump(model, file)
print("\n Model Saved : car_price_model.pkl")

with open ("label_encoders.pkl","wb") as file:
    pickle.dump(label_encoders, file)
print("Label Encoders Saved : label_encoders.pkl")


importance = pd.DataFrame({
    "Feature" : X.columns,
    "Importance" : model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n Feature Importance")
print(importance)
print("\n ====================================")
print("Training Completed Successfully")
print("===================================")
