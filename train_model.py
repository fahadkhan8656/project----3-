import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("vehicle_maintenance.csv")

# Create encoders
brand_encoder = LabelEncoder()
car_encoder = LabelEncoder()
issue_encoder = LabelEncoder()

# Encode categorical columns
df["Brand"] = brand_encoder.fit_transform(df["Brand"])
df["Car_Name"] = car_encoder.fit_transform(df["Car_Name"])
df["Issue"] = issue_encoder.fit_transform(df["Issue"])

# Features
X = df[[
    "Brand",
    "Car_Name",
    "Issue",
    "Model_Year",
    "KMs_Driven"
]]

# Target
y = df["Maintenance_Cost"]

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save model and encoders
joblib.dump(model, "vehicle_model.pkl")
joblib.dump(brand_encoder, "brand_encoder.pkl")
joblib.dump(car_encoder, "car_encoder.pkl")
joblib.dump(issue_encoder, "issue_encoder.pkl")

print("Model trained successfully!")
print("Files created:")
print("- vehicle_model.pkl")
print("- brand_encoder.pkl")
print("- car_encoder.pkl")
print("- issue_encoder.pkl")